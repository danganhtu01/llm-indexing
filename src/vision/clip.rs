//! Tags tier (V3): CLIP image embedding + zero-shot tag scoring (fastembed).
//!
//! Uses the CLIP ViT-B/32 image encoder and its paired text encoder, both
//! shipped as fastembed built-ins (`ImageEmbeddingModel::ClipVitB32` /
//! `EmbeddingModel::ClipVitB32`, the Qdrant ONNX exports). The tag vocabulary
//! (`data/vision-tags.txt`, embedded at compile time) is encoded once per
//! process by the text encoder and cached; each image is scored against it by
//! cosine similarity. Model weights come from fastembed's own cache
//! (`<data_dir>/vision`), consistent with the text embedder in `embedding.rs` —
//! no fetch-data entry and no `ort` session of our own here.

use std::path::Path;
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result};
use fastembed::{
    EmbeddingModel, ImageEmbedding, ImageEmbeddingModel, ImageInitOptions, TextEmbedding,
    TextInitOptions,
};
use image::DynamicImage;

use super::types::{TagScore, VisionResult};
use crate::config::VisionConfig;

/// Value stored in `vision.embedding_model` for the CLIP image vector.
pub const CLIP_MODEL: &str = "clip-ViT-B-32";

/// Curated zero-shot vocabulary, embedded at build time so the tag set is fixed
/// with the binary and needs no runtime data file (parsed in [`vocabulary`]).
const TAG_VOCAB: &str = include_str!("../../data/vision-tags.txt");

/// Prompt template wrapped around each label before text encoding — the
/// standard CLIP zero-shot convention, which separates classes better than the
/// bare noun. The stored/reported tag stays the bare label.
const TAG_PROMPT_PREFIX: &str = "a photo of ";

/// Process-wide CLIP handles: the image encoder (behind a `Mutex`, since
/// `embed_images` needs `&mut self`) plus the pre-normalized text embeddings of
/// the vocabulary. Built once and cached in [`engine`].
struct ClipEngine {
    image: Mutex<ImageEmbedding>,
    /// `(label, L2-normalized text embedding)` for every vocabulary entry.
    vocab: Vec<(String, Vec<f32>)>,
}

/// Parse the embedded vocabulary: one label per line, skipping blanks and
/// `#` comments.
fn vocabulary() -> Vec<String> {
    TAG_VOCAB
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(str::to_string)
        .collect()
}

/// ONNX Runtime intra-op threads for the CLIP sessions — every core, capped so
/// a many-file job's rayon workers don't oversubscribe the CPU.
fn intra_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .clamp(1, 8)
}

/// Build both CLIP encoders and pre-embed the vocabulary. The text encoder is
/// only needed for that one-time pass, so it is dropped before returning.
fn build_engine(models_dir: &Path) -> Result<ClipEngine> {
    let threads = intra_threads();
    let mut text = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::ClipVitB32)
            .with_cache_dir(models_dir.to_path_buf())
            .with_show_download_progress(false)
            .with_intra_threads(threads),
    )
    .context("loading CLIP text encoder")?;
    let labels = vocabulary();
    anyhow::ensure!(!labels.is_empty(), "vision tag vocabulary is empty");
    let prompts = labels
        .iter()
        .map(|label| format!("{TAG_PROMPT_PREFIX}{label}"))
        .collect::<Vec<_>>();
    let text_vectors = text
        .embed(prompts, None)
        .context("embedding tag vocabulary")?;
    let vocab = labels
        .into_iter()
        .zip(text_vectors)
        .map(|(label, vector)| (label, normalize(&vector)))
        .collect();

    let image = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_cache_dir(models_dir.to_path_buf())
            .with_show_download_progress(false)
            .with_intra_threads(threads),
    )
    .context("loading CLIP image encoder")?;

    Ok(ClipEngine {
        image: Mutex::new(image),
        vocab,
    })
}

/// fastembed downloads the CLIP image/text encoders from these Hugging Face
/// repos; hf-hub caches each under `<cache_dir>/models--<org>--<repo>` (see
/// `hf_hub::Repo::folder_name`). We stage both into `<data_dir>/vision` via
/// `fetch-data --vision` and check for them before a job runs.
const CLIP_VISION_CACHE: &str = "models--Qdrant--clip-ViT-B-32-vision";
const CLIP_TEXT_CACHE: &str = "models--Qdrant--clip-ViT-B-32-text";

/// Whether both CLIP encoders are already staged in the fastembed cache under
/// `models_dir` (a completed `fetch-data --vision`). When false the pre-flight
/// fails the job rather than letting fastembed's cache-first loader silently
/// reach the network mid-job — the spec forbids any index-time download.
pub(super) fn cache_present(models_dir: &Path) -> bool {
    [CLIP_VISION_CACHE, CLIP_TEXT_CACHE]
        .iter()
        .all(|repo| snapshot_has_model(&models_dir.join(repo)))
}

/// True when an hf-hub repo cache dir holds at least one snapshot with a staged
/// `model.onnx` (the file fastembed loads for both CLIP encoders).
fn snapshot_has_model(repo_dir: &Path) -> bool {
    let Ok(entries) = std::fs::read_dir(repo_dir.join("snapshots")) else {
        return false;
    };
    entries
        .flatten()
        .any(|entry| entry.path().join("model.onnx").is_file())
}

/// Stage both CLIP encoders into the fastembed cache under `models_dir`, loading
/// (and thus downloading, on a cold cache) each once. This is the one place a
/// CLIP network fetch is allowed — driven by `fetch-data --vision`; the vocabulary
/// pass is skipped since we only want the cache side effect. Idempotent: a warm
/// cache loads locally and re-downloads nothing.
pub(super) fn prefetch(models_dir: &Path) -> Result<()> {
    let threads = intra_threads();
    let _text = TextEmbedding::try_new(
        TextInitOptions::new(EmbeddingModel::ClipVitB32)
            .with_cache_dir(models_dir.to_path_buf())
            .with_show_download_progress(true)
            .with_intra_threads(threads),
    )
    .context("staging CLIP text encoder")?;
    let _image = ImageEmbedding::try_new(
        ImageInitOptions::new(ImageEmbeddingModel::ClipVitB32)
            .with_cache_dir(models_dir.to_path_buf())
            .with_show_download_progress(true)
            .with_intra_threads(threads),
    )
    .context("staging CLIP image encoder")?;
    Ok(())
}

/// The cached process-wide CLIP engine, initialised on first use. Only a
/// *successful* build is cached; a transient failure (a network blip staging the
/// model, a momentarily truncated file) caches nothing, so the next job retries
/// instead of the whole resident `serve` process being poisoned until restart. A
/// dedicated init lock serializes construction so the rayon workers don't all
/// build at once on the cold path.
fn engine(models_dir: &Path) -> Result<&'static ClipEngine> {
    static ENGINE: OnceLock<ClipEngine> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(engine) = ENGINE.get() {
        return Ok(engine);
    }
    let _guard = INIT.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(engine) = ENGINE.get() {
        return Ok(engine);
    }
    let engine = build_engine(models_dir).context("CLIP init failed")?;
    Ok(ENGINE.get_or_init(|| engine))
}

/// L2-normalize a vector; a zero vector is returned unchanged (its cosine with
/// anything is defined as 0 by the dot product below).
fn normalize(vector: &[f32]) -> Vec<f32> {
    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|value| value / norm).collect()
    } else {
        vector.to_vec()
    }
}

/// Dot product; equals cosine similarity when both operands are unit vectors.
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Score `image_vector` against the (already normalized) vocabulary and return
/// the labels scoring at least `min_score`, highest first, capped at `top_k`.
/// Pure and model-free so it is unit-testable with stub vectors.
fn score_tags(
    image_vector: &[f32],
    vocab: &[(String, Vec<f32>)],
    min_score: f32,
    top_k: usize,
) -> Vec<TagScore> {
    let query = normalize(image_vector);
    let mut scored = vocab
        .iter()
        .map(|(label, vector)| TagScore {
            tag: label.clone(),
            score: dot(&query, vector),
        })
        .filter(|tag| tag.score >= min_score)
        .collect::<Vec<_>>();
    scored.sort_by(|left, right| right.score.total_cmp(&left.score));
    scored.truncate(top_k);
    scored
}

/// Score zero-shot tags and store the CLIP image embedding into `out`.
pub(super) fn fill(
    image: &DynamicImage,
    models_dir: &Path,
    cfg: &VisionConfig,
    out: &mut VisionResult,
) -> Result<()> {
    let engine = engine(models_dir)?;
    let embedding = {
        let mut model = engine
            .image
            .lock()
            .map_err(|_| anyhow::anyhow!("CLIP image encoder mutex poisoned"))?;
        model
            .embed_images(vec![image.clone()])
            .context("CLIP image embedding")?
            .pop()
            .context("CLIP returned no image embedding")?
    };
    out.tags = score_tags(&embedding, &engine.vocab, cfg.tag_score, cfg.tag_top_k);
    out.dimensions = Some(embedding.len());
    out.embedding_model = Some(CLIP_MODEL.to_string());
    out.embedding = Some(embedding);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocabulary_is_curated_and_clean() {
        let labels = vocabulary();
        // The spec targets ~300 curated labels; guard we did not lose the file.
        assert!(labels.len() > 250, "only {} labels parsed", labels.len());
        // Comments and blanks are stripped; a couple of expected entries exist.
        assert!(labels.iter().all(|label| !label.starts_with('#')));
        assert!(labels.iter().any(|label| label == "receipt"));
        assert!(labels.iter().any(|label| label == "screenshot"));
        assert!(labels.iter().any(|label| label == "whiteboard"));
    }

    #[test]
    fn normalize_produces_unit_vectors() {
        let unit = normalize(&[3.0, 4.0]);
        assert!((dot(&unit, &unit) - 1.0).abs() < 1e-6);
        // A zero vector stays zero rather than dividing by zero.
        assert_eq!(normalize(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    fn vocab(entries: &[(&str, [f32; 3])]) -> Vec<(String, Vec<f32>)> {
        entries
            .iter()
            .map(|(label, vector)| (label.to_string(), normalize(vector)))
            .collect()
    }

    #[test]
    fn score_tags_ranks_by_cosine_and_applies_threshold() {
        // Three orthonormal directions; the query points mostly along "beach".
        let vocab = vocab(&[
            ("beach", [1.0, 0.0, 0.0]),
            ("forest", [0.0, 1.0, 0.0]),
            ("city", [0.0, 0.0, 1.0]),
        ]);
        let query = [0.9, 0.2, 0.0];
        let tags = score_tags(&query, &vocab, 0.15, 8);
        // Ranked best-first: beach dominates, forest second, city (orthogonal) out.
        assert_eq!(tags[0].tag, "beach");
        assert_eq!(tags[1].tag, "forest");
        assert!(tags.iter().all(|tag| tag.tag != "city"));
        // Scores are genuine cosines in [-1, 1] and descending.
        assert!(tags[0].score > tags[1].score);
        assert!(tags[0].score <= 1.0 + 1e-6);
    }

    #[test]
    fn score_tags_caps_at_top_k() {
        let vocab = vocab(&[
            ("a", [1.0, 0.0, 0.0]),
            ("b", [0.9, 0.1, 0.0]),
            ("c", [0.8, 0.2, 0.0]),
            ("d", [0.7, 0.3, 0.0]),
        ]);
        let tags = score_tags(&[1.0, 0.0, 0.0], &vocab, -1.0, 2);
        assert_eq!(tags.len(), 2);
        assert_eq!(tags[0].tag, "a");
        assert_eq!(tags[1].tag, "b");
    }

    #[test]
    fn score_tags_empty_when_nothing_clears_threshold() {
        let vocab = vocab(&[("beach", [1.0, 0.0, 0.0])]);
        // Query orthogonal to the only label -> cosine 0, below threshold.
        assert!(score_tags(&[0.0, 1.0, 0.0], &vocab, 0.5, 8).is_empty());
    }
}
