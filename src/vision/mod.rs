//! Local computer-vision understanding of photos and videos.
//!
//! Everything here runs locally (ONNX / pure code) — no cloud, no LLM APIs, no
//! network at index time. The feature is default OFF everywhere (see
//! [`VisionMode::Off`]); consumers see zero behaviour change until a caller opts
//! in via `JobRequest.vision` / `index --vision` under the `serve --vision-max`
//! cap.
//!
//! Ownership (VISION-SPEC section 5): V1 owns `mod.rs` + `types.rs`
//! (orchestrator, decode hardening, model registry, tier seams). The per-tier
//! bodies live in sibling files owned by V2–V5 and land as no-op stubs here so
//! the skeleton compiles and the off-path stays byte-identical.

mod caption;
mod clip;
mod detector;
mod exif;
pub mod faces;
mod phash;
mod quality;
mod video;

pub mod types;

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use image::{DynamicImage, ImageError, ImageReader, Limits};

use crate::config::{Config, VisionConfig};

pub use phash::hamming;
pub use types::{
    ExifInfo, FaceDetection, GpsCoord, ObjectDetection, TagScore, VisionMode, VisionResult,
};

/// Image extensions the vision pipeline analyses — mirrors `extract::IMAGE_EXTS`.
pub const VISION_IMAGE_EXTS: &[&str] = &[
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif",
];
/// Video extensions the vision pipeline analyses — mirrors `extract::VIDEO_EXTS`.
pub const VISION_VIDEO_EXTS: &[&str] = &[".mkv", ".mp4", ".mov", ".m4v", ".avi", ".webm"];

/// A model artifact fetched only by `fetch-data --vision`. URLs and checksums
/// are pinned in-source next to each other: the tags-tier detector
/// (RF-DETR-Nano) is pinned below; the captions-tier Florence-2 files are left
/// unpinned (`None`) because that tier ships as an unsupported stub in v1.
/// `None` means "not yet pinned" — `fetch-data --vision` skips it with a note,
/// and the verify-after-download path only runs once a real hash is present.
#[derive(Debug, Clone, Copy)]
pub struct VisionModel {
    /// Lowest tier that needs this file.
    pub tier: VisionMode,
    /// Path relative to `<data_dir>/vision`.
    pub relative: &'static str,
    /// HTTPS download URL, or `None` until pinned.
    pub url: Option<&'static str>,
    /// Pinned SHA-256 (lowercase hex), or `None` until pinned.
    pub sha256: Option<&'static str>,
    /// License note for docs/VISION.md.
    pub license: &'static str,
    /// Whether the file backs an OPT-IN sub-model its tier does not require.
    ///
    /// Required artifacts (`false`) are what [`missing_models`] reports, what
    /// the submit pre-flight rejects a job over, and what `fetch-data --vision`
    /// downloads. Optional ones (`true` — today only the face pair) are none of
    /// those things: their absence makes one capability absent instead of
    /// failing a job, and staging them takes its own explicit
    /// `fetch-data --faces`. The flag exists so ONE registry still pins every
    /// artifact's URL, hash and license in one place — the property this file's
    /// security model rests on — without a privacy-sensitive model riding into a
    /// deployment on the back of a tier it does not belong to.
    pub optional: bool,
}

/// Downloadable vision models with pinned checksums.
///
/// CLIP ViT-B/32 (tags tier) is supplied by fastembed's own cache, not fetched
/// here. The tag vocabulary (`data/vision-tags.txt`) ships in the repo. The
/// detector artifact + SHA-256 are pinned below (VISION-SPEC AMENDMENT
/// 2026-07-19: RF-DETR-Nano, Apache-2.0 — not the AGPL Ultralytics YOLO). The
/// Florence-2 captions files stay unpinned while that tier is the v1 unsupported
/// stub; keep the verify-after-download logic in `main.rs::fetch_data` intact.
pub const VISION_MODELS: &[VisionModel] = &[
    VisionModel {
        tier: VisionMode::Tags,
        relative: detector::RFDETR_NANO_ONNX,
        // onnx-community/rfdetr_nano-ONNX fp32 export (DETR-style, no NMS).
        // Quantized alternative (~28.8 MB, identical f32 I/O):
        //   .../resolve/main/onnx/model_quantized.onnx
        //   sha256 2981aa0a57781c0f1a5be171c3fded504c38fe8611326bc5e3d0f8a2e9a57085
        url: Some(
            "https://huggingface.co/onnx-community/rfdetr_nano-ONNX/resolve/main/onnx/model.onnx",
        ),
        sha256: Some("9cbac6b11ce34a03034e4d5a24cfac5f18632fd6761d1311dd640232088d7fee"),
        license: "Apache-2.0 (RF-DETR-Nano, onnx-community/rfdetr_nano-ONNX)",
        optional: false,
    },
    VisionModel {
        tier: VisionMode::Captions,
        relative: "florence2-base-encoder.onnx",
        url: None,    // Captions ships as the v1 unsupported stub (see caption.rs).
        sha256: None, // A live decode needs a multi-graph export; deferred to V6.
        license: "MIT (Microsoft Florence-2-base)",
        optional: false,
    },
    VisionModel {
        tier: VisionMode::Captions,
        relative: "florence2-base-decoder.onnx",
        url: None,    // Captions ships as the v1 unsupported stub (see caption.rs).
        sha256: None, // A live decode needs a multi-graph export; deferred to V6.
        license: "MIT (Microsoft Florence-2-base)",
        optional: false,
    },
    // The opt-in face pair (plan.md §7 B1). Pinned at the opencv_zoo 4.10.0 tag
    // rather than `main` so the URL is immutable as well as hash-checked; both
    // hashes were taken from those exact bytes on 2026-07-26. `optional: true`
    // keeps them out of every tier gate — see [`VisionModel::optional`].
    VisionModel {
        tier: VisionMode::Tags,
        relative: faces::YUNET_ONNX,
        url: Some(
            "https://github.com/opencv/opencv_zoo/raw/4.10.0/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        sha256: Some("8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
        license: "Apache-2.0 (YuNet 2023mar, opencv/opencv_zoo)",
        optional: true,
    },
    VisionModel {
        tier: VisionMode::Tags,
        relative: faces::SFACE_ONNX,
        url: Some(
            "https://github.com/opencv/opencv_zoo/raw/4.10.0/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
        sha256: Some("0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"),
        license: "Apache-2.0 (SFace 2021dec, opencv/opencv_zoo)",
        optional: true,
    },
];

/// True when `ext` (lowercase, with leading dot) is an analysable image.
pub fn is_image_ext(ext: &str) -> bool {
    VISION_IMAGE_EXTS.contains(&ext)
}

/// True when `ext` (lowercase, with leading dot) is an analysable video.
pub fn is_video_ext(ext: &str) -> bool {
    VISION_VIDEO_EXTS.contains(&ext)
}

/// True when `ext` is an image or a video the vision pipeline handles.
pub fn is_vision_ext(ext: &str) -> bool {
    is_image_ext(ext) || is_video_ext(ext)
}

/// Incremental change-detection rule: reprocess a vision-eligible file when the
/// requested tier is higher than the one recorded for it (`recorded == None`
/// means no `vision` row yet). Lowering the tier never triggers reprocessing.
pub fn needs_vision_reprocess(
    requested: VisionMode,
    recorded: Option<VisionMode>,
    ext: &str,
) -> bool {
    requested != VisionMode::Off
        && is_vision_ext(ext)
        && requested > recorded.unwrap_or(VisionMode::Off)
}

/// The model files required for `requested` that are missing under `models_dir`.
/// An empty vec means the tier's pinned files are all present. Used by the submit
/// pre-flight to fail a job as a whole rather than surprising the caller per file.
///
/// Optional artifacts ([`VisionModel::optional`]) are excluded by construction:
/// they are not required by any tier, so an unstaged face pair must never make a
/// `tags` job unsubmittable.
pub fn missing_models(models_dir: &Path, requested: VisionMode) -> Vec<PathBuf> {
    VISION_MODELS
        .iter()
        .filter(|model| !model.optional && requested.includes(model.tier))
        .map(|model| models_dir.join(model.relative))
        .filter(|path| !path.is_file())
        .collect()
}

/// Every vision prerequisite still missing for `requested` under `models_dir`, as
/// human-readable strings: the pinned model files absent on disk (see
/// [`missing_models`]) plus the CLIP fastembed cache when a tags/captions tier
/// needs it but `fetch-data --vision` never staged it. Empty ⇒ the tier can run
/// fully offline. The submit/job pre-flight fails the job as a whole on any entry
/// rather than surprising the caller per file — and, for CLIP, rather than letting
/// fastembed silently auto-download ~350 MB mid-job (VISION-SPEC §1: models are
/// fetched ONLY by `fetch-data --vision`, no auto-download anywhere else).
pub fn missing_vision_prereqs(models_dir: &Path, requested: VisionMode) -> Vec<String> {
    let mut missing: Vec<String> = missing_models(models_dir, requested)
        .into_iter()
        .map(|path| path.display().to_string())
        .collect();
    if requested.includes(VisionMode::Tags) && !clip::cache_present(models_dir) {
        missing
            .push("CLIP encoder cache (Qdrant/clip-ViT-B-32; run fetch-data --vision)".to_string());
    }
    missing
}

/// Pinned model files present under `models_dir` whose bytes do NOT match the
/// SHA-256 pinned in [`VISION_MODELS`] — a truncated, corrupt, or swapped model.
/// Empty ⇒ every present, pinned model verifies. Streams the file so a ~100 MB
/// ONNX blob is not slurped into memory; used by the job-time (blocking)
/// pre-flight as the integrity half of the spec's "verify the model files exist
/// and hash-match" (absent files are reported by [`missing_models`]).
pub fn corrupt_models(models_dir: &Path, requested: VisionMode) -> Vec<PathBuf> {
    VISION_MODELS
        .iter()
        .filter(|model| !model.optional && requested.includes(model.tier))
        .filter_map(|model| {
            let expected = model.sha256?;
            let path = models_dir.join(model.relative);
            let actual = sha256_file(&path).ok()?;
            (!actual.eq_ignore_ascii_case(expected)).then_some(path)
        })
        .collect()
}

/// Vision tiers (excluding `off`) that can actually run under `models_dir` right
/// now: every model file the tier needs is present AND hash-verified, and any
/// CLIP cache it relies on is staged. Reuses the submit/job pre-flight helpers
/// ([`missing_vision_prereqs`] + [`corrupt_models`]) so the `GET /settings`
/// capability report and the job gate never disagree. Returned ascending
/// (`meta < tags < captions`); `meta` is pure code so it is always ready.
pub fn available_tiers(models_dir: &Path) -> Vec<VisionMode> {
    [VisionMode::Meta, VisionMode::Tags, VisionMode::Captions]
        .into_iter()
        .filter(|tier| tier_available(models_dir, *tier))
        .collect()
}

/// Whether `tier` can run under `models_dir`: nothing it needs is missing and
/// nothing present is corrupt (existence + pinned-hash check, per SETTINGS-SPEC
/// §2). `off`/`meta` need no models, so they are always available.
pub fn tier_available(models_dir: &Path, tier: VisionMode) -> bool {
    missing_vision_prereqs(models_dir, tier).is_empty()
        && corrupt_models(models_dir, tier).is_empty()
}

/// Whether one pinned [`VISION_MODELS`] artifact named by `relative` is present
/// under `models_dir` and — when a SHA-256 is pinned — matches it. An unpinned
/// artifact (the Florence stub) counts as ready on mere presence, mirroring
/// `fetch-data`'s skip-verification-when-unpinned posture. CLIP is not a
/// `VISION_MODELS` entry (it lives in the fastembed cache) — see
/// [`tagger_present`].
fn model_ready(models_dir: &Path, relative: &str) -> bool {
    let Some(model) = VISION_MODELS
        .iter()
        .find(|model| model.relative == relative)
    else {
        return false;
    };
    let path = models_dir.join(model.relative);
    if !path.is_file() {
        return false;
    }
    match model.sha256 {
        Some(expected) => sha256_file(&path)
            .map(|actual| actual.eq_ignore_ascii_case(expected))
            .unwrap_or(false),
        None => true,
    }
}

/// Whether the object detector (`nano` → RF-DETR-Nano) is present and hash-valid
/// under `models_dir`. Backs the `detectors[].present` flag in `GET /settings`.
pub fn detector_present(models_dir: &Path) -> bool {
    model_ready(models_dir, detector::RFDETR_NANO_ONNX)
}

/// Whether the zero-shot tagger (`clip`) cache is staged under `models_dir`.
/// Backs the `taggers[].present` flag in `GET /settings`.
pub fn tagger_present(models_dir: &Path) -> bool {
    clip::cache_present(models_dir)
}

/// Whether the opt-in face pair (`yunet-sface`) is staged AND hash-valid under
/// `models_dir`. Backs the `faces[].present` flag in `GET /settings`, which is
/// how an app learns to grey the control out instead of offering a capability
/// this box does not have.
///
/// Existence alone is never enough here, for the same reason as the detector:
/// the whole security model is that the bytes running inference are the bytes
/// whose hash is pinned in this file.
pub fn faces_present(models_dir: &Path) -> bool {
    model_ready(models_dir, faces::YUNET_ONNX) && model_ready(models_dir, faces::SFACE_ONNX)
}

/// Face model files present under `models_dir` whose bytes do NOT match their
/// pinned SHA-256. Empty ⇒ what is staged is what was pinned (or nothing is
/// staged, which is the capability-absent case, not a corruption).
///
/// Used by the job-time pre-flight for the one case that must NOT degrade
/// silently: a face model that is present but tampered with, truncated, or
/// swapped. An ABSENT pair is a choice not to have the capability; a WRONG pair
/// is a claim about a person's identity computed by bytes nobody vouched for, so
/// it fails the job the way a corrupt detector does.
pub fn corrupt_face_models(models_dir: &Path) -> Vec<PathBuf> {
    [faces::YUNET_ONNX, faces::SFACE_ONNX]
        .into_iter()
        .filter(|relative| models_dir.join(relative).is_file())
        .filter(|relative| !model_ready(models_dir, relative))
        .map(|relative| models_dir.join(relative))
        .collect()
}

/// Incremental change-detection rule for the faces sub-tier, the companion to
/// [`needs_vision_reprocess`].
///
/// The tier ladder cannot see this on its own: turning faces on changes nothing
/// about the requested tier, so a corpus already at `tags` would be skipped
/// wholesale and the operator would watch a faces job do nothing. `recorded` is
/// the file's `vision.faces_model` — `None` when no face model has ever scanned
/// it — so a file is reworked exactly when the enabled pair is not the pair
/// already recorded for it. Turning faces OFF never reprocesses and never
/// deletes: like a lowered vision tier, it is non-destructive.
pub fn needs_faces_reprocess(enabled: Option<&str>, recorded: Option<&str>, ext: &str) -> bool {
    match enabled {
        Some(model) => is_vision_ext(ext) && recorded != Some(model),
        None => false,
    }
}

/// Whether the captioner (`florence2`) files are all present under `models_dir`.
/// The Florence artifacts are unpinned in v1, so this is a pure existence check
/// (both files) — expected `false` until the captions tier ships. Backs the
/// `captioners[].present` flag in `GET /settings`.
pub fn captioner_present(models_dir: &Path) -> bool {
    let florence: Vec<&str> = VISION_MODELS
        .iter()
        .filter(|model| model.tier == VisionMode::Captions)
        .map(|model| model.relative)
        .collect();
    !florence.is_empty()
        && florence
            .iter()
            .all(|relative| model_ready(models_dir, relative))
}

/// Stream a file's lowercase-hex SHA-256 without loading it all into memory
/// (mirrors the chunked hashing in `pipeline::sha1`).
fn sha256_file(path: &Path) -> std::io::Result<String> {
    use std::io::Read;

    use sha2::{Digest, Sha256};
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 1 << 16];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

/// Warm the CLIP encoder cache under `models_dir` by loading both encoders once —
/// the ONLY sanctioned network fetch of CLIP (VISION-SPEC §1). Called by
/// `fetch-data --vision` so the index-time path later loads CLIP from the local
/// cache and never reaches the network. Safe to re-run: fastembed's cache-first
/// loader skips the download when the files are already present.
pub fn prefetch_clip(models_dir: &Path) -> Result<()> {
    clip::prefetch(models_dir)
}

/// Why image decoding failed, mapped onto the `vision.error` values the spec
/// pins (`decode-limit` for over-limit inputs, `decode-error` for corrupt ones).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DecodeFailure {
    Limit,
    Error,
}

impl DecodeFailure {
    fn as_str(self) -> &'static str {
        match self {
            Self::Limit => "decode-limit",
            Self::Error => "decode-error",
        }
    }
}

/// Decode an image with explicit resource limits. Over-limit inputs (past
/// `cfg.max_pixels` or `cfg.max_alloc_bytes`) map to [`DecodeFailure::Limit`];
/// unreadable/corrupt inputs map to [`DecodeFailure::Error`].
fn decode_image(path: &Path, cfg: &VisionConfig) -> Result<DynamicImage, DecodeFailure> {
    // Cheap dimension probe first so an enormous header is rejected before we
    // ever allocate a pixel buffer.
    if let Some((width, height)) = ImageReader::open(path)
        .ok()
        .and_then(|reader| reader.with_guessed_format().ok())
        .and_then(|reader| reader.into_dimensions().ok())
    {
        if u64::from(width) * u64::from(height) > cfg.max_pixels {
            return Err(DecodeFailure::Limit);
        }
    }
    let mut reader = ImageReader::open(path)
        .and_then(|reader| reader.with_guessed_format())
        .map_err(|_| DecodeFailure::Error)?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(cfg.max_alloc_bytes);
    reader.limits(limits);
    match reader.decode() {
        Ok(image) => Ok(image),
        Err(ImageError::Limits(_)) => Err(DecodeFailure::Limit),
        Err(_) => Err(DecodeFailure::Error),
    }
}

/// Run one tier closure, recording only the first tier error into the result so
/// a later tier's success does not clobber an earlier failure. Returns whether
/// the tier succeeded so the caller can decide whether to advance the recorded
/// `mode` (an environment failure — missing model, CLIP init error — must not
/// stamp the requested tier, or the resume rule would strand the file forever).
fn run_tier<F>(result: &mut VisionResult, tier: F) -> bool
where
    F: FnOnce(&mut VisionResult) -> Result<()>,
{
    match tier(result) {
        Ok(()) => true,
        Err(error) => {
            if result.error.is_none() {
                result.error = Some(format!("{error:#}"));
            }
            false
        }
    }
}

/// The shared vision entry point: decode hardening plus the per-tier seams.
/// Cloned per job (constructed from [`Config`]) and shared across the rayon
/// workers, like the OCR and transcription handles.
pub struct VisionAnalyzer {
    cfg: VisionConfig,
    models_dir: PathBuf,
    /// Whether the opt-in faces sub-tier will actually run: requested by config
    /// AND staged on this box. Resolved ONCE here rather than per file so the
    /// answer is stable for a whole job (a job cannot half-scan a corpus because
    /// the models appeared mid-run) and so the off-path pays nothing — the probe
    /// is skipped entirely unless a job asked for faces.
    faces_ready: bool,
}

impl VisionAnalyzer {
    pub fn new(config: &Config) -> Result<Self> {
        let models_dir = config.vision_models_dir();
        let faces_ready = config.vision.max != VisionMode::Off
            && config.vision.faces_enabled()
            && faces::available(&models_dir);
        if config.vision.faces_enabled() && config.vision.max != VisionMode::Off && !faces_ready {
            // Capability absent, not an error: say so once, then run the rest of
            // the tier. Silence here is what would be dishonest — an operator who
            // asked for faces and got an empty table deserves the reason.
            tracing::warn!(
                models_dir = %models_dir.display(),
                "faces requested but the yunet/sface pair is not staged; \
                 skipping faces (run `llm-index fetch-data --faces`)"
            );
        }
        Ok(Self {
            cfg: config.vision.clone(),
            models_dir,
            faces_ready,
        })
    }

    /// The resolved directory model files are expected under.
    pub fn models_dir(&self) -> &Path {
        &self.models_dir
    }

    /// The face model pair this job will actually scan with, or `None` when the
    /// faces sub-tier is off or unstaged. Feeds [`needs_faces_reprocess`], so
    /// the resume rule and the run agree on one answer.
    pub fn faces_model(&self) -> Option<&'static str> {
        self.faces_ready.then_some(faces::FACE_MODEL_ID)
    }

    /// Analyse a still image up to `mode`. Never panics: decode failures and
    /// tier errors are recorded in [`VisionResult::error`] and the file
    /// continues. `mode == off` returns an empty (inert) result.
    pub fn analyze_image(&self, path: &Path, mode: VisionMode) -> VisionResult {
        let started = Instant::now();
        let mut result = VisionResult::default();
        if mode == VisionMode::Off {
            return result;
        }
        let image = match decode_image(path, &self.cfg) {
            Ok(image) => image,
            Err(failure) => {
                // Record the requested tier so a broken file is not re-attempted
                // every resume unless it actually changes on disk.
                result.mode = mode;
                result.error = Some(failure.as_str().to_string());
                result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
                return result;
            }
        };
        result.width = Some(image.width());
        result.height = Some(image.height());
        if mode.includes(VisionMode::Meta) {
            run_tier(&mut result, |out| exif::read(path, out));
            run_tier(&mut result, |out| phash::fill(&image, out));
            run_tier(&mut result, |out| quality::fill(&image, out));
            result.mode = VisionMode::Meta;
        }
        if mode.includes(VisionMode::Tags) {
            // Run each model-backed sub-tier, then only advance the recorded `mode`
            // when the ones that ran actually succeeded. The detector/tagger
            // toggles (`vision_opts.detector`/`tagger`) gate them: an `off` toggle
            // SKIPS that sub-tier entirely (no model load, no cost) and counts as
            // "ok" so it doesn't hold `mode` back at meta. A missing detector model
            // or failed CLIP init records the error but leaves `mode` at `meta`, so
            // a later `--resume` retries the file (see `needs_vision_reprocess`)
            // instead of treating the whole corpus as already at the tags tier.
            let detector_ok = if self.cfg.detector_enabled() {
                run_tier(&mut result, |out| {
                    detector::fill(&image, &self.models_dir, &self.cfg, out)
                })
            } else {
                true
            };
            let clip_ok = if self.cfg.tagger_enabled() {
                run_tier(&mut result, |out| {
                    clip::fill(&image, &self.models_dir, &self.cfg, out)
                })
            } else {
                true
            };
            // Faces is the third model-backed sub-tier and behaves like the
            // other two — except that "not staged" is not a failure here, it is
            // `faces_ready == false`, which counts as ok exactly like an `off`
            // toggle. Only a real inference failure records an error and holds
            // `mode` back so a later resume retries the file.
            let faces_ok = if self.faces_ready {
                run_tier(&mut result, |out| {
                    faces::fill(&image, &self.models_dir, &self.cfg, out)
                })
            } else {
                true
            };
            if detector_ok && clip_ok && faces_ok {
                result.mode = VisionMode::Tags;
            }
        }
        if mode.includes(VisionMode::Captions) {
            let caption_ok = if self.cfg.captioner_enabled() {
                run_tier(&mut result, |out| {
                    caption::fill(&image, &self.models_dir, &self.cfg, out)
                })
            } else {
                true
            };
            if caption_ok {
                result.mode = VisionMode::Captions;
            }
        }
        result.elapsed_ms = Some(started.elapsed().as_millis() as u64);
        result
    }

    /// Analyse a video up to `mode` (keyframes + per-frame tags + transcript
    /// merge). Delegates to the V4-owned `video` module.
    pub fn analyze_video(&self, path: &Path, mode: VisionMode) -> VisionResult {
        if mode == VisionMode::Off {
            return VisionResult::default();
        }
        video::analyze(path, &self.models_dir, &self.cfg, mode, self.faces_ready)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer_with(max_pixels: u64) -> VisionAnalyzer {
        let mut config = Config::default();
        config.vision.max_pixels = max_pixels;
        config.finalize();
        VisionAnalyzer::new(&config).unwrap()
    }

    #[test]
    fn change_detection_upgrade_rule() {
        // Higher requested tier than recorded -> reprocess.
        assert!(needs_vision_reprocess(
            VisionMode::Tags,
            Some(VisionMode::Meta),
            ".jpg"
        ));
        // No row yet -> reprocess an eligible file.
        assert!(needs_vision_reprocess(VisionMode::Meta, None, ".png"));
        // Same tier already recorded -> skip.
        assert!(!needs_vision_reprocess(
            VisionMode::Tags,
            Some(VisionMode::Tags),
            ".jpg"
        ));
        // Downgrade -> never reprocess.
        assert!(!needs_vision_reprocess(
            VisionMode::Meta,
            Some(VisionMode::Captions),
            ".jpg"
        ));
        // Off requested -> never.
        assert!(!needs_vision_reprocess(VisionMode::Off, None, ".jpg"));
        // Non-vision extension -> never, even with no row.
        assert!(!needs_vision_reprocess(VisionMode::Tags, None, ".txt"));
        // Videos are eligible too.
        assert!(needs_vision_reprocess(VisionMode::Tags, None, ".mp4"));
    }

    #[test]
    fn missing_models_reports_by_tier() {
        let dir = tempfile::tempdir().unwrap();
        // Meta needs no model files.
        assert!(missing_models(dir.path(), VisionMode::Meta).is_empty());
        assert!(missing_models(dir.path(), VisionMode::Off).is_empty());
        // Tags needs the object-detector model, which is absent here.
        let tags = missing_models(dir.path(), VisionMode::Tags);
        assert_eq!(tags.len(), 1);
        assert!(tags[0].ends_with("rf-detr-nano.onnx"));
        // Captions additionally needs the two Florence files.
        assert_eq!(missing_models(dir.path(), VisionMode::Captions).len(), 3);
    }

    #[test]
    fn missing_vision_prereqs_covers_detector_and_clip_cache() {
        let dir = tempfile::tempdir().unwrap();
        // Meta is pure code — no model files, no CLIP cache.
        assert!(missing_vision_prereqs(dir.path(), VisionMode::Meta).is_empty());
        // Tags over an empty models dir reports BOTH the detector file and the
        // un-staged CLIP cache, so the pre-flight blocks a silent mid-job fetch.
        let tags = missing_vision_prereqs(dir.path(), VisionMode::Tags);
        assert_eq!(tags.len(), 2, "{tags:?}");
        assert!(tags.iter().any(|entry| entry.contains("rf-detr-nano.onnx")));
        assert!(tags.iter().any(|entry| entry.contains("CLIP")));
    }

    #[test]
    fn decode_limit_and_error_paths() {
        let dir = tempfile::tempdir().unwrap();

        // A valid tiny image decodes and reports its dimensions.
        let good = dir.path().join("good.png");
        image::RgbImage::new(4, 3).save(&good).unwrap();
        let ok = analyzer_with(250_000_000).analyze_image(&good, VisionMode::Meta);
        assert_eq!(ok.error, None);
        assert_eq!((ok.width, ok.height), (Some(4), Some(3)));
        assert_eq!(ok.mode, VisionMode::Meta);

        // The same image under a 1-pixel cap trips the decode-limit path.
        let limited = analyzer_with(1).analyze_image(&good, VisionMode::Meta);
        assert_eq!(limited.error.as_deref(), Some("decode-limit"));
        assert_eq!(limited.mode, VisionMode::Meta);

        // Garbage bytes with an image extension trip the decode-error path.
        let bad = dir.path().join("bad.png");
        std::fs::write(&bad, b"definitely not a PNG").unwrap();
        let broken = analyzer_with(250_000_000).analyze_image(&bad, VisionMode::Tags);
        assert_eq!(broken.error.as_deref(), Some("decode-error"));
        assert_eq!(broken.mode, VisionMode::Tags);
    }

    #[test]
    fn available_tiers_gate_on_verified_model_files() {
        let dir = tempfile::tempdir().unwrap();
        // Empty models dir: only the pure-code `meta` tier is available; `tags`
        // and `captions` are gated out (detector + CLIP cache absent).
        assert_eq!(available_tiers(dir.path()), vec![VisionMode::Meta]);
        assert!(tier_available(dir.path(), VisionMode::Meta));
        assert!(!tier_available(dir.path(), VisionMode::Tags));

        // Planting a file with the detector's name but the wrong bytes does NOT
        // make `tags` available — the pinned-hash check rejects it (and the CLIP
        // cache is still absent). Existence alone is never enough.
        std::fs::write(
            dir.path().join(detector::RFDETR_NANO_ONNX),
            b"not the real detector",
        )
        .unwrap();
        assert!(!detector_present(dir.path()));
        assert!(!tier_available(dir.path(), VisionMode::Tags));
        assert_eq!(available_tiers(dir.path()), vec![VisionMode::Meta]);
    }

    #[test]
    fn captioner_present_flips_with_its_model_files() {
        let dir = tempfile::tempdir().unwrap();
        // The two (unpinned) Florence files gate the captioner flag by existence.
        let florence: Vec<&str> = VISION_MODELS
            .iter()
            .filter(|model| model.tier == VisionMode::Captions)
            .map(|model| model.relative)
            .collect();
        assert!(!captioner_present(dir.path()));
        for relative in &florence {
            std::fs::write(dir.path().join(relative), b"stub").unwrap();
        }
        assert!(captioner_present(dir.path()));
        // Removing any one required model file flips the reported capability off.
        std::fs::remove_file(dir.path().join(florence[0])).unwrap();
        assert!(!captioner_present(dir.path()));
    }

    #[test]
    fn disabled_sub_models_skip_execution() {
        // detector + tagger both `off`: the tags tier then needs no model, so
        // analyze_image over an EMPTY models dir completes without error and still
        // records the tags tier — proving the toggles gate execution rather than
        // being accepted-and-ignored (the pre-fix code tried to load the absent
        // detector, errored, and downgraded the file to meta).
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.png");
        image::RgbImage::new(4, 3).save(&good).unwrap();
        let mut config = Config::default();
        config.vision.detector = "off".into();
        config.vision.tagger = "off".into();
        config.finalize();
        let analyzer = VisionAnalyzer::new(&config).unwrap();
        let result = analyzer.analyze_image(&good, VisionMode::Tags);
        assert_eq!(result.error, None, "{:?}", result.error);
        assert_eq!(result.mode, VisionMode::Tags);
        assert!(result.objects.is_empty());
        assert!(result.tags.is_empty());
    }

    /// The whole point of the faces opt-in: an unstaged box degrades, it does
    /// not fail. Faces are ASKED for over an empty models dir, and the tags tier
    /// still completes — no error, no held-back tier, no faces, and no
    /// `faces_model` stamp (so the file stays eligible once the pair is staged).
    #[test]
    fn absent_face_models_are_a_missing_capability_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.png");
        image::RgbImage::new(4, 3).save(&good).unwrap();
        let mut config = Config::default();
        config.vision.detector = "off".into();
        config.vision.tagger = "off".into();
        config.vision.faces = faces::FACE_MODEL_ID.into();
        config.vision.max = VisionMode::Tags;
        config.finalize();
        let analyzer = VisionAnalyzer::new(&config).unwrap();
        assert_eq!(analyzer.faces_model(), None, "capability absent");
        let result = analyzer.analyze_image(&good, VisionMode::Tags);
        assert_eq!(result.error, None, "{:?}", result.error);
        assert_eq!(result.mode, VisionMode::Tags);
        assert!(result.faces.is_empty());
        assert_eq!(result.faces_model, None);
    }

    /// And an unstaged pair never reaches the tier gates: a `tags` job on a box
    /// with no face models is exactly as submittable as before this feature —
    /// the optional entries are invisible to `missing_models`,
    /// `missing_vision_prereqs`, `corrupt_models` and `available_tiers`.
    #[test]
    fn optional_face_models_are_outside_every_tier_gate() {
        let dir = tempfile::tempdir().unwrap();
        // Only the required tags artifact is reported missing — not the pair.
        let missing = missing_models(dir.path(), VisionMode::Captions);
        assert_eq!(missing.len(), 3, "{missing:?}");
        assert!(!missing
            .iter()
            .any(|path| path.ends_with(faces::YUNET_ONNX) || path.ends_with(faces::SFACE_ONNX)));

        // Stage the required tags prerequisites; `tags` becomes available with
        // no face model anywhere near the box.
        let staged = tempfile::tempdir().unwrap();
        assert!(!faces_present(staged.path()));
        assert!(
            corrupt_face_models(staged.path()).is_empty(),
            "absent is not corrupt"
        );

        // A present-but-wrong face model is neither available nor silent: the
        // presence flag stays false AND it is reported corrupt, which is what
        // fails a faces job instead of running unvouched-for bytes.
        std::fs::write(staged.path().join(faces::YUNET_ONNX), b"not yunet").unwrap();
        assert!(!faces_present(staged.path()));
        assert_eq!(corrupt_face_models(staged.path()).len(), 1);
        // …and it still does not touch the tags/captions gates.
        assert!(corrupt_models(staged.path(), VisionMode::Captions).is_empty());
        assert!(missing_vision_prereqs(staged.path(), VisionMode::Tags)
            .iter()
            .all(|entry| !entry.contains("yunet") && !entry.contains("sface")));
    }

    #[test]
    fn faces_change_detection_reworks_only_what_a_new_pair_has_not_scanned() {
        // Never scanned -> rework.
        assert!(needs_faces_reprocess(Some("yunet-sface"), None, ".jpg"));
        // Scanned by this exact pair -> leave alone, even with no faces stored.
        assert!(!needs_faces_reprocess(
            Some("yunet-sface"),
            Some("yunet-sface"),
            ".jpg"
        ));
        // Scanned by a different pair -> rework (a model change is a new answer).
        assert!(needs_faces_reprocess(
            Some("yunet-sface"),
            Some("buffalo_l"),
            ".jpg"
        ));
        // Faces off or unstaged -> never, whatever is recorded.
        assert!(!needs_faces_reprocess(None, None, ".jpg"));
        assert!(!needs_faces_reprocess(None, Some("yunet-sface"), ".jpg"));
        // Non-vision extensions are not face-eligible; videos are.
        assert!(!needs_faces_reprocess(Some("yunet-sface"), None, ".txt"));
        assert!(needs_faces_reprocess(Some("yunet-sface"), None, ".mp4"));
    }

    #[test]
    fn off_mode_is_inert() {
        let dir = tempfile::tempdir().unwrap();
        let good = dir.path().join("good.png");
        image::RgbImage::new(2, 2).save(&good).unwrap();
        let result = analyzer_with(250_000_000).analyze_image(&good, VisionMode::Off);
        assert_eq!(result.mode, VisionMode::Off);
        assert!(result.content_block().is_none());
        assert_eq!(result.width, None);
    }
}
