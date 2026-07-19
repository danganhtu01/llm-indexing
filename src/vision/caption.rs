//! Captions tier (V5): Florence-2-base greedy caption via `ort`.
//!
//! Owned by the V5 worker. This tier is **best-effort** (VISION-SPEC §5) and, as
//! of v1, ships as a clean *unsupported* stub rather than a live decoder — running
//! Florence-2 is not landable inside this worker's ownership boundary:
//!
//! * The encoder/decoder greedy loop needs `ort`'s `Session`/`Tensor` runtime API
//!   and a BART-style tokenizer. fastembed re-exports only
//!   `ort::execution_providers::ExecutionProviderDispatch`, and does not surface
//!   `tokenizers` at all, so both would have to be added as **direct** deps in
//!   `Cargo.toml` — a file this worker does not own (ownership map: `caption.rs`
//!   only).
//! * A working Florence-2-base ONNX export is a *multi-graph* model
//!   (`vision_encoder`, `embed_tokens`, `encoder_model`, `decoder_model_merged`)
//!   plus tokenizer + preprocessor config. The `VISION_MODELS` registry and the
//!   `fetch-data` slots live in `mod.rs` (also not owned here) and pin only two
//!   Florence files, so the artifacts a real decode needs cannot be fetched or
//!   hash-pinned from within `caption.rs`.
//!
//! Because the tier is best-effort, `fill` returns `Err` (recorded in
//! `vision.error` for that file) instead of blocking the release: a `captions`
//! job still stores its `meta`/`tags` output and simply notes captions were
//! unavailable. The deterministic decode primitives the future ONNX wiring needs
//! (task-prompt selection, output post-processing, greedy arg-max) are
//! implemented and unit-tested below so V6 can drop the runtime in without
//! re-deriving them. See docs/VISION.md and the handoff flags for exactly what
//! remains.

use std::path::Path;

use anyhow::{anyhow, Result};
use image::DynamicImage;

use super::types::VisionResult;
use crate::config::VisionConfig;

/// The `vision.error` value recorded when a caller opts into the `captions` tier
/// while it is the v1 unsupported stub.
const UNSUPPORTED: &str = "vision captions unsupported in v1: Florence-2 ONNX \
    greedy decode deferred to V6 — needs `ort`/`tokenizers` as direct deps and a \
    multi-graph model registry outside this worker's ownership; see docs/VISION.md";

/// Produce a one/two-sentence caption into `out.caption`.
///
/// v1: unsupported best-effort stub — returns `Err(UNSUPPORTED)` so the reason is
/// recorded in `vision.error` without discarding the lower tiers already in
/// `out`. Never panics. When V6 wires Florence-2 this becomes: preprocess
/// `image` to the model's input tensor, run the vision encoder + BART
/// encoder/decoder greedy loop (task prompt from [`decode::CaptionTask`], bounded
/// by `cfg.caption_timeout_secs`), post-process with [`decode::clean_caption`],
/// and set `out.caption`.
pub(super) fn fill(
    image: &DynamicImage,
    models_dir: &Path,
    cfg: &VisionConfig,
    out: &mut VisionResult,
) -> Result<()> {
    let _ = (image, models_dir, cfg, out);
    Err(anyhow!(UNSUPPORTED))
}

/// Deterministic Florence-2 decode primitives, proven by the unit tests below and
/// ready for the ONNX pipeline deferred to V6 (see module docs). Kept off `fill`'s
/// path today because the runtime (`ort`/`tokenizers` direct deps) and the
/// multi-graph model registry needed to run Florence-2 live outside this worker's
/// ownership — hence `dead_code` is intentional here.
#[allow(dead_code)]
mod decode {
    /// Florence-2 task prompt. The base model is prompted with a task token; for
    /// this tier we want a prose caption, so the choice is between the short and
    /// the detailed caption tokens. Defaults to the short one/two-sentence form.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub(super) enum CaptionTask {
        /// One/two-sentence caption (`<CAPTION>`).
        #[default]
        Caption,
        /// Longer, more descriptive caption (`<DETAILED_CAPTION>`).
        Detailed,
    }

    impl CaptionTask {
        /// The literal task token fed to Florence-2's text embedding.
        pub(super) fn prompt(self) -> &'static str {
            match self {
                Self::Caption => "<CAPTION>",
                Self::Detailed => "<DETAILED_CAPTION>",
            }
        }
    }

    /// Special/added tokens that must never survive into the stored caption.
    /// Covers Florence-2's BART special tokens and the task tokens it may echo.
    const SPECIAL_TOKENS: &[&str] = &[
        "<s>",
        "</s>",
        "<pad>",
        "<unk>",
        "<mask>",
        "<CAPTION>",
        "<DETAILED_CAPTION>",
        "<MORE_DETAILED_CAPTION>",
    ];

    /// Post-process a decoded Florence-2 string into a clean caption: drop the
    /// special/task tokens the model may emit and collapse runs of whitespace to
    /// single spaces. Deterministic; returns an empty string when nothing but
    /// tokens remain.
    pub(super) fn clean_caption(raw: &str) -> String {
        let mut text = raw.to_string();
        for token in SPECIAL_TOKENS {
            text = text.replace(token, " ");
        }
        text.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    /// The greedy-decode step: index of the highest logit, ties broken toward the
    /// lowest index (deterministic — no RNG, per the spec). `None` for an empty
    /// slice.
    pub(super) fn greedy_argmax(logits: &[f32]) -> Option<usize> {
        logits
            .iter()
            .enumerate()
            .fold(None, |best, (index, &value)| match best {
                Some((_, best_value)) if best_value >= value => best,
                _ => Some((index, value)),
            })
            .map(|(index, _)| index)
    }
}

#[cfg(test)]
mod tests {
    use super::decode::{clean_caption, greedy_argmax, CaptionTask};
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn task_prompts_match_florence_task_tokens() {
        assert_eq!(CaptionTask::default(), CaptionTask::Caption);
        assert_eq!(CaptionTask::Caption.prompt(), "<CAPTION>");
        assert_eq!(CaptionTask::Detailed.prompt(), "<DETAILED_CAPTION>");
    }

    #[test]
    fn clean_caption_strips_special_and_task_tokens() {
        let decoded = "<s><CAPTION>a cat sitting on a red couch</s>";
        assert_eq!(clean_caption(decoded), "a cat sitting on a red couch");
    }

    #[test]
    fn clean_caption_collapses_whitespace() {
        assert_eq!(
            clean_caption("  two   people\n\twalking  a dog  "),
            "two people walking a dog"
        );
    }

    #[test]
    fn clean_caption_of_only_tokens_is_empty() {
        assert_eq!(clean_caption("<s></s><pad>"), "");
    }

    #[test]
    fn greedy_argmax_picks_the_max_index() {
        assert_eq!(greedy_argmax(&[0.1, 0.9, 0.4, 0.2]), Some(1));
    }

    #[test]
    fn greedy_argmax_breaks_ties_toward_the_lowest_index() {
        assert_eq!(greedy_argmax(&[0.5, 0.5, 0.5]), Some(0));
    }

    #[test]
    fn greedy_argmax_of_empty_is_none() {
        assert_eq!(greedy_argmax(&[]), None);
    }

    #[test]
    fn fill_reports_unsupported_without_writing_a_caption() {
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(4, 4));
        let cfg = VisionConfig::default();
        let mut out = VisionResult::default();
        let error = fill(&image, Path::new("/nonexistent"), &cfg, &mut out)
            .expect_err("v1 captions tier is the unsupported stub");
        assert!(error.to_string().contains("unsupported"));
        assert!(out.caption.is_none());
    }

    /// Generation is gated on a real Florence-2 export being staged under the
    /// model dir — never present in CI, so this skips there (mirrors the other
    /// model-dependent integration tests). Once V6 wires the ONNX pipeline this
    /// asserts a non-empty caption; today it only exercises the skip guard and
    /// the (unsupported) `fill` path without panicking.
    #[test]
    fn generation_when_the_florence_model_is_present() {
        let models_dir = std::env::var_os("INDEX_VISION_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("vision"));
        let present = ["florence2-base-encoder.onnx", "florence2-base-decoder.onnx"]
            .iter()
            .all(|file| models_dir.join(file).is_file());
        if !present {
            eprintln!("skipping caption generation test: Florence-2 model absent");
            return;
        }
        let image = DynamicImage::ImageRgb8(image::RgbImage::new(8, 8));
        let cfg = VisionConfig::default();
        let mut out = VisionResult::default();
        let _ = fill(&image, &models_dir, &cfg, &mut out);
    }
}
