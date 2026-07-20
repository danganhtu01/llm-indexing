use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::vision::VisionMode;

/// Upper bound on indexing worker threads. The single definition of the ceiling
/// used by [`Config::finalize`], the service per-job clamp, and the `workers.max`
/// value `GET /settings` advertises.
pub const MAX_WORKERS: usize = 64;

/// Clamp a requested worker count into `1..=MAX_WORKERS` — the SINGLE clamp used
/// by [`Config::finalize`], the service per-job worker resolution (`run_job`), and
/// the `workers.default` `GET /settings` advertises, so the advertised default,
/// the config value, and the count a job actually runs with can never disagree.
pub fn clamp_workers(workers: usize) -> usize {
    workers.clamp(1, MAX_WORKERS)
}

fn default_languages() -> Vec<String> {
    vec!["vi".into(), "en".into(), "ru".into(), "de".into()]
}
fn default_ocr() -> String {
    "auto".into()
}
fn default_sidecar() -> String {
    "mirror".into()
}
fn default_workers() -> usize {
    8
}
fn default_max_bytes() -> u64 {
    100 * 1024 * 1024
}
fn default_max_chars() -> usize {
    1_000_000
}
fn default_ocr_pages() -> usize {
    20
}
fn default_tesseract() -> String {
    "tesseract".into()
}
fn default_ocr_langs() -> String {
    // Default stays "vie+eng" — the common corpus and what jobs/portal pass
    // explicitly today. Russian/German OCR (tesseract rus/deu, bundled as
    // tessdata_best) is opt-in via "vie+eng+rus+deu" to avoid slowing every
    // scan with unused language models.
    "vie+eng".into()
}
fn default_ocr_dpi() -> u32 {
    300
}
fn default_ocr_psm() -> String {
    "3".into()
}
fn default_true() -> bool {
    true
}
fn default_data_dir() -> PathBuf {
    PathBuf::from("data")
}
fn default_whisper_model() -> PathBuf {
    std::env::var_os("WHISPER_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/ggml-small.bin"))
}
fn default_embedding_cache() -> PathBuf {
    std::env::var_os("FASTEMBED_CACHE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/fastembed"))
}
fn default_embedding_model() -> String {
    "intfloat/multilingual-e5-small".into()
}
fn default_skip_dirs() -> Vec<String> {
    [
        "$RECYCLE.BIN",
        "System Volume Information",
        ".git",
        "$WinREAgent",
        "Windows",
        "node_modules",
        "index_out",
        ".venv",
        "venv",
        "site-packages",
        "__pycache__",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
fn default_skip_exts() -> Vec<String> {
    [".sys", ".dll", ".exe", ".iso", ".vmdk", ".lock"]
        .into_iter()
        .map(str::to_string)
        .collect()
}
fn default_vision_models_dir() -> PathBuf {
    PathBuf::from("vision")
}
fn default_detector() -> String {
    crate::settings::DETECTOR_DEFAULT.into()
}
fn default_tagger() -> String {
    crate::settings::TAGGER_DEFAULT.into()
}
fn default_captioner() -> String {
    crate::settings::CAPTIONER_DEFAULT.into()
}
fn default_tag_score() -> f32 {
    0.22
}
fn default_tag_top_k() -> usize {
    8
}
fn default_detector_conf() -> f32 {
    // VISION-SPEC §4: RF-DETR-Nano confidence floor. DETR-style postprocessing
    // (no NMS), so 0.5 both matches the spec and drops the low-confidence
    // false positives a looser threshold lets through.
    0.5
}
fn default_max_frames() -> usize {
    12
}
fn default_vision_timeout() -> u64 {
    60
}
fn default_caption_timeout() -> u64 {
    300
}
fn default_max_pixels() -> u64 {
    250_000_000
}
fn default_max_alloc_bytes() -> u64 {
    1 << 30
}

/// Vision-mode knobs (VISION-SPEC section 4). All default to values that keep
/// the feature off and inert; `max` is the effective tier ceiling — the native
/// `index --vision` flag sets it, and the service sets it per job after
/// validating the request against `serve --vision-max`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// Effective vision tier for a run. Default `off`.
    #[serde(default)]
    pub max: VisionMode,
    /// Model directory; resolved under `data_dir` when relative (see
    /// [`Config::vision_models_dir`]).
    #[serde(default = "default_vision_models_dir")]
    pub models_dir: PathBuf,
    /// Object-detector selection: the v1 model id (`nano`) or `off` to skip
    /// detection while still running the rest of the requested tier. A per-job
    /// `vision_opts.detector` overrides it through the settings merge.
    #[serde(default = "default_detector")]
    pub detector: String,
    /// Zero-shot tagger selection: `clip`, or `off` to skip tagging/embedding.
    #[serde(default = "default_tagger")]
    pub tagger: String,
    /// Captioner selection: `florence2`, or `off` to skip captioning.
    #[serde(default = "default_captioner")]
    pub captioner: String,
    /// Minimum CLIP zero-shot tag score to keep.
    #[serde(default = "default_tag_score")]
    pub tag_score: f32,
    /// Maximum number of tags kept per file.
    #[serde(default = "default_tag_top_k")]
    pub tag_top_k: usize,
    /// Minimum object-detector confidence to keep a detection (RF-DETR-Nano,
    /// DETR-style postprocessing, no NMS).
    #[serde(default = "default_detector_conf")]
    pub detector_conf: f32,
    /// Maximum keyframes analysed per video.
    #[serde(default = "default_max_frames")]
    pub max_frames: usize,
    /// Per-file vision timeout (seconds) for non-caption tiers.
    #[serde(default = "default_vision_timeout")]
    pub timeout_secs: u64,
    /// Per-file vision timeout (seconds) for the captions tier.
    #[serde(default = "default_caption_timeout")]
    pub caption_timeout_secs: u64,
    /// Reject images above this pixel count (`decode-limit`).
    #[serde(default = "default_max_pixels")]
    pub max_pixels: u64,
    /// Cap a single decode allocation (bytes) — image-crate `Limits`.
    #[serde(default = "default_max_alloc_bytes")]
    pub max_alloc_bytes: u64,
}

impl Default for VisionConfig {
    fn default() -> Self {
        Self {
            max: VisionMode::Off,
            models_dir: default_vision_models_dir(),
            detector: default_detector(),
            tagger: default_tagger(),
            captioner: default_captioner(),
            tag_score: default_tag_score(),
            tag_top_k: default_tag_top_k(),
            detector_conf: default_detector_conf(),
            max_frames: default_max_frames(),
            timeout_secs: default_vision_timeout(),
            caption_timeout_secs: default_caption_timeout(),
            max_pixels: default_max_pixels(),
            max_alloc_bytes: default_max_alloc_bytes(),
        }
    }
}

impl VisionConfig {
    /// Whether the object-detector sub-tier runs (`detector != "off"`). Set from
    /// `vision_opts.detector`/config and consulted by
    /// [`VisionAnalyzer::analyze_image`](crate::vision::VisionAnalyzer::analyze_image)
    /// so an `off` toggle genuinely skips detection.
    pub fn detector_enabled(&self) -> bool {
        !self.detector.eq_ignore_ascii_case("off")
    }
    /// Whether the zero-shot tagger sub-tier runs (`tagger != "off"`).
    pub fn tagger_enabled(&self) -> bool {
        !self.tagger.eq_ignore_ascii_case("off")
    }
    /// Whether the captioner sub-tier runs (`captioner != "off"`).
    pub fn captioner_enabled(&self) -> bool {
        !self.captioner.eq_ignore_ascii_case("off")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    #[serde(default = "default_languages")]
    pub languages: Vec<String>,
    #[serde(default = "default_ocr")]
    pub ocr: String,
    #[serde(default = "default_sidecar")]
    pub sidecar: String,
    #[serde(default = "default_workers")]
    pub workers: usize,
    #[serde(default = "default_max_bytes")]
    pub max_bytes: u64,
    #[serde(default = "default_max_chars")]
    pub max_chars: usize,
    pub hash: bool,
    #[serde(default = "default_ocr_pages")]
    pub ocr_max_pages: usize,
    #[serde(default = "default_tesseract")]
    pub tesseract_cmd: String,
    #[serde(default = "default_ocr_langs")]
    pub ocr_langs: String,
    /// Rasterization DPI for PDF page OCR — 300 is Tesseract's sweet spot.
    #[serde(default = "default_ocr_dpi")]
    pub ocr_dpi: u32,
    /// Tesseract page-segmentation mode; a near-empty result retries with 6.
    #[serde(default = "default_ocr_psm")]
    pub ocr_psm: String,
    /// Grayscale + deskew + contrast-stretch inputs before OCR (ImageMagick).
    #[serde(default = "default_true")]
    pub ocr_preprocess: bool,
    #[serde(default = "default_data_dir")]
    pub data_dir: PathBuf,
    #[serde(default = "default_whisper_model")]
    pub whisper_model: PathBuf,
    #[serde(default = "default_embedding_cache")]
    pub embedding_cache: PathBuf,
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,
    #[serde(default = "default_skip_dirs")]
    pub skip_dirs: Vec<String>,
    #[serde(default = "default_skip_exts")]
    pub skip_exts: Vec<String>,
    pub follow_symlinks: bool,
    #[serde(default)]
    pub vision: VisionConfig,
    #[serde(skip)]
    skip_dirs_upper: HashSet<String>,
    #[serde(skip)]
    skip_exts_lower: HashSet<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            languages: default_languages(),
            ocr: default_ocr(),
            sidecar: default_sidecar(),
            workers: default_workers(),
            max_bytes: default_max_bytes(),
            max_chars: default_max_chars(),
            hash: false,
            ocr_max_pages: default_ocr_pages(),
            tesseract_cmd: default_tesseract(),
            ocr_langs: default_ocr_langs(),
            ocr_dpi: default_ocr_dpi(),
            ocr_psm: default_ocr_psm(),
            ocr_preprocess: true,
            data_dir: default_data_dir(),
            whisper_model: default_whisper_model(),
            embedding_cache: default_embedding_cache(),
            embedding_model: default_embedding_model(),
            skip_dirs: default_skip_dirs(),
            skip_exts: default_skip_exts(),
            follow_symlinks: false,
            vision: VisionConfig::default(),
            skip_dirs_upper: HashSet::new(),
            skip_exts_lower: HashSet::new(),
        }
    }
}

impl Config {
    pub fn load(path: Option<&Path>) -> Result<Self> {
        let mut config = if let Some(path) = path {
            let raw = fs::read_to_string(path)
                .with_context(|| format!("reading config {}", path.display()))?;
            let mut parsed: Self = serde_yaml::from_str(&raw).context("parsing YAML config")?;
            if parsed.data_dir.is_relative() {
                parsed.data_dir = path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(&parsed.data_dir);
            }
            parsed
        } else {
            Self::default()
        };
        config.finalize();
        Ok(config)
    }

    pub fn finalize(&mut self) {
        self.workers = clamp_workers(self.workers);
        // Clamp the OCR/vision knobs to the settings-surface bounds (their single
        // definition in `settings.rs`). Only per-job overrides are validated at
        // submit; without this a mis-set config base would flow unvalidated into
        // every job (e.g. `pdftoppm -r 9999`) and make GET /settings advertise a
        // default outside its own min/max. Idempotent (finalize may run twice).
        use crate::settings::{
            OCR_DPI_RANGE, OCR_MAX_PAGES_RANGE, OCR_PSM_RANGE, VISION_DETECTOR_CONF_RANGE,
            VISION_MAX_FRAMES_RANGE, VISION_TAG_THRESHOLD_RANGE, VISION_TAG_TOP_K_RANGE,
            VISION_TIMEOUT_RANGE,
        };
        self.ocr_dpi = self.ocr_dpi.clamp(OCR_DPI_RANGE.0, OCR_DPI_RANGE.1);
        self.ocr_max_pages = self
            .ocr_max_pages
            .clamp(OCR_MAX_PAGES_RANGE.0, OCR_MAX_PAGES_RANGE.1);
        self.ocr_psm = match self.ocr_psm.trim().parse::<u8>() {
            Ok(value) => value.clamp(OCR_PSM_RANGE.0, OCR_PSM_RANGE.1).to_string(),
            Err(_) => default_ocr_psm(),
        };
        self.vision.detector_conf = self
            .vision
            .detector_conf
            .clamp(VISION_DETECTOR_CONF_RANGE.0, VISION_DETECTOR_CONF_RANGE.1);
        self.vision.tag_score = self
            .vision
            .tag_score
            .clamp(VISION_TAG_THRESHOLD_RANGE.0, VISION_TAG_THRESHOLD_RANGE.1);
        self.vision.tag_top_k = self
            .vision
            .tag_top_k
            .clamp(VISION_TAG_TOP_K_RANGE.0, VISION_TAG_TOP_K_RANGE.1);
        self.vision.max_frames = self
            .vision
            .max_frames
            .clamp(VISION_MAX_FRAMES_RANGE.0, VISION_MAX_FRAMES_RANGE.1);
        self.vision.timeout_secs = self
            .vision
            .timeout_secs
            .clamp(VISION_TIMEOUT_RANGE.0, VISION_TIMEOUT_RANGE.1);
        self.skip_dirs_upper = self.skip_dirs.iter().map(|s| s.to_uppercase()).collect();
        self.skip_exts_lower = self.skip_exts.iter().map(|s| s.to_lowercase()).collect();
    }

    pub fn skip_dir(&self, name: &str) -> bool {
        self.skip_dirs_upper.contains(&name.to_uppercase())
    }

    pub fn skip_ext(&self, ext: &str) -> bool {
        self.skip_exts_lower.contains(&ext.to_lowercase())
    }

    /// Absolute-ish directory the vision models live under: `vision.models_dir`
    /// verbatim when absolute, else resolved against `data_dir`. Pure (unlike
    /// [`Config::finalize`], which runs more than once) so repeated calls never
    /// double-join.
    pub fn vision_models_dir(&self) -> PathBuf {
        if self.vision.models_dir.is_absolute() {
            self.vision.models_dir.clone()
        } else {
            self.data_dir.join(&self.vision.models_dir)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Config;

    #[test]
    fn ocr_defaults() {
        let config = Config::default();
        assert_eq!(config.ocr_dpi, 300);
        assert_eq!(config.ocr_psm, "3");
        assert!(config.ocr_preprocess);
    }

    #[test]
    fn yaml_overrides_ocr_dpi() {
        let config: Config = serde_yaml::from_str("ocr_dpi: 200").unwrap();
        assert_eq!(config.ocr_dpi, 200);
        // Unspecified OCR knobs fall back to their defaults.
        assert_eq!(config.ocr_psm, "3");
        assert!(config.ocr_preprocess);
    }

    #[test]
    fn yaml_overrides_ocr_langs_multilingual() {
        let config: Config = serde_yaml::from_str("ocr_langs: vie+eng+rus+deu").unwrap();
        assert_eq!(config.ocr_langs, "vie+eng+rus+deu");
    }

    #[test]
    fn finalize_clamps_out_of_range_config_knobs() {
        // A mis-set config base is corrected to the settings-surface bounds so it
        // cannot flow unvalidated into jobs or make GET /settings self-contradict.
        let mut config: Config = serde_yaml::from_str(
            "ocr_dpi: 9999\nocr_max_pages: 100000\nocr_psm: '99'\n\
             vision:\n  tag_top_k: 999\n  timeout_secs: 100000\n  detector_conf: 5.0\n",
        )
        .unwrap();
        config.finalize();
        assert_eq!(config.ocr_dpi, 1200);
        assert_eq!(config.ocr_max_pages, 500);
        assert_eq!(config.ocr_psm, "13");
        assert_eq!(config.vision.tag_top_k, 32);
        assert_eq!(config.vision.timeout_secs, 1800);
        assert!((config.vision.detector_conf - 0.95).abs() < 1e-6);
    }

    #[test]
    fn finalize_resets_unparsable_psm_to_default() {
        let mut config: Config = serde_yaml::from_str("ocr_psm: auto").unwrap();
        config.finalize();
        assert_eq!(config.ocr_psm, "3");
    }
}
