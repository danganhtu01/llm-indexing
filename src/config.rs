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
    let mut dirs: Vec<String> = [
        "$RECYCLE.BIN",
        "System Volume Information",
        ".git",
        "$WinREAgent",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    // Deliberate platform split. `skip_dirs` matches a BARE DIRECTORY BASENAME
    // anywhere in the tree (see `Config::skip_dir`), so the entry "Windows" both
    // under-defended (it never covered `Program Files`, `ProgramData`, or
    // `AppData`) and over-matched (it silently dropped any user folder called
    // "Windows", e.g. `D:\projects\Windows`). On Windows the OS tree is now
    // excluded by the ANCHORED `?:\Windows` entry in `default_skip_paths`, so
    // the bare name is dropped here. On non-Windows `default_skip_paths` is
    // empty by design, so the bare name stays in its original position to keep
    // Linux defaults byte-identical to what they were before.
    #[cfg(not(windows))]
    dirs.push("Windows".to_string());
    dirs.extend(
        [
            "node_modules",
            "index_out",
            ".venv",
            "venv",
            "site-packages",
            "__pycache__",
        ]
        .into_iter()
        .map(str::to_string),
    );
    dirs
}

/// Anchored, full-path directory exclusions — the OS locations that made a
/// whole-drive index walk 800k files of Windows itself. Windows-only: a Linux
/// build gets an EMPTY list, so `skip_paths` is inert there unless an operator
/// sets it in their config YAML.
#[cfg(windows)]
fn default_skip_paths() -> Vec<String> {
    [
        r"?:\Windows",
        r"?:\Program Files",
        r"?:\Program Files (x86)",
        r"?:\ProgramData",
        // Only the AppData segment under a user is excluded. Everything else in
        // a profile — Documents, Desktop, Downloads, ... — is still indexed.
        r"?:\Users\*\AppData",
        r"?:\$WinREAgent",
        r"?:\Recovery",
        r"?:\PerfLogs",
        r"?:\System Volume Information",
        r"?:\$Recycle.Bin",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

/// No default path exclusions off Windows — see the `cfg(windows)` twin.
#[cfg(not(windows))]
fn default_skip_paths() -> Vec<String> {
    Vec::new()
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

/// One segment of a compiled [`skip_paths`](Config::skip_paths) pattern.
#[derive(Debug, Clone)]
enum PathSeg {
    /// The root marker for a pattern written rooted at a separator (`/usr/lib`).
    /// Only ever compared against the same marker, so a RELATIVE path can never
    /// match a rooted pattern — this is what makes the exclusions anchored.
    Root,
    /// Leading `?:` — any drive-letter segment (`C:`, `D:`, ...).
    AnyDrive,
    /// A segment with no `*`; already upper-cased, compared by equality.
    Literal(String),
    /// A segment containing `*`; already upper-cased. `*` matches any run of
    /// characters but never crosses a separator, because matching is per-segment.
    Glob(String),
}

/// Split a path or pattern into upper-cased segments for anchored matching.
///
/// Both separators are accepted so a config file need not be Windows-quoted, and
/// the Windows verbatim prefix (`\\?\`) is stripped: the walker canonicalizes
/// every root, and [`Path::canonicalize`] returns `\\?\C:\...` on Windows, so
/// without this every pattern would miss in production while passing in tests
/// built from literal strings.
fn path_segments(raw: &str) -> Vec<String> {
    let unified = raw.replace('/', "\\");
    let stripped = match unified.strip_prefix(r"\\?\UNC\") {
        // `\\?\UNC\server\share` denotes `\\server\share`.
        Some(rest) => format!(r"\\{rest}"),
        None => unified
            .strip_prefix(r"\\?\")
            .unwrap_or(unified.as_str())
            .to_string(),
    };
    let mut segments = Vec::new();
    if stripped.starts_with('\\') {
        segments.push(String::new());
    }
    segments.extend(
        stripped
            .split('\\')
            .filter(|s| !s.is_empty())
            .map(|s| s.to_uppercase()),
    );
    segments
}

/// Whether `seg` is a drive-letter segment, i.e. what a leading `?:` matches.
fn is_drive_segment(seg: &str) -> bool {
    let bytes = seg.as_bytes();
    bytes.len() == 2 && bytes[0].is_ascii_alphabetic() && bytes[1] == b':'
}

/// Match one already-upper-cased segment against a pattern segment in which `*`
/// matches any run of characters. Iterative backtracking, so it cannot blow the
/// stack on a pattern full of stars.
fn glob_segment(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star, mut retry) = (None, 0usize);
    while t < txt.len() {
        if p < pat.len() && pat[p] == '*' {
            star = Some(p);
            retry = t;
            p += 1;
        } else if p < pat.len() && pat[p] == txt[t] {
            p += 1;
            t += 1;
        } else if let Some(s) = star {
            // Backtrack: let the last `*` swallow one more character.
            p = s + 1;
            retry += 1;
            t = retry;
        } else {
            return false;
        }
    }
    pat[p..].iter().all(|c| *c == '*')
}

/// A `skip_paths` pattern compiled once in [`Config::finalize`]. The walker asks
/// [`Config::skip_path`] about every directory it discovers — millions on a
/// whole-drive run — so nothing here may be recompiled per call.
#[derive(Debug, Clone)]
struct PathPattern(Vec<PathSeg>);

impl PathPattern {
    /// Compile a raw pattern, or `None` when it has no segments at all — which
    /// only an empty config entry produces, since a separators-only entry like
    /// `"/"` still yields the root marker and compiles to a live `[Root]`.
    ///
    /// Dropping it matters because [`matches`](Self::matches) is a whole-path
    /// equality: a zero-segment pattern would match exactly those paths that
    /// also decompose to zero segments, and `all()` over an empty zip is `true`.
    /// The walker only ever passes real directory paths, so this is defence in
    /// depth against a stray empty line in a config file rather than a live
    /// hazard — it is not, despite an earlier comment here, a pattern that
    /// would prefix-match every path and prune the whole tree. Matching has
    /// never been prefix-based.
    fn compile(raw: &str) -> Option<Self> {
        let segments = path_segments(raw);
        if segments.is_empty() {
            return None;
        }
        let compiled = segments
            .into_iter()
            .enumerate()
            .map(|(i, seg)| {
                if seg.is_empty() {
                    PathSeg::Root
                } else if i == 0 && seg == "?:" {
                    PathSeg::AnyDrive
                } else if seg.contains('*') {
                    PathSeg::Glob(seg)
                } else {
                    PathSeg::Literal(seg)
                }
            })
            .collect();
        Some(Self(compiled))
    }

    /// Whether `segments` (an upper-cased path from [`path_segments`]) IS the
    /// directory this pattern names.
    ///
    /// Deliberately an exact match, not a prefix match. The walker prunes on a
    /// hit and never descends, so matching the boundary directory is sufficient
    /// to exclude the whole subtree — and exact matching keeps an explicitly
    /// requested root working. Roots are never tested, so `index C:\Windows\Fonts`
    /// still honours the operator's intent; under prefix semantics its
    /// subdirectories would be pruned out from under it. That case is not
    /// hypothetical: on Windows `tempfile::tempdir()` hands back a path under
    /// `C:\Users\<name>\AppData\Local\Temp`, which prefix semantics would gut.
    fn matches(&self, segments: &[String]) -> bool {
        if self.0.len() != segments.len() {
            return false;
        }
        self.0
            .iter()
            .zip(segments)
            .all(|(pattern, seg)| match pattern {
                PathSeg::Root => seg.is_empty(),
                PathSeg::AnyDrive => is_drive_segment(seg),
                PathSeg::Literal(literal) => literal == seg,
                PathSeg::Glob(glob) => glob_segment(glob, seg),
            })
    }
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
    /// Bare directory BASENAMES pruned anywhere in the tree (`node_modules`).
    #[serde(default = "default_skip_dirs")]
    pub skip_dirs: Vec<String>,
    /// ANCHORED full-path directory patterns, matched case-insensitively against
    /// a directory's whole path. `*` matches within one segment and never
    /// crosses a separator; a leading `?:` matches any drive letter; `/` and `\`
    /// are interchangeable. Unlike [`Config::skip_dirs`] these cannot over-match
    /// a same-named user folder elsewhere in the tree.
    #[serde(default = "default_skip_paths")]
    pub skip_paths: Vec<String>,
    #[serde(default = "default_skip_exts")]
    pub skip_exts: Vec<String>,
    pub follow_symlinks: bool,
    #[serde(default)]
    pub vision: VisionConfig,
    #[serde(skip)]
    skip_dirs_upper: HashSet<String>,
    #[serde(skip)]
    skip_paths_compiled: Vec<PathPattern>,
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
            skip_paths: default_skip_paths(),
            skip_exts: default_skip_exts(),
            follow_symlinks: false,
            vision: VisionConfig::default(),
            skip_dirs_upper: HashSet::new(),
            skip_paths_compiled: Vec::new(),
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
        // Compile the anchored path patterns ONCE — `skip_path` runs per
        // directory over millions of entries. Uncompilable (empty) entries are
        // dropped rather than kept as match-everything patterns.
        self.skip_paths_compiled = self
            .skip_paths
            .iter()
            .filter_map(|raw| PathPattern::compile(raw))
            .collect();
        self.skip_exts_lower = self.skip_exts.iter().map(|s| s.to_lowercase()).collect();
    }

    pub fn skip_dir(&self, name: &str) -> bool {
        self.skip_dirs_upper.contains(&name.to_uppercase())
    }

    /// Whether `path` is an excluded OS location per [`Config::skip_paths`].
    /// Matching is case-insensitive and ANCHORED: a pattern must match the path
    /// in full, so `?:\Windows` prunes `C:\Windows` while leaving a user's
    /// `D:\projects\Windows` alone.
    ///
    /// Returns `false` immediately when no patterns are configured, which is the
    /// default on every non-Windows build.
    pub fn skip_path(&self, path: &Path) -> bool {
        if self.skip_paths_compiled.is_empty() {
            return false;
        }
        let segments = path_segments(&path.to_string_lossy());
        self.skip_paths_compiled
            .iter()
            .any(|pattern| pattern.matches(&segments))
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
    use std::path::Path;

    /// `Config::default()` does not compile the skip patterns — only
    /// `finalize()` (and therefore `Config::load`) does.
    fn finalized() -> Config {
        let mut config = Config::default();
        config.finalize();
        config
    }

    /// A finalized config whose only path exclusions are `patterns`, so a test
    /// can assert matcher semantics without depending on the platform defaults.
    fn with_skip_paths(patterns: &[&str]) -> Config {
        let mut config = Config {
            skip_paths: patterns.iter().map(|p| p.to_string()).collect(),
            ..Config::default()
        };
        config.finalize();
        config
    }

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

    #[test]
    fn skip_path_is_inert_without_patterns() {
        // The non-Windows default, and the fast path the walker relies on.
        let config = with_skip_paths(&[]);
        assert!(!config.skip_path(Path::new(r"C:\Windows")));
        assert!(!config.skip_path(Path::new("/usr/lib")));
    }

    #[test]
    fn an_empty_skip_path_pattern_is_dropped_rather_than_compiled() {
        // Pins the `segments.is_empty()` guard in `PathPattern::compile`.
        //
        // The obvious assertions here do NOT pin it: `matches` compares whole
        // lengths first, so a zero-segment pattern can never reach a path with
        // any segments, and an unguarded build passes them just as happily.
        // The guard is only observable against a path that ALSO decomposes to
        // zero segments, because `all()` over an empty zip is `true` — so that
        // is what this asserts.
        let config = with_skip_paths(&[""]);
        assert!(
            !config.skip_path(Path::new("")),
            "an empty pattern must be dropped, not compiled into one that \
             matches the empty path"
        );
        // And it does not somehow catch real paths either.
        assert!(!config.skip_path(Path::new(r"C:\Users\danga\Documents")));
    }

    #[test]
    fn a_separators_only_pattern_is_the_root_not_an_empty_pattern() {
        // `path_segments` pushes a root marker before filtering empties, so "/"
        // and "\" are NOT zero-segment: they compile to a live `[Root]` that
        // matches the filesystem root and nothing else. Worth pinning because
        // it is the natural way to misread the guard above.
        let config = with_skip_paths(&["/"]);
        assert!(!config.skip_path(Path::new(r"C:\Users\danga\Documents")));
        assert!(!config.skip_path(Path::new("/home/danga/documents")));
        assert!(
            config.skip_path(Path::new("/")),
            "a separators-only pattern names the root itself"
        );
    }

    #[test]
    fn skip_path_patterns_are_anchored_not_bare_names() {
        // The core fix, asserted platform-independently by setting the pattern
        // explicitly: an anchored pattern matches the OS location and NOTHING
        // that merely shares its basename.
        let config = with_skip_paths(&[r"?:\Windows"]);
        assert!(config.skip_path(Path::new(r"C:\Windows")));
        assert!(!config.skip_path(Path::new(r"D:\projects\Windows")));
        assert!(!config.skip_path(Path::new(r"C:\Users\danga\src\Windows")));
    }

    #[test]
    fn skip_path_wildcard_does_not_cross_separators() {
        let config = with_skip_paths(&[r"?:\Users\*\AppData"]);
        assert!(config.skip_path(Path::new(r"C:\Users\danga\AppData")));
        // `*` must not swallow `danga\Local` and match a deeper AppData.
        assert!(!config.skip_path(Path::new(r"C:\Users\danga\Local\AppData")));
        // ...nor collapse to zero segments and match a top-level AppData.
        assert!(!config.skip_path(Path::new(r"C:\Users\AppData")));
    }

    #[test]
    fn skip_path_accepts_both_separators_in_patterns() {
        // A config file must not have to be Windows-quoted.
        let config = with_skip_paths(&["?:/Program Files"]);
        assert!(config.skip_path(Path::new(r"C:\Program Files")));
    }

    #[test]
    fn skip_path_rejects_relative_paths_against_rooted_patterns() {
        let config = with_skip_paths(&["/usr/lib"]);
        assert!(config.skip_path(Path::new("/usr/lib")));
        // Anchoring: a relative path that merely ends the same way is not a hit.
        assert!(!config.skip_path(Path::new("usr/lib")));
        assert!(!config.skip_path(Path::new("/opt/usr/lib")));
    }

    #[cfg(windows)]
    #[test]
    fn windows_defaults_prune_os_locations() {
        let config = finalized();
        for excluded in [
            r"C:\Windows",
            r"C:\Program Files",
            r"C:\Program Files (x86)",
            r"C:\ProgramData",
            r"C:\$WinREAgent",
            r"C:\Recovery",
            r"C:\PerfLogs",
            r"C:\System Volume Information",
            r"C:\$Recycle.Bin",
            // `?:` is any drive, not just C:.
            r"D:\Windows",
        ] {
            assert!(config.skip_path(Path::new(excluded)), "{excluded}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_defaults_prune_appdata_but_keep_user_documents() {
        // The correctness bar: excluding user data is worse than indexing junk.
        let config = finalized();
        assert!(config.skip_path(Path::new(r"C:\Users\danga\AppData")));
        for kept in [
            r"C:\Users\danga\Documents",
            r"C:\Users\danga\Desktop",
            r"C:\Users\danga\Downloads",
            r"C:\Users\danga",
            r"C:\Users",
            r"D:\projects",
        ] {
            assert!(!config.skip_path(Path::new(kept)), "{kept}");
        }
    }

    #[cfg(windows)]
    #[test]
    fn windows_defaults_are_case_insensitive() {
        let config = finalized();
        assert!(config.skip_path(Path::new(r"c:\program files")));
        assert!(config.skip_path(Path::new(r"C:\PROGRAM FILES (X86)")));
        assert!(config.skip_path(Path::new(r"c:\users\DANGA\appdata")));
    }

    #[cfg(windows)]
    #[test]
    fn windows_defaults_match_canonicalized_verbatim_paths() {
        // `walk` canonicalizes every root and `Path::canonicalize` returns
        // `\\?\C:\...` on Windows, so the verbatim prefix must be stripped or
        // every default would miss in production while passing on literals.
        let config = finalized();
        assert!(config.skip_path(Path::new(r"\\?\C:\Windows")));
        assert!(config.skip_path(Path::new(r"\\?\C:\Users\danga\AppData")));
        assert!(!config.skip_path(Path::new(r"\\?\D:\projects\Windows")));
    }

    #[cfg(windows)]
    #[test]
    fn windows_drops_bare_windows_skip_dir_in_favour_of_anchored_path() {
        // The anchoring fix, end to end. Under the OLD behaviour "Windows" was a
        // bare basename in skip_dirs, so skip_dir("Windows") was true and the
        // walker pruned D:\projects\Windows. Both assertions below fail against
        // that behaviour; together they pin the whole change.
        let config = finalized();
        assert!(!config.skip_dirs.iter().any(|d| d == "Windows"));
        assert!(!config.skip_dir("Windows"));
        assert!(config.skip_path(Path::new(r"C:\Windows")));
        assert!(!config.skip_path(Path::new(r"D:\projects\Windows")));
        // The genuinely name-based entries are untouched.
        assert!(config.skip_dir("node_modules"));
        assert!(config.skip_dir(".git"));
        assert!(config.skip_dir("__pycache__"));
    }

    #[cfg(not(windows))]
    #[test]
    fn non_windows_defaults_are_unchanged() {
        // Linux Docker deployments must see byte-identical defaults: no path
        // exclusions at all, and "Windows" still a bare skip_dirs entry in its
        // original position.
        let config = finalized();
        assert!(super::default_skip_paths().is_empty());
        assert!(config.skip_paths.is_empty());
        assert_eq!(
            config.skip_dirs,
            vec![
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
        );
        assert!(config.skip_dir("Windows"));
        assert!(!config.skip_path(Path::new("/anything")));
    }

    #[test]
    fn yaml_can_override_skip_paths() {
        // Operator-tunable through --config with no further plumbing.
        let mut config: Config =
            serde_yaml::from_str("skip_paths:\n  - \"?:/Steam/steamapps\"\n").unwrap();
        config.finalize();
        assert!(config.skip_path(Path::new(r"E:\Steam\steamapps")));
        // An explicit list replaces the defaults rather than extending them.
        assert!(!config.skip_path(Path::new(r"C:\Windows")));
    }
}
