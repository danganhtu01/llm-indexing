//! Per-job OCR / vision settings surface (SETTINGS-SPEC sections 1 & 3).
//!
//! One struct pair — [`OcrSettings`] and [`VisionSettings`] — serves BOTH the
//! service-wide config defaults AND the per-job overrides. Every field is
//! `Option`, so the same type expresses a fully-populated base (built from
//! [`Config`]) and a partial per-job override (from `JobRequest.ocr_opts` /
//! `vision_opts` or the native `index --ocr-*` / `--vision-*` flags). The two
//! are combined by [`OcrSettings::merge`] / [`VisionSettings::merge`] — THE
//! single merge path — with precedence `per-job override > service config >
//! built-in default`. The built-in defaults live once in `config.rs`; these
//! structs never redefine a knob, they only carry and merge it.
//!
//! Validation ([`OcrSettings::validate`] / [`VisionSettings::validate`]) reports
//! a field-specific message so submit can answer `400` naming the offending
//! knob; OCR languages are checked against the INSTALLED tessdata the same way
//! `ocr::TesseractOcr` resolves them — tesseract reads ONE source, so a job's
//! languages must all live together in the bundled `<data_dir>/tessdata` set or
//! all in the system packs it reports — never a hardcoded list.

use std::collections::BTreeSet;
use std::path::Path;
use std::process::Command;

use serde::{Deserialize, Serialize};

use crate::config::Config;

// ── Bounds (SETTINGS-SPEC section 1). Single definition of each knob's range. ──
const DPI_MIN: u32 = 150;
const DPI_MAX: u32 = 1200;
const PSM_MIN: u8 = 0;
const PSM_MAX: u8 = 13;
const MAX_PAGES_MIN: usize = 1;
const MAX_PAGES_MAX: usize = 500;
const DETECTOR_CONF_MIN: f32 = 0.05;
const DETECTOR_CONF_MAX: f32 = 0.95;
const TAG_THRESHOLD_MIN: f32 = 0.0;
const TAG_THRESHOLD_MAX: f32 = 1.0;
const TAG_TOP_K_MIN: usize = 1;
const TAG_TOP_K_MAX: usize = 32;
const MAX_FRAMES_MIN: usize = 1;
const MAX_FRAMES_MAX: usize = 64;
const TIMEOUT_MIN: u64 = 5;
const TIMEOUT_MAX: u64 = 1800;

/// The v1 model selected by each vision sub-model toggle when it is not `off` —
/// the SINGLE definition of these ids, referenced by both the accepted-value
/// lists below and the `VisionConfig` sub-model defaults (`config.rs`), so the id
/// is never spelled twice.
pub const DETECTOR_DEFAULT: &str = "nano";
pub const TAGGER_DEFAULT: &str = "clip";
pub const CAPTIONER_DEFAULT: &str = "florence2";

/// Accepted values for the vision sub-model toggles. `off` disables the
/// sub-tier; the named value selects the v1 model for it. Public so the
/// `GET /settings` capability surface enumerates the accepted ids from this one
/// definition instead of re-listing them.
pub const DETECTORS: &[&str] = &["off", DETECTOR_DEFAULT];
pub const TAGGERS: &[&str] = &["off", TAGGER_DEFAULT];
pub const CAPTIONERS: &[&str] = &["off", CAPTIONER_DEFAULT];

/// Numeric bounds for the per-job OCR knobs, as `(min, max)` — exposed so
/// `GET /settings` renders each range from THIS single definition (SETTINGS-SPEC
/// §2: ranges exist in one place) rather than re-stating the limits.
pub const OCR_DPI_RANGE: (u32, u32) = (DPI_MIN, DPI_MAX);
pub const OCR_PSM_RANGE: (u8, u8) = (PSM_MIN, PSM_MAX);
pub const OCR_MAX_PAGES_RANGE: (usize, usize) = (MAX_PAGES_MIN, MAX_PAGES_MAX);

/// Numeric bounds for the per-job vision knobs, as `(min, max)` — exposed (like
/// the OCR ranges) so [`Config::finalize`](crate::config::Config::finalize) can
/// clamp a mis-set config base to the SAME limits submit validates a per-job
/// override against, from this one definition, rather than letting an
/// out-of-range base flow into jobs.
pub const VISION_DETECTOR_CONF_RANGE: (f32, f32) = (DETECTOR_CONF_MIN, DETECTOR_CONF_MAX);
pub const VISION_TAG_THRESHOLD_RANGE: (f32, f32) = (TAG_THRESHOLD_MIN, TAG_THRESHOLD_MAX);
pub const VISION_TAG_TOP_K_RANGE: (usize, usize) = (TAG_TOP_K_MIN, TAG_TOP_K_MAX);
pub const VISION_MAX_FRAMES_RANGE: (usize, usize) = (MAX_FRAMES_MIN, MAX_FRAMES_MAX);
pub const VISION_TIMEOUT_RANGE: (u64, u64) = (TIMEOUT_MIN, TIMEOUT_MAX);

/// Per-job OCR overrides. Absent (`None`) fields fall through to the service
/// config, then the built-in default, via [`OcrSettings::merge`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct OcrSettings {
    /// Rasterization DPI for PDF page OCR (`150..=1200`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
    /// Tesseract page-segmentation mode as an engine-style string (`"0".."13"`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub psm: Option<String>,
    /// ImageMagick pre-pass (grayscale/deskew/contrast) on or off.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub preprocess: Option<bool>,
    /// Maximum PDF pages OCR'd per file (`1..=500`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_pages: Option<usize>,
    /// Tesseract language string (`vie+eng+…`); validated against the installed
    /// tessdata. Wins over the legacy top-level `ocr_langs`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub langs: Option<String>,
}

impl OcrSettings {
    /// The base settings a job starts from: every OCR knob read out of `config`
    /// (built-in default overlaid by the YAML config). All fields are `Some`, so
    /// merging a partial override onto it always yields a complete result.
    pub fn from_config(config: &Config) -> Self {
        Self {
            dpi: Some(config.ocr_dpi),
            psm: Some(config.ocr_psm.clone()),
            preprocess: Some(config.ocr_preprocess),
            max_pages: Some(config.ocr_max_pages),
            langs: Some(config.ocr_langs.clone()),
        }
    }

    /// THE merge path: each field of `over` wins when set, else `base` is kept.
    pub fn merge(base: &Self, over: &Self) -> Self {
        Self {
            dpi: over.dpi.or(base.dpi),
            psm: over.psm.clone().or_else(|| base.psm.clone()),
            preprocess: over.preprocess.or(base.preprocess),
            max_pages: over.max_pages.or(base.max_pages),
            langs: over.langs.clone().or_else(|| base.langs.clone()),
        }
    }

    /// Resolve the effective settings for a job: the config-derived base overlaid
    /// by the per-job override (if any), through the single [`merge`](Self::merge)
    /// path. `None` ⇒ exactly the config defaults.
    pub fn resolve(config: &Config, over: Option<&Self>) -> Self {
        let base = Self::from_config(config);
        match over {
            Some(over) => Self::merge(&base, over),
            None => base,
        }
    }

    /// Write the resolved knobs back into `config` so the existing pipeline
    /// plumbing (TesseractOcr handle + PDF rasterizer) carries the per-job
    /// values. Unset fields leave `config` untouched, keeping the off-path
    /// byte-identical.
    pub fn apply_to(&self, config: &mut Config) {
        config.ocr_dpi = self.dpi.unwrap_or(config.ocr_dpi);
        if let Some(psm) = &self.psm {
            config.ocr_psm = psm.clone();
        }
        config.ocr_preprocess = self.preprocess.unwrap_or(config.ocr_preprocess);
        config.ocr_max_pages = self.max_pages.unwrap_or(config.ocr_max_pages);
        if let Some(langs) = &self.langs {
            // Write the CANONICAL langs (same normalization validation used), not
            // the raw string: "vie + eng" / "vie++eng" validate (per-component
            // trim + drop-empty) but must not reach tesseract `-l` verbatim, or it
            // fails loading lang "vie " on every page and the scan indexes empty.
            let requested = normalized_langs(langs);
            if !requested.is_empty() {
                config.ocr_langs = requested.join("+");
            }
        }
    }

    /// Validate every set field's range/format (all knobs EXCEPT `langs`, which
    /// needs the installed-tessdata set — see [`validate_langs`](Self::validate_langs)).
    /// Returns a field-specific message on the first violation.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(dpi) = self.dpi {
            if !(DPI_MIN..=DPI_MAX).contains(&dpi) {
                return Err(format!(
                    "ocr.dpi must be between {DPI_MIN} and {DPI_MAX} (got {dpi})"
                ));
            }
        }
        if let Some(psm) = &self.psm {
            match psm.trim().parse::<u8>() {
                Ok(value) if (PSM_MIN..=PSM_MAX).contains(&value) => {}
                Ok(value) => {
                    return Err(format!(
                        "ocr.psm must be between {PSM_MIN} and {PSM_MAX} (got {value})"
                    ))
                }
                Err(_) => {
                    return Err(format!(
                        "ocr.psm must be an integer {PSM_MIN}-{PSM_MAX} (got '{psm}')"
                    ))
                }
            }
        }
        if let Some(max_pages) = self.max_pages {
            if !(MAX_PAGES_MIN..=MAX_PAGES_MAX).contains(&max_pages) {
                return Err(format!(
                    "ocr.max_pages must be between {MAX_PAGES_MIN} and {MAX_PAGES_MAX} (got {max_pages})"
                ));
            }
        }
        Ok(())
    }

    /// Reject a `langs` value that cannot load under the same all-or-nothing-per-
    /// source resolution `ocr::TesseractOcr` uses. Tesseract reads ONE tessdata
    /// source, so every requested language must live together in the bundled
    /// `<data_dir>/tessdata` set OR together in the system-pack set. A language
    /// absent from both is "uninstalled"; a combo split across the two sources
    /// (each present, but in different sources) cannot load together and is
    /// rejected too — otherwise it passes submit but tesseract fails per page at
    /// run time and the scan indexes empty. No `langs` ⇒ `Ok`.
    pub fn validate_langs(
        &self,
        bundled: &BTreeSet<String>,
        system: &BTreeSet<String>,
    ) -> Result<(), String> {
        let Some(langs) = &self.langs else {
            return Ok(());
        };
        let requested = normalized_langs(langs);
        if requested.is_empty() {
            return Err("ocr.langs must name at least one tesseract language".to_string());
        }
        let missing: Vec<&str> = requested
            .iter()
            .map(String::as_str)
            .filter(|lang| !bundled.contains(*lang) && !system.contains(*lang))
            .collect();
        if !missing.is_empty() {
            let mut union: BTreeSet<&str> = BTreeSet::new();
            union.extend(bundled.iter().map(String::as_str));
            union.extend(system.iter().map(String::as_str));
            let available = if union.is_empty() {
                "none".to_string()
            } else {
                union.into_iter().collect::<Vec<_>>().join(", ")
            };
            return Err(format!(
                "ocr.langs contains uninstalled tesseract language(s): {} (installed: {available})",
                missing.join(", ")
            ));
        }
        // Every requested language is installed somewhere; now require they share
        // ONE source, mirroring TesseractOcr (bundled only when ALL requested are
        // bundled, else system-only).
        let all_bundled = requested.iter().all(|lang| bundled.contains(lang.as_str()));
        let all_system = requested.iter().all(|lang| system.contains(lang.as_str()));
        if !all_bundled && !all_system {
            let bundled_only: Vec<&str> = requested
                .iter()
                .map(String::as_str)
                .filter(|lang| bundled.contains(*lang) && !system.contains(*lang))
                .collect();
            let system_only: Vec<&str> = requested
                .iter()
                .map(String::as_str)
                .filter(|lang| system.contains(*lang) && !bundled.contains(*lang))
                .collect();
            return Err(format!(
                "ocr.langs mixes tesseract sources that cannot load together \
                 (bundled-only: {}; system-only: {}); use languages from a single source",
                bundled_only.join(", "),
                system_only.join(", "),
            ));
        }
        Ok(())
    }
}

/// Canonicalize a tesseract langs string: split on `+`, trim each component, drop
/// empties. The ONE normalization shared by [`OcrSettings::validate_langs`] and
/// [`OcrSettings::apply_to`], so a value that validates is byte-for-byte the value
/// that reaches the tesseract `-l` invocation — no whitespace/empty-component
/// drift between the gate and the run.
fn normalized_langs(langs: &str) -> Vec<String> {
    langs
        .split('+')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(str::to_string)
        .collect()
}

/// Per-job vision overrides. Active only when the requested tier is not `off`
/// (the tier is carried by `JobRequest.vision` and capped by `--vision-max`);
/// absent fields fall through to the service config, then the built-in default.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct VisionSettings {
    /// Object detector: `off` | `nano` (RF-DETR-Nano).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector: Option<String>,
    /// Minimum detector confidence to keep a detection (`0.05..=0.95`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detector_conf: Option<f32>,
    /// Zero-shot tagger: `off` | `clip`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tagger: Option<String>,
    /// Minimum CLIP tag score to keep (`0.0..=1.0`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag_threshold: Option<f32>,
    /// Maximum tags kept per file (`1..=32`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tag_top_k: Option<usize>,
    /// Captioner: `off` | `florence2`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub captioner: Option<String>,
    /// Maximum video keyframes analysed (`1..=64`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_frames: Option<usize>,
    /// Per-file vision timeout in seconds (`5..=1800`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_secs: Option<u64>,
}

impl VisionSettings {
    /// The base settings a job starts from: every knob — the numeric ones and the
    /// detector/tagger/captioner selections — read out of `config.vision`, whose
    /// sub-model defaults are the single `DETECTOR_DEFAULT`/`TAGGER_DEFAULT`/
    /// `CAPTIONER_DEFAULT` source (not re-spelled here). All fields `Some`, so
    /// merging a partial override always yields a complete result.
    pub fn from_config(config: &Config) -> Self {
        Self {
            detector: Some(config.vision.detector.clone()),
            detector_conf: Some(config.vision.detector_conf),
            tagger: Some(config.vision.tagger.clone()),
            tag_threshold: Some(config.vision.tag_score),
            tag_top_k: Some(config.vision.tag_top_k),
            captioner: Some(config.vision.captioner.clone()),
            max_frames: Some(config.vision.max_frames),
            timeout_secs: Some(config.vision.timeout_secs),
        }
    }

    /// THE merge path: each field of `over` wins when set, else `base` is kept.
    pub fn merge(base: &Self, over: &Self) -> Self {
        Self {
            detector: over.detector.clone().or_else(|| base.detector.clone()),
            detector_conf: over.detector_conf.or(base.detector_conf),
            tagger: over.tagger.clone().or_else(|| base.tagger.clone()),
            tag_threshold: over.tag_threshold.or(base.tag_threshold),
            tag_top_k: over.tag_top_k.or(base.tag_top_k),
            captioner: over.captioner.clone().or_else(|| base.captioner.clone()),
            max_frames: over.max_frames.or(base.max_frames),
            timeout_secs: over.timeout_secs.or(base.timeout_secs),
        }
    }

    /// Resolve the effective settings for a job through the single
    /// [`merge`](Self::merge) path. `None` ⇒ exactly the config defaults.
    pub fn resolve(config: &Config, over: Option<&Self>) -> Self {
        let base = Self::from_config(config);
        match over {
            Some(over) => Self::merge(&base, over),
            None => base,
        }
    }

    /// Write the resolved knobs back into `config.vision` so the vision pipeline
    /// carries the per-job values. The detector/tagger/captioner toggles are
    /// normalized (trim + lowercase) and stored too; `VisionAnalyzer::analyze_image`
    /// consults `VisionConfig::{detector,tagger,captioner}_enabled`, so an `off`
    /// toggle actually SKIPS that sub-tier (no model load, no cost) instead of
    /// being accepted-and-ignored.
    pub fn apply_to(&self, config: &mut Config) {
        if let Some(value) = &self.detector {
            config.vision.detector = value.trim().to_ascii_lowercase();
        }
        if let Some(value) = self.detector_conf {
            config.vision.detector_conf = value;
        }
        if let Some(value) = &self.tagger {
            config.vision.tagger = value.trim().to_ascii_lowercase();
        }
        if let Some(value) = self.tag_threshold {
            config.vision.tag_score = value;
        }
        if let Some(value) = self.tag_top_k {
            config.vision.tag_top_k = value;
        }
        if let Some(value) = &self.captioner {
            config.vision.captioner = value.trim().to_ascii_lowercase();
        }
        if let Some(value) = self.max_frames {
            config.vision.max_frames = value;
        }
        if let Some(value) = self.timeout_secs {
            config.vision.timeout_secs = value;
        }
    }

    /// Validate every set field's range/enum. Returns a field-specific message on
    /// the first violation.
    pub fn validate(&self) -> Result<(), String> {
        validate_enum("detector", &self.detector, DETECTORS)?;
        validate_enum("tagger", &self.tagger, TAGGERS)?;
        validate_enum("captioner", &self.captioner, CAPTIONERS)?;
        if let Some(conf) = self.detector_conf {
            if !(DETECTOR_CONF_MIN..=DETECTOR_CONF_MAX).contains(&conf) {
                return Err(format!(
                    "vision.detector_conf must be between {DETECTOR_CONF_MIN} and {DETECTOR_CONF_MAX} (got {conf})"
                ));
            }
        }
        if let Some(threshold) = self.tag_threshold {
            if !(TAG_THRESHOLD_MIN..=TAG_THRESHOLD_MAX).contains(&threshold) {
                return Err(format!(
                    "vision.tag_threshold must be between {TAG_THRESHOLD_MIN} and {TAG_THRESHOLD_MAX} (got {threshold})"
                ));
            }
        }
        if let Some(top_k) = self.tag_top_k {
            if !(TAG_TOP_K_MIN..=TAG_TOP_K_MAX).contains(&top_k) {
                return Err(format!(
                    "vision.tag_top_k must be between {TAG_TOP_K_MIN} and {TAG_TOP_K_MAX} (got {top_k})"
                ));
            }
        }
        if let Some(frames) = self.max_frames {
            if !(MAX_FRAMES_MIN..=MAX_FRAMES_MAX).contains(&frames) {
                return Err(format!(
                    "vision.max_frames must be between {MAX_FRAMES_MIN} and {MAX_FRAMES_MAX} (got {frames})"
                ));
            }
        }
        if let Some(timeout) = self.timeout_secs {
            if !(TIMEOUT_MIN..=TIMEOUT_MAX).contains(&timeout) {
                return Err(format!(
                    "vision.timeout_secs must be between {TIMEOUT_MIN} and {TIMEOUT_MAX} (got {timeout})"
                ));
            }
        }
        Ok(())
    }
}

/// Validate an optional enum-style string field against its accepted set,
/// comparing case-insensitively. Returns a `vision.<field> must be one of …`
/// message naming the offending value.
fn validate_enum(field: &str, value: &Option<String>, allowed: &[&str]) -> Result<(), String> {
    if let Some(value) = value {
        let normalized = value.trim().to_ascii_lowercase();
        if !allowed.contains(&normalized.as_str()) {
            return Err(format!(
                "vision.{field} must be one of: {} (got '{value}')",
                allowed.join(", ")
            ));
        }
    }
    Ok(())
}

/// The bundled and system tessdata language sets kept SEPARATE, so validation can
/// mirror `TesseractOcr`'s all-or-nothing-per-source resolution (a job's
/// languages must all load from one source). `.0` is the bundled
/// `<data_dir>/tessdata` set; `.1` is what `tesseract --list-langs` reports.
pub fn tessdata_sources(config: &Config) -> (BTreeSet<String>, BTreeSet<String>) {
    (
        langs_in_dir(&config.data_dir.join("tessdata")),
        system_tessdata_langs(&config.tesseract_cmd),
    )
}

/// Every tesseract language available to a job under `config` — the bundled
/// `<data_dir>/tessdata` set unioned with the system packs — for the
/// `GET /settings` discovery list. (Whether a specific combo can LOAD together is
/// the per-source question [`OcrSettings::validate_langs`] answers.) Never a
/// hardcoded list (SETTINGS-SPEC section 2).
pub fn installed_tessdata_langs(config: &Config) -> BTreeSet<String> {
    let (mut langs, system) = tessdata_sources(config);
    langs.extend(system);
    langs
}

/// The `*.traineddata` language stems present directly under `dir` (empty when
/// the directory is absent or unreadable).
pub fn langs_in_dir(dir: &Path) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("traineddata") {
                if let Some(stem) = path.file_stem().and_then(|stem| stem.to_str()) {
                    out.insert(stem.to_string());
                }
            }
        }
    }
    out
}

/// Languages `tesseract --list-langs` reports (its own resolution of the system
/// tessdata / `TESSDATA_PREFIX`), excluding the non-language `osd`
/// script/orientation model. Empty when tesseract is not runnable.
fn system_tessdata_langs(tesseract_cmd: &str) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    let Ok(output) = Command::new(tesseract_cmd).arg("--list-langs").output() else {
        return out;
    };
    if !output.status.success() {
        return out;
    }
    // The header line ("List of available languages …") carries spaces and a
    // colon; genuine language codes are bare single tokens. Filter structurally
    // rather than by a fixed skip count, which differs across tesseract builds.
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let lang = line.trim();
        if lang.is_empty() || lang == "osd" {
            continue;
        }
        if lang.contains(char::is_whitespace) || lang.contains(':') {
            continue;
        }
        out.insert(lang.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ocr::TesseractOcr;

    fn config_with_ocr(dpi: u32, psm: &str, langs: &str) -> Config {
        let mut config = Config::default();
        config.ocr_dpi = dpi;
        config.ocr_psm = psm.to_string();
        config.ocr_langs = langs.to_string();
        config
    }

    #[test]
    fn merge_is_the_single_precedence_path() {
        let base = OcrSettings {
            dpi: Some(300),
            psm: Some("3".into()),
            preprocess: Some(true),
            max_pages: Some(20),
            langs: Some("vie+eng".into()),
        };
        // A set override field wins; an unset one keeps the base.
        let over = OcrSettings {
            dpi: Some(700),
            psm: None,
            preprocess: Some(false),
            max_pages: None,
            langs: Some("eng".into()),
        };
        let merged = OcrSettings::merge(&base, &over);
        assert_eq!(merged.dpi, Some(700)); // override wins
        assert_eq!(merged.psm.as_deref(), Some("3")); // base kept
        assert_eq!(merged.preprocess, Some(false)); // override wins
        assert_eq!(merged.max_pages, Some(20)); // base kept
        assert_eq!(merged.langs.as_deref(), Some("eng")); // override wins
    }

    #[test]
    fn resolve_orders_job_over_config_over_builtin() {
        // Built-in default (Config::default has dpi 300), no override.
        let builtin = OcrSettings::resolve(&Config::default(), None);
        assert_eq!(builtin.dpi, Some(300));

        // Service config sets dpi 200; still no per-job override -> config wins
        // over built-in.
        let configured = config_with_ocr(200, "3", "vie+eng");
        assert_eq!(OcrSettings::resolve(&configured, None).dpi, Some(200));

        // Per-job override sets dpi 500 -> wins over both config and built-in.
        let over = OcrSettings {
            dpi: Some(500),
            ..Default::default()
        };
        assert_eq!(
            OcrSettings::resolve(&configured, Some(&over)).dpi,
            Some(500)
        );
    }

    #[test]
    fn ocr_bounds_accept_edges_and_reject_beyond() {
        let ok = |s: &OcrSettings| s.validate().is_ok();
        // dpi edges.
        assert!(ok(&OcrSettings {
            dpi: Some(150),
            ..Default::default()
        }));
        assert!(ok(&OcrSettings {
            dpi: Some(1200),
            ..Default::default()
        }));
        assert!(OcrSettings {
            dpi: Some(149),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("ocr.dpi"));
        assert!(OcrSettings {
            dpi: Some(1201),
            ..Default::default()
        }
        .validate()
        .is_err());
        // psm edges + non-numeric.
        assert!(ok(&OcrSettings {
            psm: Some("0".into()),
            ..Default::default()
        }));
        assert!(ok(&OcrSettings {
            psm: Some("13".into()),
            ..Default::default()
        }));
        assert!(OcrSettings {
            psm: Some("14".into()),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(OcrSettings {
            psm: Some("auto".into()),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("ocr.psm"));
        // max_pages edges.
        assert!(ok(&OcrSettings {
            max_pages: Some(1),
            ..Default::default()
        }));
        assert!(ok(&OcrSettings {
            max_pages: Some(500),
            ..Default::default()
        }));
        assert!(OcrSettings {
            max_pages: Some(0),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(OcrSettings {
            max_pages: Some(501),
            ..Default::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn vision_bounds_accept_edges_and_reject_beyond() {
        let edge = VisionSettings {
            detector_conf: Some(0.05),
            tag_threshold: Some(0.0),
            tag_top_k: Some(1),
            max_frames: Some(1),
            timeout_secs: Some(5),
            ..Default::default()
        };
        assert!(edge.validate().is_ok());
        let edge_high = VisionSettings {
            detector_conf: Some(0.95),
            tag_threshold: Some(1.0),
            tag_top_k: Some(32),
            max_frames: Some(64),
            timeout_secs: Some(1800),
            ..Default::default()
        };
        assert!(edge_high.validate().is_ok());

        assert!(VisionSettings {
            detector_conf: Some(0.04),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("detector_conf"));
        assert!(VisionSettings {
            tag_threshold: Some(1.01),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(VisionSettings {
            tag_top_k: Some(33),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(VisionSettings {
            max_frames: Some(65),
            ..Default::default()
        }
        .validate()
        .is_err());
        assert!(VisionSettings {
            timeout_secs: Some(4),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("timeout_secs"));
    }

    #[test]
    fn vision_enum_toggles_validate() {
        for value in ["off", "nano", "NANO"] {
            assert!(VisionSettings {
                detector: Some(value.into()),
                ..Default::default()
            }
            .validate()
            .is_ok());
        }
        assert!(VisionSettings {
            detector: Some("yolo".into()),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("vision.detector"));
        assert!(VisionSettings {
            tagger: Some("blip".into()),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("vision.tagger"));
        assert!(VisionSettings {
            captioner: Some("gpt".into()),
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .contains("vision.captioner"));
    }

    #[test]
    fn langs_validation_rejects_uninstalled_packs() {
        let bundled: BTreeSet<String> = ["eng", "vie"].iter().map(|s| s.to_string()).collect();
        let system: BTreeSet<String> = BTreeSet::new();
        // Every requested language installed -> ok.
        assert!(OcrSettings {
            langs: Some("vie+eng".into()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .is_ok());
        // rus is absent from both sources -> rejected, message names the missing pack.
        let error = OcrSettings {
            langs: Some("vie+rus".into()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .unwrap_err();
        assert!(error.contains("rus"), "{error}");
        assert!(error.contains("ocr.langs"), "{error}");
        // No langs field -> nothing to validate.
        assert!(OcrSettings::default()
            .validate_langs(&bundled, &system)
            .is_ok());
    }

    #[test]
    fn langs_validation_rejects_cross_source_combo() {
        // vie is bundled-only, fra is system-only. Both are installed, but tesseract
        // reads ONE source, so `vie+fra` cannot load together — it must be rejected
        // at submit rather than silently OCR'ing every page empty at run time.
        let bundled: BTreeSet<String> = ["vie", "eng"].iter().map(|s| s.to_string()).collect();
        let system: BTreeSet<String> = ["eng", "fra"].iter().map(|s| s.to_string()).collect();
        let error = OcrSettings {
            langs: Some("vie+fra".into()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .unwrap_err();
        assert!(error.contains("vie"), "{error}");
        assert!(error.contains("fra"), "{error}");
        // A combo entirely within one source loads fine: eng+fra ⊆ system…
        assert!(OcrSettings {
            langs: Some("eng+fra".into()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .is_ok());
        // …and vie+eng ⊆ bundled.
        assert!(OcrSettings {
            langs: Some("vie+eng".into()),
            ..Default::default()
        }
        .validate_langs(&bundled, &system)
        .is_ok());
    }

    #[test]
    fn apply_to_writes_canonical_langs_string() {
        // Whitespace / empty '+' components validate (per-component trim) but must
        // NOT reach the tesseract -l invocation raw — apply_to writes the canonical
        // form so a validated value is exactly what runs.
        let mut config = Config::default();
        OcrSettings {
            langs: Some("  vie +  eng ".into()),
            ..Default::default()
        }
        .apply_to(&mut config);
        assert_eq!(config.ocr_langs, "vie+eng");

        let mut config = Config::default();
        OcrSettings {
            langs: Some("vie++eng".into()),
            ..Default::default()
        }
        .apply_to(&mut config);
        assert_eq!(config.ocr_langs, "vie+eng");
    }

    #[test]
    fn vision_toggles_reach_vision_config() {
        let mut config = Config::default();
        // The base selections come from the single DETECTOR/TAGGER/CAPTIONER_DEFAULT
        // source, not a literal re-spelled in from_config.
        assert_eq!(
            VisionSettings::from_config(&config).detector.as_deref(),
            Some(DETECTOR_DEFAULT)
        );
        // `off` (any case) is honored: it is applied to config.vision and disables
        // the sub-tier via the enabled-helpers analyze_image consults.
        let over = VisionSettings {
            detector: Some("off".into()),
            tagger: Some("OFF".into()),
            ..Default::default()
        };
        VisionSettings::resolve(&config, Some(&over)).apply_to(&mut config);
        assert_eq!(config.vision.detector, "off");
        assert_eq!(config.vision.tagger, "off"); // normalized to lowercase
        assert!(!config.vision.detector_enabled());
        assert!(!config.vision.tagger_enabled());
        assert!(config.vision.captioner_enabled()); // untouched default stays on
    }

    #[test]
    fn langs_in_dir_enumerates_traineddata_stems() {
        let dir = tempfile::tempdir().unwrap();
        for name in ["eng.traineddata", "vie.traineddata", "readme.txt"] {
            std::fs::write(dir.path().join(name), b"x").unwrap();
        }
        let langs = langs_in_dir(dir.path());
        assert_eq!(langs.len(), 2);
        assert!(langs.contains("eng"));
        assert!(langs.contains("vie"));
        assert!(!langs.contains("readme"));
        // An absent directory enumerates to empty rather than erroring.
        assert!(langs_in_dir(&dir.path().join("nope")).is_empty());
    }

    #[test]
    fn per_job_settings_reach_ocr_config_and_tesseract_handle() {
        // The seam: a submitted per-job override resolved to effective settings…
        let mut config = config_with_ocr(300, "3", "vie+eng");
        config.finalize();
        let over = OcrSettings {
            dpi: Some(700),
            psm: Some("7".into()),
            preprocess: Some(false),
            max_pages: Some(5),
            langs: Some("eng".into()),
        };
        let effective = OcrSettings::resolve(&config, Some(&over));
        assert_eq!(effective.dpi, Some(700));
        assert_eq!(effective.psm.as_deref(), Some("7"));

        // …applied onto the config the pipeline threads everywhere.
        effective.apply_to(&mut config);
        // Reaches the config the PDF rasterizer reads (`pdftoppm -r <ocr_dpi>`,
        // `-l <ocr_max_pages>` in extract.rs) and the preprocess toggle.
        assert_eq!(config.ocr_dpi, 700);
        assert_eq!(config.ocr_max_pages, 5);
        assert!(!config.ocr_preprocess);
        // Reaches the OCR invocation builder itself (TesseractOcr).
        let ocr = TesseractOcr::new(&config);
        assert_eq!(ocr.langs, "eng");
        assert_eq!(ocr.psm(), "7");
    }

    #[test]
    fn cli_and_http_overrides_resolve_identically() {
        // The HTTP shape (`ocr_opts` JSON) and the native CLI flags converge on
        // the SAME OcrSettings, so both feed the one merge path.
        let http: OcrSettings = serde_json::from_value(serde_json::json!({
            "dpi": 700, "psm": "7", "preprocess": false, "max_pages": 5
        }))
        .unwrap();
        let cli = OcrSettings {
            dpi: Some(700),
            psm: Some("7".into()),
            preprocess: Some(false),
            max_pages: Some(5),
            langs: None,
        };
        assert_eq!(http, cli);

        let config = config_with_ocr(300, "3", "vie+eng");
        assert_eq!(
            OcrSettings::resolve(&config, Some(&http)),
            OcrSettings::resolve(&config, Some(&cli))
        );

        // Unknown fields are permissively ignored (forward-compat serde posture).
        let forward: OcrSettings =
            serde_json::from_value(serde_json::json!({"dpi": 300, "future_knob": 9})).unwrap();
        assert_eq!(forward.dpi, Some(300));
    }

    #[test]
    fn absent_opts_leave_config_unchanged() {
        // The off-path (no ocr_opts / vision_opts) must be byte-identical: a
        // None-override resolve+apply reproduces the config verbatim.
        let mut config = Config::default();
        config.finalize();
        let before = (
            config.ocr_dpi,
            config.ocr_psm.clone(),
            config.ocr_preprocess,
            config.ocr_max_pages,
            config.ocr_langs.clone(),
            config.vision.detector_conf,
            config.vision.tag_score,
            config.vision.tag_top_k,
            config.vision.max_frames,
            config.vision.timeout_secs,
        );
        OcrSettings::resolve(&config, None).apply_to(&mut config);
        VisionSettings::resolve(&config, None).apply_to(&mut config);
        let after = (
            config.ocr_dpi,
            config.ocr_psm.clone(),
            config.ocr_preprocess,
            config.ocr_max_pages,
            config.ocr_langs.clone(),
            config.vision.detector_conf,
            config.vision.tag_score,
            config.vision.tag_top_k,
            config.vision.max_frames,
            config.vision.timeout_secs,
        );
        assert_eq!(before, after);
    }

    #[test]
    fn vision_opts_reach_vision_config() {
        let mut config = Config::default();
        let over = VisionSettings {
            detector_conf: Some(0.6),
            tag_threshold: Some(0.3),
            tag_top_k: Some(16),
            max_frames: Some(24),
            timeout_secs: Some(120),
            ..Default::default()
        };
        VisionSettings::resolve(&config, Some(&over)).apply_to(&mut config);
        assert_eq!(config.vision.detector_conf, 0.6);
        assert_eq!(config.vision.tag_score, 0.3);
        assert_eq!(config.vision.tag_top_k, 16);
        assert_eq!(config.vision.max_frames, 24);
        assert_eq!(config.vision.timeout_secs, 120);
    }
}
