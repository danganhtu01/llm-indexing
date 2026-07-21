use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use tempfile::TempDir;

use crate::config::Config;
use crate::runtime::RuntimeKnobs;

/// Tesseract driver. Prefers the bundled `tessdata_best` LSTM models (fetched
/// into `<data_dir>/tessdata` at image build) over the distribution's fast
/// integer models, optionally cleans inputs up with ImageMagick first, and
/// falls back to sparse-text segmentation when a page comes back empty.
#[derive(Debug, Clone)]
pub struct TesseractOcr {
    command: String,
    pub langs: String,
    pub available: bool,
    psm: String,
    tessdata: Option<PathBuf>,
    preprocess_cmd: Option<String>,
    /// Live stage settings, consulted at SPAWN TIME rather than captured here,
    /// so retuning `ocr` mid-job lands on the next file. `None` (the CLI and
    /// probe paths) keeps the pre-existing behaviour of not setting
    /// `OMP_THREAD_LIMIT` at all.
    runtime: Option<Arc<RuntimeKnobs>>,
}

impl TesseractOcr {
    pub fn new(config: &Config) -> Self {
        let command = config.tesseract_cmd.clone();
        let available = Command::new(&command)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if !available {
            // Silent degradation here used to look like "indexing worked but
            // every scan is empty" — make the misconfiguration loud instead.
            tracing::warn!(
                command = %command,
                "tesseract is not runnable — OCR is DISABLED and scanned documents will index empty"
            );
        }
        let best = config.data_dir.join("tessdata");
        // Only prefer the bundled tessdata_best models when EVERY configured OCR
        // language has a traineddata file there. A single missing model (e.g.
        // rus/deu when the bundle only shipped vie/eng) would otherwise make
        // tesseract fail for that language, so fall back to the system models.
        let langs: Vec<&str> = config
            .ocr_langs
            .split('+')
            .filter(|l| !l.is_empty())
            .collect();
        let bundle_complete = !langs.is_empty()
            && langs
                .iter()
                .all(|lang| best.join(format!("{lang}.traineddata")).exists());
        let tessdata = bundle_complete.then_some(best);
        match &tessdata {
            Some(dir) => {
                tracing::info!(tessdata = %dir.display(), "OCR using bundled tessdata_best models")
            }
            None => tracing::info!(
                langs = %config.ocr_langs,
                "OCR using system tessdata models (tessdata_best not bundled for every language)"
            ),
        }
        let preprocess_cmd = if config.ocr_preprocess {
            let found = ["magick", "convert"].into_iter().find(|candidate| {
                Command::new(candidate)
                    .arg("--version")
                    .output()
                    .map(|output| output.status.success())
                    .unwrap_or(false)
            });
            if found.is_none() {
                tracing::warn!(
                    "ocr_preprocess is enabled but ImageMagick is missing — OCR runs on raw images"
                );
            }
            found.map(str::to_string)
        } else {
            None
        };
        Self {
            command,
            langs: config.ocr_langs.clone(),
            available,
            psm: config.ocr_psm.clone(),
            tessdata,
            preprocess_cmd,
            runtime: None,
        }
    }

    /// Attach live stage settings. The handle keeps the [`Arc`], not the value,
    /// so every spawn re-reads the current `ocr` setting.
    pub fn with_runtime(mut self, runtime: Arc<RuntimeKnobs>) -> Self {
        self.runtime = Some(runtime);
        self
    }

    /// The page-segmentation mode this handle drives tesseract with. Per-job
    /// `ocr_opts.psm` flows here through the settings merge (`settings.rs`).
    pub fn psm(&self) -> &str {
        &self.psm
    }

    pub fn image_to_text(&self, path: &Path) -> String {
        if !self.available {
            return String::new();
        }
        // Bind the temp dir so the cleaned image outlives the tesseract run.
        let cleaned = self.preprocess(path);
        let input = cleaned.as_ref().map(|(_, p)| p.as_path()).unwrap_or(path);
        let text = self.run_tesseract(input, &self.psm);
        if !text.trim().is_empty() || self.psm != "3" {
            return text;
        }
        // Fully-automatic segmentation found nothing — retry assuming a single
        // uniform block (stamps, tables, sparse scans).
        self.run_tesseract(input, "6")
    }

    /// `OMP_THREAD_LIMIT` for the NEXT tesseract spawn, or `None` to leave the
    /// environment alone.
    ///
    /// Resolved per call rather than cached on the handle, which is precisely
    /// what makes the `ocr` stage live: tesseract is spawned once per file, so
    /// re-reading here means a change applies to the next file with no restart
    /// of the job, the pool, or the process. Caching it at construction — the
    /// obvious simplification — would silently demote this to a next-job knob.
    fn omp_thread_limit(&self) -> Option<String> {
        self.runtime
            .as_ref()
            .map(|runtime| runtime.ocr().to_string())
    }

    fn run_tesseract(&self, path: &Path, psm: &str) -> String {
        let mut command = Command::new(&self.command);
        command
            .arg(path)
            .arg("stdout")
            .args(["-l", &self.langs, "--oem", "1", "--psm", psm]);
        if let Some(dir) = &self.tessdata {
            command.env("TESSDATA_PREFIX", dir);
        }
        if let Some(limit) = self.omp_thread_limit() {
            command.env("OMP_THREAD_LIMIT", limit);
        }
        match command.output() {
            Ok(output) if output.status.success() => {
                String::from_utf8_lossy(&output.stdout).to_string()
            }
            Ok(output) => {
                tracing::warn!(
                    path = %path.display(),
                    status = %output.status,
                    stderr = %String::from_utf8_lossy(&output.stderr).trim(),
                    "tesseract failed for page"
                );
                String::new()
            }
            Err(error) => {
                tracing::warn!(path = %path.display(), %error, "tesseract could not be spawned");
                String::new()
            }
        }
    }

    #[cfg(test)]
    fn thread_limit_for_test(&self) -> Option<String> {
        self.omp_thread_limit()
    }

    /// Grayscale + deskew + contrast-stretch into a temp PNG. Failures fall
    /// back to the raw input — preprocessing must never lose a page.
    fn preprocess(&self, path: &Path) -> Option<(TempDir, PathBuf)> {
        let magick = self.preprocess_cmd.as_ref()?;
        let temp = tempfile::tempdir().ok()?;
        let out = temp.path().join("clean.png");
        let status = Command::new(magick)
            .arg(path)
            .args([
                "-colorspace",
                "Gray",
                "-deskew",
                "40%",
                "-contrast-stretch",
                "1%x1%",
            ])
            .arg(&out)
            .status();
        match status {
            Ok(code) if code.success() && out.exists() => Some((temp, out)),
            Ok(code) => {
                tracing::debug!(path = %path.display(), status = %code, "image preprocess failed; using raw input");
                None
            }
            Err(error) => {
                tracing::debug!(path = %path.display(), %error, "image preprocess could not be spawned; using raw input");
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::RuntimeKnobs;
    use serde_json::{json, Map, Value};

    fn apply(runtime: &RuntimeKnobs, body: Value) {
        let map: Map<String, Value> = body.as_object().expect("object").clone();
        runtime.apply(&map).expect("valid stage");
    }

    #[test]
    fn without_runtime_the_ocr_environment_is_untouched() {
        // The CLI and the `GET /settings` probe build handles this way. Setting
        // OMP_THREAD_LIMIT unconditionally would silently change how every
        // existing deployment's tesseract parallelises.
        let ocr = TesseractOcr::new(&Config::default());
        assert_eq!(ocr.thread_limit_for_test(), None);
    }

    #[test]
    fn the_ocr_thread_limit_is_re_read_for_every_spawn() {
        // The liveness claim for the `ocr` stage, pinned. tesseract is spawned
        // per file, so re-reading per spawn is what makes a change land on the
        // NEXT FILE. If this value were captured when the handle was built, the
        // second assertion would still see 3 and the stage would be a next-job
        // knob wearing a `live: true` label.
        let runtime = Arc::new(RuntimeKnobs::from_config(&Config::default()));
        apply(&runtime, json!({"ocr": 3}));
        let ocr = TesseractOcr::new(&Config::default()).with_runtime(runtime.clone());
        assert_eq!(ocr.thread_limit_for_test().as_deref(), Some("3"));

        // Retune WITHOUT rebuilding the handle — exactly what a mid-job
        // POST /jobs/{id}/runtime does.
        apply(&runtime, json!({"ocr": 7}));
        assert_eq!(
            ocr.thread_limit_for_test().as_deref(),
            Some("7"),
            "the handle must not have cached the old limit"
        );
    }

    #[test]
    fn the_ocr_default_matches_what_openmp_would_have_chosen() {
        // Defaulting to 1 would have been a silent, unrequested slowdown for
        // every single-worker deployment; defaulting to what OpenMP would have
        // picked makes setting the variable a no-op relative to leaving it
        // unset. An operator's own OMP_THREAD_LIMIT wins over the CPU count,
        // because this stage OVERWRITES that variable on the child process and
        // must not quietly raise a limit someone deliberately lowered.
        let expected = std::env::var("OMP_THREAD_LIMIT")
            .ok()
            .and_then(|raw| raw.trim().parse::<usize>().ok())
            .filter(|limit| *limit > 0)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        let runtime = RuntimeKnobs::from_config(&Config::default());
        assert_eq!(runtime.ocr(), expected.clamp(1, 64));
    }
}
