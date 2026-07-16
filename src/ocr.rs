use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

use crate::config::Config;

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
        let tessdata = best.join("vie.traineddata").exists().then_some(best);
        match &tessdata {
            Some(dir) => {
                tracing::info!(tessdata = %dir.display(), "OCR using bundled tessdata_best models")
            }
            None => tracing::info!("OCR using system tessdata models (tessdata_best not bundled)"),
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
        }
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

    fn run_tesseract(&self, path: &Path, psm: &str) -> String {
        let mut command = Command::new(&self.command);
        command
            .arg(path)
            .arg("stdout")
            .args(["-l", &self.langs, "--oem", "1", "--psm", psm]);
        if let Some(dir) = &self.tessdata {
            command.env("TESSDATA_PREFIX", dir);
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
