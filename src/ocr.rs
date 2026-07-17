use std::path::Path;
use std::process::Command;

use crate::config::Config;

#[derive(Debug, Clone)]
pub struct TesseractOcr {
    command: String,
    pub langs: String,
    pub available: bool,
}

impl TesseractOcr {
    pub fn new(config: &Config) -> Self {
        let command = config.tesseract_cmd.clone();
        let available = Command::new(&command)
            .arg("--version")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        Self {
            command,
            langs: config.ocr_langs.clone(),
            available,
        }
    }

    pub fn image_to_text(&self, path: &Path) -> String {
        if !self.available {
            return String::new();
        }
        Command::new(&self.command)
            .arg(path)
            .arg("stdout")
            .args(["-l", &self.langs, "--oem", "1", "--psm", "3"])
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).to_string())
            .unwrap_or_default()
    }
}
