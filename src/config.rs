use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

fn default_languages() -> Vec<String> {
    vec!["vi".into(), "en".into()]
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
    "vie+eng".into()
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
            data_dir: default_data_dir(),
            whisper_model: default_whisper_model(),
            embedding_cache: default_embedding_cache(),
            embedding_model: default_embedding_model(),
            skip_dirs: default_skip_dirs(),
            skip_exts: default_skip_exts(),
            follow_symlinks: false,
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
        self.workers = self.workers.clamp(1, 64);
        self.skip_dirs_upper = self.skip_dirs.iter().map(|s| s.to_uppercase()).collect();
        self.skip_exts_lower = self.skip_exts.iter().map(|s| s.to_lowercase()).collect();
    }

    pub fn skip_dir(&self, name: &str) -> bool {
        self.skip_dirs_upper.contains(&name.to_uppercase())
    }

    pub fn skip_ext(&self, ext: &str) -> bool {
        self.skip_exts_lower.contains(&ext.to_lowercase())
    }
}
