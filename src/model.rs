use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
pub struct FileRec {
    pub path: String,
    pub name: String,
    pub ext: String,
    pub dir: String,
    pub drive: String,
    pub size: u64,
    pub mtime: f64,
}

#[derive(Debug, Clone)]
pub struct ProcessedFile {
    pub rec: FileRec,
    pub content: String,
    pub tokens: Vec<String>,
    pub lang: String,
    pub method: String,
    pub ocr_used: bool,
    pub pages: usize,
    pub sha1: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub files: usize,
    pub bytes: u64,
    pub ocr_files: usize,
    pub errors: usize,
    pub skipped: usize,
    pub elapsed_seconds: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub path: String,
    pub dir: String,
    pub lang: String,
    pub method: String,
    pub size: u64,
    pub snippet: String,
}
