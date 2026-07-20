use serde::{Deserialize, Serialize};

use crate::embedding::EmbeddedChunk;
use crate::vision::VisionResult;

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
    pub chunks: Vec<EmbeddedChunk>,
    /// Vision analysis for image/video files when a job opts in; `None` for the
    /// off-path and non-vision files.
    pub vision: Option<VisionResult>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub files: usize,
    pub bytes: u64,
    pub ocr_files: usize,
    pub errors: usize,
    pub skipped: usize,
    pub incomplete: usize,
    pub embedded_chunks: usize,
    pub removed: usize,
    pub vision_files: usize,
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
