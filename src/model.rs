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
    /// Wall time this file cost, extraction + vision + embedding, stored on the
    /// row. The point is the rows that never finish: without it the price of a
    /// file that fails on every resume is invisible, and "how much of this run
    /// went into work that produced nothing" is unanswerable from the corpus.
    /// The stage that spent it is already in `method` (`error:embed:…` against
    /// any other `error:…`).
    pub elapsed_ms: u64,
    /// Carried straight from `Extracted::page_segments` (see `extract.rs`):
    /// `(page_number, text)` pairs for the extraction paths that can
    /// attribute text to a page. Empty for everything else. The embedder
    /// chunks these directly when present so `EmbeddedChunk::page_start`/
    /// `page_end` — and from there `VectorHit`'s — are populated (P0-8).
    pub page_segments: Vec<(usize, String)>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub files: usize,
    pub bytes: u64,
    pub ocr_files: usize,
    pub errors: usize,
    pub skipped: usize,
    /// The subset of `skipped` that resume declined because the row has already
    /// burned its attempt budget rather than because it is finished. Reported
    /// separately because the two are opposite outcomes wearing the same number:
    /// a large `skipped` is a resume working, a large `capped` is a corpus with
    /// work it has given up on.
    pub capped: usize,
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
