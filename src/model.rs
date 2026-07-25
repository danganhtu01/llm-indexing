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
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndexStats {
    pub files: usize,
    pub bytes: u64,
    pub ocr_files: usize,
    pub errors: usize,
    /// The subset of `errors` that are password-protected PDFs (B2's
    /// `error:encrypted` rows) — broken out so the operator can see "N
    /// encrypted PDFs skipped" without a SQL query, same motivation as
    /// `capped` being split out from `skipped` below.
    pub encrypted: usize,
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
    /// Faces stored by this run, summed over files. Zero on every job that did
    /// not turn the opt-in faces sub-tier on (and on every job whose box has no
    /// face models staged), so an operator can tell at a glance whether the
    /// capability actually did anything. Deliberately a COUNT and nothing more:
    /// no paths, no boxes, no vectors leave the corpus in a job summary.
    pub faces: usize,
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
