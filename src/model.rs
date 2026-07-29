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
    /// The subset of `errors` that are password-protected PDFs (B2's
    /// `error:encrypted` rows) — broken out so the operator can see "N
    /// encrypted PDFs skipped" without a SQL query, same motivation as
    /// `capped` being split out from `skipped` below.
    pub encrypted: usize,
    /// Walked files this run will not touch at all. Rows the sha1 backfill lane
    /// claimed are NOT here — the lane does touch them (see `hashed`).
    pub skipped: usize,
    /// The subset of `skipped` that resume declined because the row has already
    /// burned its attempt budget rather than because it is finished. Reported
    /// separately because the two are opposite outcomes wearing the same number:
    /// a large `skipped` is a resume working, a large `capped` is a corpus with
    /// work it has given up on. Still a strict subset: the backfill lane never
    /// claims a capped row, precisely so this stays true.
    pub capped: usize,
    /// Rows the sha1 BACKFILL lane hashed: already-finished rows that carried no
    /// `sha1`, given one without their content being touched.
    ///
    /// Not a subset of anything else here. The lane's rows count in the run's
    /// `total`/`processed` (a hash pass is real, rate-predictive work, and an ETA
    /// derived from those counters is only honest if it can see it) and are
    /// deliberately EXCLUDED from `skipped`, which means "this run will not touch
    /// this row". They are also disjoint from `files`, which counts rows this run
    /// WROTE from a `ProcessedFile`: a backfilled row was never extracted, never
    /// embedded and never re-written, so a pure backfill pass reports
    /// `files: 0`, `embedded_chunks: 0` and `hashed: N`.
    ///
    /// Zero on every run that did not turn `hash_backfill` on, which is every run
    /// by default.
    pub hashed: usize,
    /// Rows the sha1 backfill lane claimed and could NOT hash, because their file
    /// would not open or would not read: locked mail stores, VM disks a
    /// hypervisor holds open, anything the account running the job cannot read.
    ///
    /// Its own counter rather than a fold into `errors`, on purpose. What `errors`
    /// means is "this run PROCESSED a file and the processing failed" — it is
    /// incremented for every incoming `error:` method, and most of those do land
    /// as `error:` rows a reader can go find. Not all of them, and the exception
    /// is worth stating rather than glossing: the keep-on-failure branch counts
    /// the failure and then deliberately keeps the old complete row instead of
    /// replacing it, so that one has no `error:` row behind it. (`encrypted` is a
    /// strict subset of `errors`. `incomplete` is NOT — it also counts
    /// `name-only` and `-partial` rows, which are not errors.)
    ///
    /// A hash miss is not that. Nothing was processed: no extraction was
    /// attempted, no method was produced, and the stored row is a successful
    /// extraction that is still true — a hash that could not be taken says
    /// nothing about it. Folding the two would put "could not open a file to hash
    /// it" inside a counter that otherwise means "tried to index a file and
    /// failed", for nothing.
    ///
    /// (An earlier revision of this comment justified the split by a
    /// `worked = processed` identity it said F-C would compose. That identity
    /// does not hold and must not: `worked` counts INDEXED files only, so the
    /// whole backfill lane — hits and misses alike — is deliberately excluded
    /// from it. See [`crate::pipeline::Progress::worked`]. Nothing about this
    /// counter's existence rests on that; the SILENT DROP argument below is the
    /// reason, and it stands on its own.)
    ///
    /// It exists because the alternative is a SILENT DROP. The lane's rows leave
    /// `skipped` by construction, and a miss puts nothing in `hashed`, nothing in
    /// `files` and nothing in `errors` — so without this the only trace would be
    /// an unexplained gap between the owed count a run announces and the hashes
    /// it produces, indistinguishable from a bug in the lane. **The accounting
    /// closes here:** for a lane that ran to completion,
    /// `hashed + hash_failed` is exactly the owed count.
    ///
    /// It is also the convergence signal. These are precisely the rows that never
    /// acquire a `sha1`, so the lane re-claims every one of them on every armed
    /// run; the backfill is done when this is all that is left.
    pub hash_failed: usize,
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
