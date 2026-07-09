//! The stage contract shared by every step of the L&C document pipeline.
//!
//! A *stage* is a container with its input directory bound read-only, its output
//! directory bound read-write, and a `run.json` written on completion. That is the
//! whole interface. Because it is the whole interface, the language a stage happens to
//! be written in is an implementation detail — stage 1 is Python today and Rust later,
//! and nothing downstream can tell.
//!
//! ```text
//! /srv/lc/compliance-team/   stage 0  rclone copy, every minute (see archff/rclone/)
//!         /index/           stage 1  index.sqlite, manifest.jsonl, sidecar/**.txt
//!         /redacted/        stage 2  redacted text mirror + redactions.jsonl
//!         /corpus/          stage 3  corpus.sqlite -- the document-generation database
//!         /generated/       stage 4  emitted .docx / PDF
//!         /state/           runs.sqlite, locks
//! ```
//!
//! # The security invariant
//!
//! `/srv/lc/compliance-team` and `/srv/lc/index` hold **un-redacted PII**. Only
//! `/srv/lc/redacted` and `/srv/lc/generated` may ever be reachable from an HTTP
//! handler. [`Stage::is_web_servable`] encodes that so a reviewer can grep for it.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Default pipeline root. Override with `LC_ROOT` (used by tests and by `docker run`).
pub const DEFAULT_LC_ROOT: &str = "/srv/lc";

/// Where the pipeline lives on this machine.
pub fn lc_root() -> PathBuf {
    std::env::var_os("LC_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_LC_ROOT))
}

/// One step of the pipeline. Ordering is the pipeline order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Stage {
    /// Stage 0 — the rclone mirror of the Compliance-Team Drive. Not run by us;
    /// `archff/rclone/compliance-team-copy.timer` owns it.
    Ingest,
    /// Stage 1 — walk, extract, OCR, index.
    Index,
    /// Stage 2 — strip PII from the extracted text.
    Redact,
    /// Stage 3 — build the document-generation database.
    Corpus,
    /// Stage 4 — emit documents.
    Generate,
}

impl Stage {
    pub const ALL: [Stage; 5] = [
        Stage::Ingest,
        Stage::Index,
        Stage::Redact,
        Stage::Corpus,
        Stage::Generate,
    ];

    /// Directory name under [`lc_root`] holding this stage's output.
    pub fn dir_name(self) -> &'static str {
        match self {
            Stage::Ingest => "compliance-team",
            Stage::Index => "index",
            Stage::Redact => "redacted",
            Stage::Corpus => "corpus",
            Stage::Generate => "generated",
        }
    }

    /// The systemd unit that runs this stage.
    pub fn unit(self) -> &'static str {
        match self {
            Stage::Ingest => "compliance-team-copy.service",
            Stage::Index => "lc-index.service",
            Stage::Redact => "lc-redact.service",
            Stage::Corpus => "lc-corpus.service",
            Stage::Generate => "lc-generate.service",
        }
    }

    /// Absolute output directory.
    pub fn output_dir(self) -> PathBuf {
        lc_root().join(self.dir_name())
    }

    /// The stage whose output this one consumes.
    pub fn input_stage(self) -> Option<Stage> {
        match self {
            Stage::Ingest => None,
            Stage::Index => Some(Stage::Ingest),
            Stage::Redact => Some(Stage::Index),
            Stage::Corpus => Some(Stage::Redact),
            Stage::Generate => Some(Stage::Corpus),
        }
    }

    /// **May the web app serve files from this stage's directory?**
    ///
    /// Only redacted artefacts and generated documents. `ingest` is the raw Drive mirror
    /// and `index` contains `index.sqlite` with full un-redacted document text; exposing
    /// either over HTTP would defeat the entire point of the redaction stage.
    pub fn is_web_servable(self) -> bool {
        matches!(self, Stage::Redact | Stage::Generate)
    }
}

/// Written to `<output_dir>/run.json` when a stage finishes, successfully or not.
///
/// This is the only thing the orchestrator reads to answer "did it work, and what did it
/// do". Keep it small, keep it stable, and never put document content in it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunManifest {
    pub stage: Stage,
    /// Unix seconds. Not a wall-clock string: timezones are somebody else's problem.
    pub started_at: u64,
    pub finished_at: u64,
    /// 0 = success. Mirrors the process exit code so `systemctl status` and this agree.
    pub exit_code: i32,
    /// Free-form counters: `files_seen`, `files_written`, `spans_redacted`, ...
    pub counts: serde_json::Map<String, serde_json::Value>,
    /// Non-fatal problems worth surfacing in the UI. A stage that fills this and still
    /// exits 0 is saying "I finished, but look at these".
    pub warnings: Vec<String>,
}

impl RunManifest {
    pub fn start(stage: Stage) -> Self {
        Self {
            stage,
            started_at: now_secs(),
            finished_at: 0,
            exit_code: 0,
            counts: serde_json::Map::new(),
            warnings: Vec::new(),
        }
    }

    pub fn count(&mut self, key: &str, n: u64) {
        self.counts.insert(key.to_string(), serde_json::json!(n));
    }

    pub fn warn(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }

    /// Stamp the finish time and write `<dir>/run.json` atomically.
    ///
    /// Atomic because the orchestrator polls this file: a half-written manifest read at
    /// the wrong moment would look like a crashed stage.
    pub fn finish(mut self, dir: &Path, exit_code: i32) -> Result<()> {
        self.finished_at = now_secs();
        self.exit_code = exit_code;

        std::fs::create_dir_all(dir)
            .with_context(|| format!("creating output dir {}", dir.display()))?;

        let tmp = dir.join("run.json.tmp");
        let final_path = dir.join("run.json");
        std::fs::write(&tmp, serde_json::to_vec_pretty(&self)?)
            .with_context(|| format!("writing {}", tmp.display()))?;
        std::fs::rename(&tmp, &final_path)
            .with_context(|| format!("renaming into {}", final_path.display()))?;
        Ok(())
    }

    pub fn read(dir: &Path) -> Result<Self> {
        let raw = std::fs::read(dir.join("run.json"))?;
        Ok(serde_json::from_slice(&raw)?)
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_redacted_and_generated_are_web_servable() {
        // If this test ever goes red, someone is about to serve un-redacted PII.
        assert!(!Stage::Ingest.is_web_servable());
        assert!(!Stage::Index.is_web_servable());
        assert!(!Stage::Corpus.is_web_servable());
        assert!(Stage::Redact.is_web_servable());
        assert!(Stage::Generate.is_web_servable());
    }

    #[test]
    fn stages_chain_back_to_ingest() {
        let mut s = Stage::Generate;
        let mut hops = 0;
        while let Some(prev) = s.input_stage() {
            s = prev;
            hops += 1;
            assert!(hops < 10, "cycle in stage graph");
        }
        assert_eq!(s, Stage::Ingest);
    }

    #[test]
    fn manifest_round_trips() {
        let dir = std::env::temp_dir().join(format!("lc-core-test-{}", std::process::id()));
        let mut m = RunManifest::start(Stage::Redact);
        m.count("files_seen", 42);
        m.warn("one file had no text layer");
        m.finish(&dir, 0).unwrap();

        let back = RunManifest::read(&dir).unwrap();
        assert_eq!(back.stage, Stage::Redact);
        assert_eq!(back.exit_code, 0);
        assert_eq!(back.counts["files_seen"], 42);
        assert_eq!(back.warnings.len(), 1);
        assert!(back.finished_at >= back.started_at);
        std::fs::remove_dir_all(&dir).ok();
    }
}
