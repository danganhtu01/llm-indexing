//! Persisted job envelopes (`jobs.sqlite`) — the llm-indexing half of P0-11
//! (restart/crash reconciliation).
//!
//! `AppState.jobs` (see `service.rs`) is an in-memory map only: every job it
//! knows about is gone the moment the process restarts. That has two
//! consequences without this module. First, `GET /jobs/{id}` for a job the
//! caller has not yet polled to completion turns into a bare 404 after a
//! restart — indistinguishable from "this job id never existed" — instead of
//! "the service restarted while this job was in flight". Second, a job the
//! restart caught mid-run (`queued`/`running`/`cancelling`) is simply gone: no
//! terminal state is ever recorded for it, so a poller keeps retrying against
//! a job that will never answer again.
//!
//! This store keeps the same envelope `/jobs/{id}` already returns in a small
//! SQLite database beside the corpora this service publishes:
//!
//! - every status transition worth reconciling on (queued at submit, running
//!   at dispatch, cancelling on request, and every terminal state) is
//!   persisted here, NOT the frequent processed/total progress ticks, which
//!   would multiply writes for no reconciliation benefit;
//! - on startup, [`JobsStore::sweep_interrupted`] rewrites any row still
//!   `queued`/`running`/`cancelling` — there is no worker left to finish it —
//!   to a terminal `error` envelope naming the restart, so `GET /jobs/{id}`
//!   answers honestly instead of 404ing or hanging forever as "running";
//! - history is bounded ([`prune`](JobsStore::prune)) so a long-lived service
//!   does not grow this file forever.
//!
//! Schema follows the same `PRAGMA user_version` migration convention as
//! `store.rs`'s corpus schema: `MIGRATIONS` is append-only, never edited once
//! shipped — a new column or table is a new entry.

use std::path::Path;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

/// One entry per schema version, applied in order starting from wherever the
/// database's own `PRAGMA user_version` says it is — the same convention
/// `store::MIGRATIONS` uses. Append-only.
const MIGRATIONS: &[&str] = &[SCHEMA_V1];

const SCHEMA_V1: &str = r#"
CREATE TABLE IF NOT EXISTS jobs(
  id TEXT PRIMARY KEY,
  status TEXT NOT NULL,
  envelope_json TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_updated_at ON jobs(updated_at);
"#;

/// How many job envelopes `jobs.sqlite` retains. An unbounded history in a
/// long-lived service would grow this database forever; this is deliberately
/// its own constant rather than sharing `service::MAX_HISTORY` — the two
/// stores prune independently (the in-memory map and the persisted history
/// need not move together, and a future change to one must not silently
/// change the other).
pub const MAX_PERSISTED_HISTORY: usize = 500;

/// The error message stamped onto every job the startup sweep finds
/// non-terminal — named once so the sweep and its tests cannot drift apart.
pub const INTERRUPTED_ERROR: &str = "interrupted by service restart; resubmit with resume";

/// The reserved output filename this module owns. A job may not publish a
/// corpus named `jobs.sqlite` (it would collide with this store on disk), and
/// the corpus read surface (`/corpus/*`) refuses to open it as a corpus for
/// the same reason — `service::valid_output_name` enforces both.
pub const RESERVED_OUTPUT_NAME: &str = "jobs.sqlite";

fn migrate(connection: &Connection) -> Result<()> {
    let current: i64 = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
    let current = current.max(0) as usize;
    if current > MIGRATIONS.len() {
        anyhow::bail!(
            "jobs database is at schema version {current}, newer than this binary knows \
             (version {}); refusing to open it — upgrade llm-indexing before touching it",
            MIGRATIONS.len()
        );
    }
    for (index, statement) in MIGRATIONS.iter().enumerate().skip(current) {
        let version = index + 1;
        connection
            .execute_batch("BEGIN IMMEDIATE")
            .with_context(|| format!("opening the transaction for jobs migration {version}"))?;
        if let Err(error) = connection.execute_batch(statement) {
            // Best effort: if the rollback itself fails the connection is
            // already in an unknown state and the error below is what surfaces.
            let _ = connection.execute_batch("ROLLBACK");
            return Err(error).with_context(|| format!("applying jobs migration {version}"));
        }
        connection
            .pragma_update(None, "user_version", version as i64)
            .with_context(|| format!("stamping jobs user_version {version}"))?;
        connection
            .execute_batch("COMMIT")
            .with_context(|| format!("committing jobs migration {version}"))?;
    }
    Ok(())
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or_default()
}

/// A small SQLite-backed job-envelope store: one row per job, keyed by id.
///
/// Every method locks the connection for the span of one statement (or, for
/// [`sweep_interrupted`](Self::sweep_interrupted), one row-by-row rewrite) and
/// takes `&self` — callers reach it through an `Arc<JobsStore>` shared across
/// the async handlers and the worker loop, and every call site in
/// `service.rs` runs it inside `spawn_blocking`, so a lock held for one SQLite
/// statement never parks the async executor.
pub struct JobsStore {
    connection: Mutex<Connection>,
}

impl JobsStore {
    /// Open (creating if absent) `jobs.sqlite` under `output_root` and bring
    /// it to the current schema version.
    pub fn open(output_root: &Path) -> Result<Self> {
        std::fs::create_dir_all(output_root)?;
        let path = output_root.join(RESERVED_OUTPUT_NAME);
        let connection =
            Connection::open(&path).with_context(|| format!("opening {}", path.display()))?;
        connection.busy_timeout(crate::store::BUSY_TIMEOUT)?;
        migrate(&connection).context("applying jobs schema migrations")?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// An in-memory instance for tests that want the schema without a tempdir.
    #[cfg(test)]
    pub fn open_in_memory() -> Result<Self> {
        let connection = Connection::open_in_memory()?;
        migrate(&connection)?;
        Ok(Self {
            connection: Mutex::new(connection),
        })
    }

    /// Persist (insert or replace) one job's current envelope — the exact
    /// `Value` `GET /jobs/{id}` returns for it. Called on every status
    /// transition worth reconciling on: queued at submit, running at
    /// dispatch, cancelling on request, and every terminal state.
    pub fn record(&self, id: &str, envelope: &Value) -> Result<()> {
        let status = envelope["status"].as_str().unwrap_or("unknown").to_string();
        let envelope_json = serde_json::to_string(envelope)?;
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        connection.execute(
            "INSERT INTO jobs(id, status, envelope_json, updated_at) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(id) DO UPDATE SET status = excluded.status,
                envelope_json = excluded.envelope_json, updated_at = excluded.updated_at",
            params![id, status, envelope_json, now()],
        )?;
        Ok(())
    }

    /// The persisted envelope for `id`, if any — the fallback `GET /jobs/{id}`
    /// reads once the in-memory map no longer holds it (aged out, or the
    /// process restarted since).
    pub fn get(&self, id: &str) -> Result<Option<Value>> {
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let json: Option<String> = connection
            .query_row(
                "SELECT envelope_json FROM jobs WHERE id = ?1",
                params![id],
                |row| row.get(0),
            )
            .optional()?;
        json.map(|text| Ok(serde_json::from_str(&text)?))
            .transpose()
    }

    /// Rewrite every non-terminal row (`queued`/`running`/`cancelling`) to a
    /// terminal `error` envelope naming the restart. Run once at startup,
    /// before the HTTP server binds: a job left in one of those states has no
    /// worker left to finish it — the previous process died — so the honest
    /// terminal state is "the restart killed it, resubmit", never "still
    /// running" forever. Returns the ids rewritten, for a startup log line.
    pub fn sweep_interrupted(&self) -> Result<Vec<String>> {
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut statement = connection.prepare(
            "SELECT id, envelope_json FROM jobs \
             WHERE status IN ('queued', 'running', 'cancelling')",
        )?;
        let rows: Vec<(String, String)> = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?)))?
            .collect::<rusqlite::Result<_>>()?;
        drop(statement);
        let timestamp = now();
        let mut swept = Vec::with_capacity(rows.len());
        for (id, envelope_json) in rows {
            let mut envelope: Value =
                serde_json::from_str(&envelope_json).unwrap_or_else(|_| json!({"id": id}));
            envelope["status"] = json!("error");
            envelope["error"] = json!(INTERRUPTED_ERROR);
            envelope["completed_at"] = json!(timestamp);
            let updated_json = serde_json::to_string(&envelope)?;
            connection.execute(
                "UPDATE jobs SET status = 'error', envelope_json = ?2, updated_at = ?3 \
                 WHERE id = ?1",
                params![id, updated_json, timestamp],
            )?;
            swept.push(id);
        }
        Ok(swept)
    }

    /// Keep only the `keep` most recently updated rows — bounds an otherwise
    /// ever-growing history in a long-lived service.
    pub fn prune(&self, keep: usize) -> Result<()> {
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        connection.execute(
            "DELETE FROM jobs WHERE id NOT IN (
                SELECT id FROM jobs ORDER BY updated_at DESC LIMIT ?1
             )",
            params![keep as i64],
        )?;
        Ok(())
    }

    /// Row count — test/diagnostic use.
    #[cfg(test)]
    pub fn row_count(&self) -> Result<usize> {
        let connection = self
            .connection
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        Ok(
            connection.query_row("SELECT COUNT(*) FROM jobs", [], |row| row.get::<_, i64>(0))?
                as usize,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_then_get_round_trips_the_envelope() {
        let store = JobsStore::open_in_memory().unwrap();
        let envelope = json!({"id":"a","status":"running","processed":3,"total":10});
        store.record("a", &envelope).unwrap();
        assert_eq!(store.get("a").unwrap(), Some(envelope));
    }

    #[test]
    fn record_overwrites_the_prior_envelope_for_the_same_id() {
        let store = JobsStore::open_in_memory().unwrap();
        store
            .record("a", &json!({"id":"a","status":"queued"}))
            .unwrap();
        store
            .record("a", &json!({"id":"a","status":"complete","files":5}))
            .unwrap();
        let value = store.get("a").unwrap().unwrap();
        assert_eq!(value["status"], "complete");
        assert_eq!(value["files"], 5);
        assert_eq!(store.row_count().unwrap(), 1);
    }

    #[test]
    fn get_of_an_unknown_id_is_none_not_an_error() {
        let store = JobsStore::open_in_memory().unwrap();
        assert_eq!(store.get("nope").unwrap(), None);
    }

    #[test]
    fn sweep_rewrites_every_non_terminal_status_to_error_and_leaves_terminal_rows_alone() {
        let store = JobsStore::open_in_memory().unwrap();
        store
            .record("queued-job", &json!({"id":"queued-job","status":"queued"}))
            .unwrap();
        store
            .record(
                "running-job",
                &json!({"id":"running-job","status":"running","processed":4,"total":9}),
            )
            .unwrap();
        store
            .record(
                "cancelling-job",
                &json!({"id":"cancelling-job","status":"cancelling"}),
            )
            .unwrap();
        store
            .record(
                "done-job",
                &json!({"id":"done-job","status":"complete","files":2}),
            )
            .unwrap();
        store
            .record(
                "errored-job",
                &json!({"id":"errored-job","status":"error","error":"boom"}),
            )
            .unwrap();

        let mut swept = store.sweep_interrupted().unwrap();
        swept.sort();
        assert_eq!(swept, vec!["cancelling-job", "queued-job", "running-job"]);

        for id in ["queued-job", "running-job", "cancelling-job"] {
            let value = store.get(id).unwrap().unwrap();
            assert_eq!(value["status"], "error", "{id}: {value}");
            assert_eq!(value["error"], INTERRUPTED_ERROR, "{id}: {value}");
            assert!(value["completed_at"].is_number(), "{id}: {value}");
        }
        // Fields the pre-sweep envelope carried (e.g. progress) survive the
        // rewrite — only status/error/completed_at change.
        let running = store.get("running-job").unwrap().unwrap();
        assert_eq!(running["processed"], 4);
        assert_eq!(running["total"], 9);

        // Already-terminal rows are untouched.
        let done = store.get("done-job").unwrap().unwrap();
        assert_eq!(done["status"], "complete");
        assert_eq!(done["files"], 2);
        let errored = store.get("errored-job").unwrap().unwrap();
        assert_eq!(errored["status"], "error");
        assert_eq!(errored["error"], "boom");
    }

    #[test]
    fn sweep_is_idempotent() {
        let store = JobsStore::open_in_memory().unwrap();
        store
            .record("a", &json!({"id":"a","status":"running"}))
            .unwrap();
        let first = store.sweep_interrupted().unwrap();
        assert_eq!(first, vec!["a"]);
        let second = store.sweep_interrupted().unwrap();
        assert!(second.is_empty(), "an already-error row is not re-swept");
    }

    #[test]
    fn prune_keeps_only_the_most_recently_updated_rows() {
        let store = JobsStore::open_in_memory().unwrap();
        for index in 0..5 {
            let id = format!("job-{index}");
            store
                .record(&id, &json!({"id": id, "status": "complete"}))
                .unwrap();
            // Force distinct `updated_at` values — the fixture writes fast
            // enough that `now()` alone could tie two rows.
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        store.prune(2).unwrap();
        assert_eq!(store.row_count().unwrap(), 2);
        assert!(store.get("job-3").unwrap().is_some());
        assert!(store.get("job-4").unwrap().is_some());
        assert!(store.get("job-0").unwrap().is_none());
    }

    #[test]
    fn opening_persists_across_reopen_of_the_same_file() {
        let temp = tempfile::tempdir().unwrap();
        {
            let store = JobsStore::open(temp.path()).unwrap();
            store
                .record("a", &json!({"id":"a","status":"running"}))
                .unwrap();
        }
        let reopened = JobsStore::open(temp.path()).unwrap();
        let value = reopened.get("a").unwrap().unwrap();
        assert_eq!(value["status"], "running");
        assert!(temp.path().join(RESERVED_OUTPUT_NAME).is_file());
    }

    #[test]
    fn a_newer_schema_version_refuses_to_open() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join(RESERVED_OUTPUT_NAME);
        {
            let connection = Connection::open(&path).unwrap();
            connection
                .pragma_update(None, "user_version", MIGRATIONS.len() as i64 + 1)
                .unwrap();
        }
        assert!(
            JobsStore::open(temp.path()).is_err(),
            "a schema version newer than this binary knows must refuse to open"
        );
    }
}
