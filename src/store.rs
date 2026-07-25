use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};

use crate::config::Config;
use crate::model::{ProcessedFile, SearchHit};
use crate::normalize::{fold, words, Normalizer};
use crate::pipeline::{row_complete, MAX_ATTEMPTS};
use crate::vision::{FaceDetection, VisionResult};

const SCHEMA: &str = r#"
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE,
  drive TEXT,
  dir TEXT,
  name TEXT,
  ext TEXT,
  size INTEGER,
  mtime REAL,
  lang TEXT,
  method TEXT,
  ocr_used INTEGER,
  pages INTEGER,
  chars INTEGER,
  sha1 TEXT,
  indexed_at REAL,
  attempts INTEGER NOT NULL DEFAULT 0,
  last_attempt_at REAL,
  elapsed_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_files_dir ON files(dir);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  name, path, content, tokens,
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);
CREATE TABLE IF NOT EXISTS chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL,
  content TEXT NOT NULL,
  embedding BLOB NOT NULL,
  dimensions INTEGER NOT NULL,
  model TEXT NOT NULL,
  UNIQUE(file_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS idx_chunks_file ON chunks(file_id);
CREATE TABLE IF NOT EXISTS vision(
  file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  width INTEGER, height INTEGER,
  phash TEXT,
  exif_json TEXT, quality_json TEXT,
  objects_json TEXT,
  tags_json TEXT,
  caption TEXT,
  embedding BLOB, embedding_model TEXT, dimensions INTEGER,
  frames INTEGER,
  elapsed_ms INTEGER, error TEXT,
  faces_model TEXT
);
CREATE INDEX IF NOT EXISTS idx_vision_phash ON vision(phash);
CREATE TABLE IF NOT EXISTS faces(
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  face_index INTEGER NOT NULL,
  x INTEGER NOT NULL, y INTEGER NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL,
  quality REAL NOT NULL,
  embedding BLOB, dimensions INTEGER,
  model TEXT NOT NULL,
  frame INTEGER,
  PRIMARY KEY(file_id, face_index)
);
CREATE INDEX IF NOT EXISTS idx_faces_file ON faces(file_id);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
"#;

/// Columns added to `files` after the corpus format's first release, in the order
/// they must be applied.
///
/// [`SCHEMA`] is all `IF NOT EXISTS`, which is why new TABLES appear on a live
/// corpus by themselves — but it never touches a `files` table that already
/// exists, so a corpus keeps whatever column set it was created with. Every
/// column added since therefore has to be re-added here, per corpus, at open.
/// Kept as data rather than a script so the check is `PRAGMA table_info` and the
/// migration is a no-op on a corpus that already has them: re-running is free,
/// and a corpus written by a NEWER build than the one opening it keeps its extra
/// columns untouched.
const ADDED_FILE_COLUMNS: &[(&str, &str)] = &[
    ("attempts", "INTEGER NOT NULL DEFAULT 0"),
    ("last_attempt_at", "REAL"),
    ("elapsed_ms", "INTEGER"),
];

/// The same story for the `vision` table: [`SCHEMA`] created it once and never
/// revisits it, so a column added later has to be re-added per corpus at open.
///
/// `faces_model` needs no backfill and must not get one. NULL is already the
/// truthful value for every pre-existing row — nothing scanned those files for
/// faces — and it is exactly what makes the first faces job pick them up.
const ADDED_VISION_COLUMNS: &[(&str, &str)] = &[("faces_model", "TEXT")];

/// How long a WRITER waits for a lock before giving up. An indexing job now
/// writes into the same file readers open, and a batch commit holds an exclusive
/// lock for the duration of one flush, so both sides must wait rather than fail.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(30);

/// Unix seconds, for the corpus meta timestamps.
fn now_unix() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64())
        .unwrap_or_default()
}

/// How long a READER waits. Deliberately far shorter than the writer's: a
/// consumer polling `/corpus/*` while a job indexes wants a prompt, honest
/// "busy, retry" rather than a stall lasting the writer's whole commit window.
/// Long enough to ride out a small flush, short enough not to look hung.
pub const READ_BUSY_TIMEOUT: Duration = Duration::from_secs(3);

/// Files written per transaction. The corpus is written in place, so everything
/// committed survives a kill and the batch size is simply how much extraction
/// and OCR work a crash throws away. 100 buys durability for one extra fsync per
/// 100 files — invisible next to per-file extraction cost — where the previous
/// 500 could discard several minutes of OCR.
/// The DEFAULT batch; `Config::commit_batch` overrides it per deployment, and
/// `default_commit_batch()` must equal this so an unset config is unchanged.
pub const COMMIT_FILES: usize = 100;

/// Ceiling on how long work can sit uncommitted when files are slow. Exhaustive
/// OCR of a large PDF runs into minutes per file, so the count alone would leave
/// long unprotected windows on exactly the runs that cost the most to redo.
const COMMIT_INTERVAL: Duration = Duration::from_secs(30);

/// The corpus database addressed by `out`. Service jobs name the published
/// `<name>.sqlite` file directly — writes land in the file consumers read, which
/// is what makes an interrupted job resumable — while the CLI names an output
/// directory that also holds `manifest.jsonl`, `catalog.csv` and reports.
pub fn database_path(out: &Path) -> PathBuf {
    if out.extension().is_some_and(|ext| ext == "sqlite") {
        out.to_path_buf()
    } else {
        out.join("index.sqlite")
    }
}

/// The rollback journal SQLite keeps beside `database` while a transaction is
/// open. Both the overwrite path (which must not leave one behind for a fresh
/// database to adopt as a hot journal) and the read path (which recovers one)
/// need this name, and neither may build it by formatting a lossy `display()`.
pub fn journal_path(database: &Path) -> PathBuf {
    let mut name = database.as_os_str().to_os_string();
    name.push("-journal");
    PathBuf::from(name)
}

/// Delete the database `out` addresses, along with any rollback journal left by
/// a killed writer. Callers must have exhausted everything else that can fail
/// first: the corpus is written in place, so this is the point of no return.
pub fn remove_database(out: &Path) -> Result<()> {
    let database = database_path(out);
    if !database.exists() {
        return Ok(());
    }
    fs::remove_file(&database).with_context(|| format!("replacing {}", database.display()))?;
    // Best effort: an orphaned journal with no database is inert, and SQLite
    // discards one whose header does not match the database it opens.
    let _ = fs::remove_file(journal_path(&database));
    Ok(())
}

/// Bring an existing `files` table up to the current column set.
///
/// Runs on every open, straight after [`SCHEMA`], and is a no-op once the columns
/// are there — `ALTER TABLE ADD COLUMN` is a schema-only edit in SQLite, so on a
/// corpus with hundreds of thousands of rows the columns themselves cost nothing
/// and existing rows simply read back the declared default.
///
/// The one-off cost is the backfill, and it runs EXACTLY when the `attempts`
/// column is first created — the column's own absence is the "this corpus has
/// never been counted" evidence, so no separate marker can drift from it. All of
/// it lands in one transaction: an ALTER that committed without its backfill
/// would look counted while claiming every unfinished row had never been tried,
/// and the whole corpus would be re-attempted [`MAX_ATTEMPTS`] more times before
/// converging.
///
/// See [`attempts_backfill`] for what the stamped value means.
fn migrate_files_table(connection: &Connection) -> Result<()> {
    let present = existing_columns(connection, "files")?;
    let missing = ADDED_FILE_COLUMNS
        .iter()
        .filter(|(name, _)| !present.contains(*name))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let mut script = String::from("BEGIN IMMEDIATE;\n");
    for (name, declaration) in &missing {
        script.push_str(&format!(
            "ALTER TABLE files ADD COLUMN {name} {declaration};\n"
        ));
    }
    if missing.iter().any(|(name, _)| *name == "attempts") {
        script.push_str(&attempts_backfill());
    }
    script.push_str("COMMIT;");
    connection
        .execute_batch(&script)
        .context("migrating the files table")?;
    Ok(())
}

/// The column names an existing table already has, by `PRAGMA table_info`. The
/// one source both migrations ask, so neither can drift into believing a column
/// is there because a `CREATE TABLE IF NOT EXISTS` mentions it.
fn existing_columns(connection: &Connection, table: &str) -> Result<HashSet<String>> {
    let mut statement = connection.prepare(&format!("PRAGMA table_info({table})"))?;
    let present = statement
        .query_map([], |row| row.get::<_, String>(1))?
        .flatten()
        .collect::<HashSet<_>>();
    Ok(present)
}

/// Bring an existing `vision` table up to the current column set — the same
/// no-op-when-current, additive `ALTER TABLE ADD COLUMN` shape as
/// [`migrate_files_table`], with no backfill to do (see
/// [`ADDED_VISION_COLUMNS`]).
///
/// A corpus that predates the vision table has no `vision` table to migrate;
/// [`SCHEMA`] has just created it with the current columns, so `present` already
/// contains them and this returns immediately.
fn migrate_vision_table(connection: &Connection) -> Result<()> {
    let present = existing_columns(connection, "vision")?;
    if present.is_empty() {
        return Ok(());
    }
    let missing = ADDED_VISION_COLUMNS
        .iter()
        .filter(|(name, _)| !present.contains(*name))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        return Ok(());
    }
    let mut script = String::from("BEGIN IMMEDIATE;\n");
    for (name, declaration) in &missing {
        script.push_str(&format!(
            "ALTER TABLE vision ADD COLUMN {name} {declaration};\n"
        ));
    }
    script.push_str("COMMIT;");
    connection
        .execute_batch(&script)
        .context("migrating the vision table")?;
    Ok(())
}

/// What a corpus that predates the attempt counter is stamped with.
///
/// Every row it matches is one the resume predicate would re-process — the
/// unfinished ones — and every one of them is the durable record of at least one
/// completed attempt, timestamped by `indexed_at`. On the live corpora these rows
/// have been re-attempted on every resume for as long as the corpus has existed;
/// starting them at zero would assert the opposite and mandate
/// [`MAX_ATTEMPTS`] more full passes over exactly the files that have never once
/// succeeded. They are stamped as spent instead, so the first resume after this
/// migration converges rather than re-burning them.
///
/// Nothing is closed off by that. A row re-opens when its file changes (size or
/// mtime), when an upgrade makes a better extraction available (exhaustive OCR, a
/// higher vision tier), when the embedding model changes, when
/// [`crate::extract::extractor_revision`] moves because the build learned a new
/// format, or when a run is submitted with `retry_errors`.
///
/// `excluded:` rows are left alone: they are terminal by decision, not by
/// exhaustion, and stamping them as burned attempts would misreport why they were
/// never processed.
fn attempts_backfill() -> String {
    format!(
        "UPDATE files \
            SET attempts = {MAX_ATTEMPTS}, last_attempt_at = indexed_at \
          WHERE method NOT LIKE 'excluded:%' \
            AND (method = 'name-only' \
                 OR method LIKE 'error:%' \
                 OR method LIKE '%-partial' \
                 OR NOT EXISTS (SELECT 1 FROM chunks c WHERE c.file_id = files.id));"
    )
}

/// One stored row as resume sees it: everything the skip predicate compares,
/// nothing else. Returned by [`IndexStore::existing_keys`] keyed by path.
#[derive(Debug, Clone)]
pub struct ExistingRow {
    pub size: u64,
    /// Truncated to whole seconds (`mtime as i64`), the comparison every caller
    /// makes; the column itself is a float.
    pub mtime: i64,
    pub method: String,
    pub has_chunks: bool,
    /// Attempts that have ended without finishing this row. Reset to 0 by a
    /// finished outcome and by a change to the file's bytes, so it counts
    /// CONSECUTIVE failures on the file as it stands rather than lifetime ones.
    pub attempts: u32,
}

pub struct IndexStore {
    out: PathBuf,
    connection: Connection,
    sidecar: String,
    jsonl: Option<BufWriter<File>>,
    catalog: Option<csv::Writer<File>>,
    pending: usize,
    committed: Instant,
    /// Files per batched commit — the throughput/durability lever. A larger
    /// batch amortises the commit's fsync over more work (faster), at the cost of
    /// re-doing more files if the job is killed mid-batch (which resume handles).
    /// Sourced from config so an operator can raise it; `COMMIT_FILES` is the
    /// default.
    commit_batch: usize,
    /// Set when a per-file rollback itself failed, leaving the open transaction
    /// in an unknown state. `finish` then discards it instead of committing.
    poisoned: bool,
    /// The corpus' `vec0` shadow index, when it has one — `None` for every
    /// corpus that has never been through `llm-index vector-index`, which is
    /// every corpus by default and the reason nothing here costs anything
    /// unless an operator asked for it.
    ///
    /// Held in memory because it is a pair of counts that has to move with the
    /// rows: [`crate::vec0::IndexState::vectors`] and
    /// [`crate::vec0::IndexState::chunks`] are what let a reader prove the
    /// index still covers the corpus, so they are re-stamped inside the same
    /// per-file savepoint that writes the chunks themselves.
    vec0: Option<crate::vec0::IndexState>,
}

impl IndexStore {
    pub fn open(out: &Path, config: &Config, resume: bool, artifacts: bool) -> Result<Self> {
        let database = database_path(out);
        // Artifacts and sidecars live beside the database whichever way `out`
        // addressed it.
        let root = database.parent().unwrap_or(Path::new(".")).to_path_buf();
        fs::create_dir_all(&root)?;
        // Must precede the open: a connection only picks up the `vec0` module
        // from SQLite's auto-extension list as it is created, so a writer opened
        // first could not maintain a corpus that has a shadow index — it would
        // fail on `no such module: vec0` mid-job. Inert for the corpora that
        // have none, which is all of them by default.
        crate::vec0::register();
        let connection = Connection::open(&database)?;
        // Journal mode is left at the rollback-journal default on purpose: the
        // corpus is copied and served as a bare single file, and WAL would add
        // `-wal`/`-shm` sidecars that a copy silently leaves behind. The
        // rollback journal is transient and gone after every commit.
        //
        // Readers (the /corpus routes, consumer apps) now share the file with a
        // live writer, so both sides need to wait out the other's lock instead
        // of failing: without this a reader's shared lock can abort the writer's
        // COMMIT and fail the whole job.
        connection.busy_timeout(BUSY_TIMEOUT)?;
        // Write durability. The default stays FULL — safe against a power loss
        // mid-commit. An operator can opt into NORMAL (`sync_normal: true`),
        // which skips some fsyncs for throughput; in rollback-journal mode that
        // carries a small database-corruption risk on a power loss / hard reset,
        // acceptable only because the corpus is regenerable and resumable. Left
        // at the default here means every existing deployment is byte-unchanged.
        if config.sync_normal {
            connection.pragma_update(None, "synchronous", "NORMAL")?;
        }
        connection
            .execute_batch(SCHEMA)
            .context("creating SQLite FTS5 schema")?;
        // Only ever a no-op for a corpus this build created; the live ones were
        // created by builds whose `files` table stops at `indexed_at`.
        migrate_files_table(&connection)?;
        // Likewise for `vision`, whose column set grew after the tiers shipped.
        migrate_vision_table(&connection)?;
        // Self-description (previously the corpus was anonymous — after a
        // restart nothing recorded what produced it): created_at once, plus
        // per-job values the pipeline stamps via `set_meta`. `IF NOT EXISTS`
        // in the schema means old corpora gain the table on their next open.
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('created_at', ?1) \
             ON CONFLICT(key) DO NOTHING",
            [format!("{:.0}", now_unix())],
        )?;
        let jsonl = if artifacts {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(resume)
                .truncate(!resume)
                .open(root.join("manifest.jsonl"))?;
            Some(BufWriter::new(file))
        } else {
            None
        };
        let catalog_path = root.join("catalog.csv");
        let append = artifacts
            && resume
            && catalog_path
                .metadata()
                .map(|m| m.len() > 0)
                .unwrap_or(false);
        let catalog = if artifacts {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(append)
                .truncate(!append)
                .open(&catalog_path)?;
            let mut writer = csv::WriterBuilder::new()
                .has_headers(false)
                .from_writer(file);
            if !append {
                writer.write_record([
                    "path", "name", "ext", "size", "mtime", "lang", "method", "ocr_used", "chars",
                ])?;
            }
            Some(writer)
        } else {
            None
        };
        // Read BEFORE the write transaction opens, like the rest of open's
        // inspection. A shadow index whose table exists but whose build never
        // finished has no state to load, and is left exactly as it is: this job
        // will not maintain a table it cannot vouch for, and the build the
        // operator re-runs replaces it wholesale.
        let vec0 = if crate::vec0::present(&connection)? {
            crate::vec0::state(&connection)?
        } else {
            None
        };
        connection.execute_batch("BEGIN IMMEDIATE")?;
        Ok(Self {
            out: root,
            connection,
            sidecar: config.sidecar.clone(),
            jsonl,
            catalog,
            pending: 0,
            committed: Instant::now(),
            commit_batch: config.commit_batch.max(1),
            poisoned: false,
            vec0,
        })
    }

    pub fn existing_keys(&self) -> Result<HashMap<String, ExistingRow>> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT f.path,f.size,f.mtime,f.method,EXISTS(SELECT 1 FROM chunks c WHERE c.file_id=f.id),\
                 f.attempts \
                 FROM files f",
            )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                ExistingRow {
                    size: row.get::<_, i64>(1)? as u64,
                    mtime: row.get::<_, f64>(2)? as i64,
                    method: row.get::<_, String>(3)?,
                    has_chunks: row.get::<_, i64>(4)? != 0,
                    attempts: row.get::<_, i64>(5)?.clamp(0, i64::from(u32::MAX)) as u32,
                },
            ))
        })?;
        Ok(rows.flatten().collect())
    }

    /// The highest vision tier recorded per file path, for the resume
    /// change-detection upgrade rule. Absent files simply aren't in the map.
    pub fn existing_vision_modes(&self) -> Result<HashMap<String, String>> {
        let mut statement = self
            .connection
            .prepare("SELECT f.path, v.mode FROM vision v JOIN files f ON f.id = v.file_id")?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        Ok(rows.flatten().collect())
    }

    /// The face model recorded per file path, for the faces change-detection
    /// rule. A path present with `None` was scanned by a build that wrote no
    /// model id; a path ABSENT from the map has no vision row at all. Both mean
    /// "not scanned by the pair this job runs", which is what the rule needs.
    ///
    /// A sibling of [`existing_vision_modes`](Self::existing_vision_modes)
    /// rather than a widening of it: the pipeline only asks when a job has faces
    /// enabled AND staged, so a corpus that never uses the feature never pays
    /// for the scan.
    pub fn existing_face_models(&self) -> Result<HashMap<String, Option<String>>> {
        let mut statement = self.connection.prepare(
            "SELECT f.path, v.faces_model FROM vision v JOIN files f ON f.id = v.file_id",
        )?;
        let rows = statement.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
        })?;
        Ok(rows.flatten().collect())
    }

    /// Delete rows for files that have disappeared — but ONLY under the job's
    /// own roots. A row outside every walked root was never visible to this
    /// job's walk, so its absence from `current` says nothing about the file:
    /// pruning it would let a sub-path resume silently destroy the rest of a
    /// whole-drive corpus (the walk of `I:\Docs` does not contain `I:\Photos\a`,
    /// and before this scoping that absence deleted it).
    ///
    /// `roots` are the walker's canonical root strings ([`crate::walker::canonical_root`]),
    /// matched against row paths by exact-prefix-plus-separator (or equality),
    /// the same string forms the walker wrote.
    pub fn prune_missing(&mut self, roots: &[String], current: &HashSet<String>) -> Result<usize> {
        let under_a_root = |path: &str| {
            roots.iter().any(|root| {
                let trimmed = root.trim_end_matches(std::path::MAIN_SEPARATOR);
                path == trimmed
                    || (path.starts_with(trimmed)
                        && path[trimmed.len()..].starts_with(std::path::MAIN_SEPARATOR))
            })
        };
        let mut statement = self.connection.prepare("SELECT id,path FROM files")?;
        let stale = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .flatten()
            .filter(|(_, path)| under_a_root(path) && !current.contains(path))
            .collect::<Vec<_>>();
        drop(statement);
        for (id, _) in &stale {
            self.unindex_chunks_of(*id)?;
            self.connection
                .execute("DELETE FROM chunks WHERE file_id=?1", [id])?;
            self.connection
                .execute("DELETE FROM vision WHERE file_id=?1", [id])?;
            self.connection
                .execute("DELETE FROM faces WHERE file_id=?1", [id])?;
            self.connection
                .execute("DELETE FROM fts WHERE rowid=?1", [id])?;
            self.connection
                .execute("DELETE FROM files WHERE id=?1", [id])?;
        }
        self.stamp_vec0()?;
        Ok(stale.len())
    }

    /// Upsert one self-description key (job metadata, model identity,
    /// lifecycle timestamps). Rides the store's open transaction, so meta
    /// changes commit atomically with the file batches around them.
    pub fn set_meta(&mut self, key: &str, value: &str) -> Result<()> {
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?1,?2) \
             ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            rusqlite::params![key, value],
        )?;
        Ok(())
    }

    pub fn get_meta(&self, key: &str) -> Result<Option<String>> {
        use rusqlite::OptionalExtension;
        Ok(self
            .connection
            .query_row("SELECT value FROM meta WHERE key=?1", [key], |row| {
                row.get(0)
            })
            .optional()?)
    }

    pub fn add(&mut self, file: &ProcessedFile, indexed_at: f64) -> Result<()> {
        // Every row a file needs goes in under its own savepoint. Without one, a
        // failure part-way — a chunk insert that fails once the files and fts
        // rows are already in — leaves that debris in the open transaction, and
        // `finish` commits it: a published file whose vectors are incomplete.
        // Resume treats any file holding at least one chunk as done, so that row
        // would never be repaired. Rolling back leaves the file absent instead,
        // which is precisely what makes resume redo it.
        self.connection.execute_batch("SAVEPOINT file")?;
        // The shadow index' counts live in memory as well as in `meta`, and only
        // the `meta` half is inside the savepoint. Snapshot them so a rolled-back
        // file leaves both halves agreeing — a stranded in-memory count would be
        // stamped onto the NEXT file and make the index look stale for good.
        let vec0 = self.vec0.clone();
        if let Err(error) = self.write_rows(file, indexed_at) {
            self.vec0 = vec0;
            if let Err(rollback) = self
                .connection
                .execute_batch("ROLLBACK TO file; RELEASE file")
            {
                // The transaction's contents are now unknown, so nothing in it
                // may be published; batches committed earlier are untouched.
                self.poisoned = true;
                return Err(error.context(format!("rolling back partial file: {rollback}")));
            }
            return Err(error);
        }
        self.connection.execute_batch("RELEASE file")?;
        // Artifacts are written only once the file's rows are safely in. They
        // are derived views of the database rather than part of it, so a failure
        // here stops the run without invalidating what was stored.
        self.write_artifacts(file)?;
        self.pending += 1;
        if self.pending >= self.commit_batch || self.committed.elapsed() >= COMMIT_INTERVAL {
            self.commit()?;
        }
        Ok(())
    }

    /// Count an attempt that produced nothing to store.
    ///
    /// The keep-on-failure path drops its result entirely to preserve a still
    /// valid stored row, so without this the work it burned leaves no trace at
    /// all: the row's `indexed_at` still points at the successful run that wrote
    /// it, and a file whose upgrade fails on every resume looks, from the corpus,
    /// like a file nobody has touched since it succeeded. Only the attempt
    /// columns move; the row's content and `indexed_at` describe what is stored
    /// and must keep describing it.
    ///
    /// Rides the store's open transaction like every other write here.
    pub fn record_failed_attempt(&mut self, path: &str, elapsed_ms: u64, at: f64) -> Result<()> {
        self.connection.execute(
            "UPDATE files SET attempts=attempts+1,last_attempt_at=?2,elapsed_ms=?3 WHERE path=?1",
            params![path, at, elapsed_ms as i64],
        )?;
        Ok(())
    }

    /// Mirror one just-written `chunks` row into the shadow index.
    ///
    /// A no-op — not a branch taken cheaply, but no work at all — for a corpus
    /// without an index, which is every corpus that has not been through
    /// `llm-index vector-index`. That is what makes the index optional at index
    /// time as well as at query time: a job on an unindexed corpus writes
    /// exactly the rows it wrote before this existed.
    ///
    /// A chunk the index does not cover (another embedding model, another
    /// width) is counted and not inserted, which is the same rule
    /// [`crate::vec0::build`] applies. Both counters move here so the state a
    /// reader validates against is written by the code that writes the rows.
    fn index_chunk(&mut self, id: i64, embedding: &[u8]) -> Result<()> {
        let Some(state) = self.vec0.as_mut() else {
            return Ok(());
        };
        state.chunks += 1;
        if state.model != crate::embedding::EMBEDDING_MODEL
            || embedding.len() != state.dimensions * 4
        {
            return Ok(());
        }
        state.vectors += 1;
        crate::vec0::insert(&self.connection, id, embedding)
    }

    /// Drop a file's chunks out of the shadow index, ahead of the `DELETE` that
    /// removes the rows themselves.
    ///
    /// Ahead, not after: the ids and the widths this needs live in the rows
    /// being deleted, so reading them afterwards would read nothing and leave
    /// the index holding vectors for chunks that no longer exist — which a
    /// later k-NN would return as candidates and the re-score would silently
    /// drop, quietly shrinking every result page near a re-indexed file.
    fn unindex_chunks_of(&mut self, file_id: i64) -> Result<()> {
        let Some(state) = self.vec0.as_ref() else {
            return Ok(());
        };
        let width = state.dimensions * 4;
        let model = state.model.clone();
        let mut statement = self
            .connection
            .prepare("SELECT id,model,LENGTH(embedding) FROM chunks WHERE file_id=?1")?;
        let doomed = statement
            .query_map([file_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>, _>>()?;
        drop(statement);
        for (id, row_model, length) in doomed {
            let indexed = row_model == model && length as usize == width;
            if indexed {
                crate::vec0::delete(&self.connection, id)?;
            }
            let state = self.vec0.as_mut().expect("checked above");
            state.chunks = state.chunks.saturating_sub(1);
            if indexed {
                state.vectors = state.vectors.saturating_sub(1);
            }
        }
        Ok(())
    }

    /// Re-stamp the shadow index' `meta` record from the in-memory counters.
    ///
    /// Called at the end of every unit of work that moved them, INSIDE that
    /// unit's savepoint/transaction, so the counts a reader validates against
    /// are published by exactly the commit that published the rows they
    /// describe. A corpus without an index has nothing to stamp.
    ///
    /// The job stamp is re-read here rather than captured at open: the pipeline
    /// writes `meta.last_job_started_at` AFTER opening the store, so a value
    /// captured at open would be the PREVIOUS job's and leave this job's own
    /// work looking like a foreign build's. Reading it back is one indexed row
    /// lookup against the extraction cost of the file that triggered it.
    fn stamp_vec0(&mut self) -> Result<()> {
        // Checked before the lookup, not after: a corpus with no index must not
        // pay a `meta` query per file for a stamp it will never write.
        if self.vec0.is_none() {
            return Ok(());
        }
        let job = crate::vec0::job_stamp(&self.connection);
        let state = self.vec0.as_mut().expect("checked above");
        state.job = job;
        crate::vec0::write_state(&self.connection, state)
    }

    /// Every database row one file contributes, run inside the caller's
    /// savepoint so the set lands whole or not at all.
    fn write_rows(&mut self, file: &ProcessedFile, indexed_at: f64) -> Result<()> {
        // The file row is INSERT OR REPLACE'd, which mints a new rowid on the
        // UNIQUE(path) conflict, so the previous id's chunks/fts are deleted
        // here and its vision row is reconciled after the re-insert.
        // UNCONDITIONAL (previously resume-only): any path re-add through any
        // flow — including a non-resume run pointed at an existing corpus —
        // would otherwise strand the old rowid's fts text in the index, and
        // stale FTS rows surface as ghost search hits that nothing ever
        // cleans up.
        let old = self
            .connection
            .query_row(
                "SELECT id,size,mtime,attempts FROM files WHERE path=?1",
                [&file.rec.path],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, f64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .ok();
        let old_id = old.map(|(id, _, _, _)| id);
        // Byte-for-byte identical to the stored row, by the same truncated-mtime
        // comparison resume uses. Needed twice below: the attempt counter and the
        // vision carry-forward both key on it.
        let unchanged = old.is_some_and(|(_, size, mtime, _)| {
            size == file.rec.size as i64 && mtime as i64 == file.rec.mtime as i64
        });
        // A CHANGED file is a different file, whatever the path says, so it opens
        // with a full budget rather than inheriting the failures of the bytes it
        // replaced.
        let attempts = if row_complete(&file.method, !file.chunks.is_empty()) {
            0
        } else if unchanged {
            old.map_or(0, |(_, _, _, attempts)| attempts)
                .saturating_add(1)
        } else {
            1
        };
        // Capture the old vision row BEFORE the INSERT OR REPLACE below: the
        // bundled SQLite runs with foreign_keys ON, so replacing the files row
        // cascade-deletes its vision row. A carry-forward therefore has to
        // re-insert the captured row under the new rowid, not re-point the old
        // one (which no longer exists). Only needed when this job produced no
        // vision result of its own.
        let carried_vision: Option<Vec<rusqlite::types::Value>> = match old_id {
            Some(old_id) if file.vision.is_none() => self
                .connection
                .query_row(
                    "SELECT mode,width,height,phash,exif_json,quality_json,objects_json,\
                     tags_json,caption,embedding,embedding_model,dimensions,frames,elapsed_ms,\
                     error,faces_model FROM vision WHERE file_id=?1",
                    [old_id],
                    |row| {
                        (0..16)
                            .map(|i| row.get::<_, rusqlite::types::Value>(i))
                            .collect()
                    },
                )
                .optional()?,
            _ => None,
        };
        // The same capture for the face rows, on its own condition. Faces has to
        // be asked separately because it is a sub-tier, not a tier: a job can run
        // vision with faces OFF over a file whose faces were recorded by an
        // earlier job, and the vision carry-forward above would not fire (this
        // job DID produce a vision result). Dropping the rows then would make
        // turning faces off destructive, which the rest of vision never is.
        let carried_faces: Option<Vec<Vec<rusqlite::types::Value>>> = match old_id {
            Some(old_id)
                if file
                    .vision
                    .as_ref()
                    .is_none_or(|result| result.faces_model.is_none()) =>
            {
                Some(self.stored_faces(old_id)?)
            }
            _ => None,
        };
        if let Some(old_id) = old_id {
            self.unindex_chunks_of(old_id)?;
            self.connection
                .execute("DELETE FROM chunks WHERE file_id=?1", [old_id])?;
            self.connection
                .execute("DELETE FROM fts WHERE rowid=?1", [old_id])?;
        }
        self.connection.execute(
            "INSERT OR REPLACE INTO files(path,drive,dir,name,ext,size,mtime,lang,method,ocr_used,pages,chars,sha1,indexed_at,attempts,last_attempt_at,elapsed_ms) \
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
            params![file.rec.path, file.rec.drive, file.rec.dir, file.rec.name, file.rec.ext,
                file.rec.size as i64, file.rec.mtime, file.lang, file.method, file.ocr_used as i64,
                file.pages as i64, file.content.chars().count() as i64, file.sha1, indexed_at,
                attempts, indexed_at, file.elapsed_ms as i64])?;
        let id = self.connection.last_insert_rowid();
        self.connection.execute(
            "INSERT INTO fts(rowid,name,path,content,tokens) VALUES(?1,?2,?3,?4,?5)",
            params![
                id,
                file.rec.name,
                file.rec.path,
                file.content,
                file.tokens.join(" ")
            ],
        )?;
        for chunk in &file.chunks {
            let embedding = crate::embedding::vector_to_bytes(&chunk.vector);
            self.connection.execute(
                "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                 VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    id,
                    chunk.index as i64,
                    chunk.content,
                    embedding,
                    chunk.vector.len() as i64,
                    crate::embedding::EMBEDDING_MODEL,
                ],
            )?;
            self.index_chunk(self.connection.last_insert_rowid(), &embedding)?;
        }
        // Vision reconciliation across the rowid change on resume. The REPLACE
        // above cascade-dropped any old vision row (foreign_keys ON):
        //  - this job produced a result -> write it under the new rowid;
        //  - it did not, and the bytes are UNCHANGED -> carry the captured row
        //    forward (spec: a lower/off tier must never drop vision);
        //  - it did not, and the bytes CHANGED -> leave it dropped, since the
        //    old phash/tags/embedding would now lie about the new content.
        match (&file.vision, old_id) {
            (Some(result), _) => {
                self.upsert_vision(id, result)?;
                if result.faces_model.is_some() {
                    self.upsert_faces(id, &result.faces)?;
                }
            }
            (None, Some(old_id)) => {
                // Belt-and-braces: on a foreign_keys=OFF build the old row would
                // survive under the stale id, so clear it before re-attaching.
                self.connection
                    .execute("DELETE FROM vision WHERE file_id=?1", [old_id])?;
                if let (true, Some(values)) = (unchanged, carried_vision) {
                    let mut row: Vec<rusqlite::types::Value> = Vec::with_capacity(17);
                    row.push(rusqlite::types::Value::Integer(id));
                    row.extend(values);
                    self.connection.execute(
                        "INSERT OR REPLACE INTO vision(file_id,mode,width,height,phash,exif_json,\
                         quality_json,objects_json,tags_json,caption,embedding,embedding_model,\
                         dimensions,frames,elapsed_ms,error,faces_model) \
                         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
                        rusqlite::params_from_iter(row),
                    )?;
                }
            }
            (None, None) => {}
        }
        // Face rows follow the same three-way rule as the vision row, evaluated
        // on its own capture: scanned this run -> already written above;
        // not scanned and the bytes are UNCHANGED -> carry the old rows forward
        // (turning faces off, or running a plain OCR pass, must not erase them);
        // not scanned and the bytes CHANGED -> leave them dropped, since boxes
        // and vectors describing the previous content would now be a claim about
        // a person that the file no longer supports.
        if let Some(carried) = &carried_faces {
            if let Some(old_id) = old_id {
                // Belt-and-braces on a foreign_keys=OFF build, exactly as above:
                // clear the stale id before deciding whether to re-attach. Only
                // reachable when this run wrote no faces of its own, so it can
                // never delete what was just written — including the case where
                // SQLite hands the replaced row's freed rowid straight back.
                self.connection
                    .execute("DELETE FROM faces WHERE file_id=?1", [old_id])?;
            }
            if unchanged {
                self.restore_faces(id, carried)?;
            }
        }
        self.stamp_vec0()
    }

    /// The manifest/catalog/sidecar views of one stored file.
    fn write_artifacts(&mut self, file: &ProcessedFile) -> Result<()> {
        if let Some(jsonl) = &mut self.jsonl {
            serde_json::to_writer(
                &mut *jsonl,
                &json!({
                    "path": file.rec.path, "name": file.rec.name, "ext": file.rec.ext,
                    "dir": file.rec.dir, "drive": file.rec.drive, "size": file.rec.size,
                    "mtime": file.rec.mtime, "lang": file.lang, "method": file.method,
                    "ocr_used": file.ocr_used, "pages": file.pages,
                    "chars": file.content.chars().count(),
                    "snippet": file.content.chars().take(400).collect::<String>(),
                }),
            )?;
            jsonl.write_all(b"\n")?;
        }
        if let Some(catalog) = &mut self.catalog {
            catalog.write_record([
                file.rec.path.as_str(),
                file.rec.name.as_str(),
                file.rec.ext.as_str(),
                &file.rec.size.to_string(),
                &format!("{:.0}", file.rec.mtime),
                file.lang.as_str(),
                file.method.as_str(),
                if file.ocr_used { "1" } else { "0" },
                &file.content.chars().count().to_string(),
            ])?;
        }
        // `excluded:` rows carry the name+dir fallback as content, same as
        // `name-only`, and nothing was extracted from the file itself — a sidecar
        // for one would be a `.txt` restating the filename next to every object
        // file on the drive.
        if self.sidecar != "none"
            && !file.content.trim().is_empty()
            && !matches!(file.method.as_str(), "text" | "name-only")
            && !file.method.starts_with("error:")
            && !file.method.starts_with("excluded:")
        {
            self.write_sidecar(file);
        }
        Ok(())
    }

    /// Publish everything written so far and open the next transaction. Each
    /// commit is a durability checkpoint: a crash after it keeps the work, a
    /// crash before it loses only this batch.
    fn commit(&mut self) -> Result<()> {
        self.connection.execute_batch("COMMIT; BEGIN IMMEDIATE")?;
        self.pending = 0;
        self.committed = Instant::now();
        if let Some(writer) = &mut self.jsonl {
            writer.flush()?
        }
        if let Some(writer) = &mut self.catalog {
            writer.flush()?
        }
        Ok(())
    }

    /// Publish the open transaction. Called on every exit path — success,
    /// cancellation and mid-run failure alike — because a partial corpus is the
    /// point: it is what resume continues from. Every file in the transaction
    /// is whole, since `add` rolls back any it could not write completely.
    pub fn finish(mut self) -> Result<()> {
        if self.poisoned {
            // A per-file rollback failed earlier, so what is in this transaction
            // is no longer known. Discard it rather than publish rows that
            // cannot be vouched for; earlier batches are already committed.
            self.connection.execute_batch("ROLLBACK")?;
            anyhow::bail!("transaction discarded after a failed per-file rollback")
        }
        // Unconditional, not only when files were written: this job stamped
        // `meta.last_job_started_at` at its start, which is exactly the witness
        // a reader compares the shadow index against. A run that wrote nothing
        // would otherwise leave its own index looking stale — a maintained
        // corpus must not lose its fast path to a no-op job.
        self.stamp_vec0()?;
        self.connection.execute_batch("COMMIT")?;
        if let Some(writer) = &mut self.jsonl {
            writer.flush()?
        }
        if let Some(writer) = &mut self.catalog {
            writer.flush()?
        }
        Ok(())
    }

    /// Insert or replace the `vision` row for `file_id`. Keyed on the file's
    /// primary key, so a re-analysis overwrites cleanly.
    pub fn upsert_vision(&self, file_id: i64, vision: &VisionResult) -> Result<()> {
        let exif_json = vision
            .exif
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let quality_json = vision
            .quality
            .as_ref()
            .map(serde_json::to_string)
            .transpose()?;
        let objects_json = (!vision.objects.is_empty())
            .then(|| serde_json::to_string(&vision.objects))
            .transpose()?;
        let tags_json = (!vision.tags.is_empty())
            .then(|| serde_json::to_string(&vision.tags))
            .transpose()?;
        let embedding = vision
            .embedding
            .as_ref()
            .map(|vector| crate::embedding::vector_to_bytes(vector));
        self.connection.execute(
            "INSERT OR REPLACE INTO vision(file_id,mode,width,height,phash,exif_json,quality_json,\
             objects_json,tags_json,caption,embedding,embedding_model,dimensions,frames,elapsed_ms,\
             error,faces_model) \
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17)",
            params![
                file_id,
                vision.mode.as_str(),
                vision.width.map(i64::from),
                vision.height.map(i64::from),
                vision.phash,
                exif_json,
                quality_json,
                objects_json,
                tags_json,
                vision.caption,
                embedding,
                vision.embedding_model,
                vision.dimensions.map(|value| value as i64),
                vision.frames.map(|value| value as i64),
                vision.elapsed_ms.map(|value| value as i64),
                vision.error,
                vision.faces_model,
            ],
        )?;
        Ok(())
    }

    /// Replace the `faces` rows for `file_id` with `faces`.
    ///
    /// `face_index` is the position in the detector's deterministic best-first
    /// order, so re-running the same file against the same models rewrites the
    /// same rows. The old rows are deleted first rather than upserted over: a
    /// re-analysis that finds FEWER faces must not leave the tail of the
    /// previous one behind, still attributed to a file that no longer shows
    /// those people.
    pub fn upsert_faces(&self, file_id: i64, faces: &[FaceDetection]) -> Result<()> {
        self.connection
            .execute("DELETE FROM faces WHERE file_id=?1", [file_id])?;
        for (index, face) in faces.iter().enumerate() {
            let embedding = face
                .embedding
                .as_ref()
                .map(|vector| crate::embedding::vector_to_bytes(vector));
            self.connection.execute(
                "INSERT INTO faces(file_id,face_index,x,y,width,height,quality,embedding,\
                 dimensions,model,frame) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                params![
                    file_id,
                    index as i64,
                    face.x,
                    face.y,
                    face.width,
                    face.height,
                    face.quality,
                    embedding,
                    face.embedding.as_ref().map(|vector| vector.len() as i64),
                    crate::vision::faces::FACE_MODEL_ID,
                    face.frame,
                ],
            )?;
        }
        Ok(())
    }

    /// The face rows recorded for `file_id`, in stored order — the capture half
    /// of the carry-forward that has to survive the rowid change on a re-add.
    fn stored_faces(&self, file_id: i64) -> Result<Vec<Vec<rusqlite::types::Value>>> {
        let mut statement = self.connection.prepare(
            "SELECT face_index,x,y,width,height,quality,embedding,dimensions,model,frame \
             FROM faces WHERE file_id=?1 ORDER BY face_index",
        )?;
        let rows = statement.query_map([file_id], |row| {
            (0..10)
                .map(|index| row.get::<_, rusqlite::types::Value>(index))
                .collect::<rusqlite::Result<Vec<_>>>()
        })?;
        Ok(rows.flatten().collect())
    }

    /// Re-attach captured face rows under a new `file_id`.
    fn restore_faces(&self, file_id: i64, faces: &[Vec<rusqlite::types::Value>]) -> Result<()> {
        for face in faces {
            let mut row: Vec<rusqlite::types::Value> = Vec::with_capacity(11);
            row.push(rusqlite::types::Value::Integer(file_id));
            row.extend(face.iter().cloned());
            self.connection.execute(
                "INSERT OR REPLACE INTO faces(file_id,face_index,x,y,width,height,quality,\
                 embedding,dimensions,model,frame) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11)",
                rusqlite::params_from_iter(row),
            )?;
        }
        Ok(())
    }

    fn write_sidecar(&self, file: &ProcessedFile) {
        let target = if self.sidecar == "inplace" {
            PathBuf::from(format!("{}.txt", file.rec.path))
        } else {
            let relative = file.rec.path.trim_start_matches(['/', '\\']);
            self.out
                .join("sidecar")
                .join(file.rec.drive.replace([':', '/', '\\'], "_"))
                .join(format!("{relative}.txt"))
        };
        if let Some(parent) = target.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let _ = fs::write(target, &file.content);
    }
}

pub fn connect(index: &Path) -> Result<Connection> {
    let path = if index.is_dir() {
        index.join("index.sqlite")
    } else {
        index.to_path_buf()
    };
    // Before the open: `vec0` reaches a connection through SQLite's
    // auto-extension list, consulted once as the connection is created, so a
    // connection opened first could never query a corpus' shadow index. See
    // [`crate::vec0::register`].
    crate::vec0::register();
    let connection = Connection::open(path).context("opening index database")?;
    // A corpus can be read while a job is writing into it; wait out the writer's
    // commit instead of failing the query.
    connection.busy_timeout(BUSY_TIMEOUT)?;
    Ok(connection)
}

pub fn build_match(normalizer: &Normalizer, query: &str) -> String {
    let mut terms = words(query);
    terms.extend(words(query).into_iter().map(|word| fold(&word)));
    terms.extend(normalizer.query_tokens(query));
    terms.sort();
    terms.dedup();
    terms
        .into_iter()
        .filter(|term| !term.is_empty())
        .map(|term| format!("\"{}\"", term.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" OR ")
}

pub fn search(
    connection: &Connection,
    normalizer: &Normalizer,
    query: &str,
    limit: usize,
    fuzzy: bool,
) -> Result<Vec<SearchHit>> {
    let expression = build_match(normalizer, query);
    let mut statement = connection.prepare(
        "SELECT f.path,f.dir,f.lang,f.method,f.size,snippet(fts,2,'[',']',' … ',12) \
         FROM fts JOIN files f ON f.id=fts.rowid WHERE fts MATCH ?1 ORDER BY bm25(fts) LIMIT ?2",
    )?;
    let hits = statement
        .query_map(params![expression, limit as i64], |row| {
            Ok(SearchHit {
                path: row.get(0)?,
                dir: row.get(1)?,
                lang: row.get(2)?,
                method: row.get(3)?,
                size: row.get::<_, i64>(4)? as u64,
                snippet: row.get(5)?,
            })
        })?
        .flatten()
        .collect::<Vec<_>>();
    if !hits.is_empty() || !fuzzy {
        return Ok(hits);
    }
    fuzzy_names(connection, query, limit)
}

pub fn top_folders(
    connection: &Connection,
    normalizer: &Normalizer,
    query: &str,
    limit: usize,
) -> Result<Vec<(String, usize)>> {
    let expression = build_match(normalizer, query);
    let mut statement = connection.prepare(
        "SELECT f.dir,COUNT(*) FROM fts JOIN files f ON f.id=fts.rowid \
         WHERE fts MATCH ?1 GROUP BY f.dir ORDER BY COUNT(*) DESC LIMIT ?2",
    )?;
    let rows = statement
        .query_map(params![expression, limit as i64], |row| {
            Ok((row.get(0)?, row.get::<_, i64>(1)? as usize))
        })?
        .flatten()
        .collect();
    Ok(rows)
}

fn fuzzy_names(connection: &Connection, query: &str, limit: usize) -> Result<Vec<SearchHit>> {
    let mut statement = connection.prepare("SELECT path,dir,lang,method,size,name FROM files")?;
    let mut rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)? as u64,
                row.get::<_, String>(5)?,
            ))
        })?
        .flatten()
        .map(|row| {
            (
                strsim::jaro_winkler(&query.to_lowercase(), &row.5.to_lowercase()),
                row,
            )
        })
        .collect::<Vec<_>>();
    rows.sort_by(|a, b| b.0.total_cmp(&a.0));
    Ok(rows
        .into_iter()
        .take(limit)
        .map(|(score, row)| SearchHit {
            path: row.0,
            dir: row.1,
            lang: row.2,
            method: row.3,
            size: row.4,
            snippet: format!("~{:.0}% name match", score * 100.0),
        })
        .collect())
}

pub fn analyze(connection: &Connection) -> Result<Value> {
    let files: i64 = connection.query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))?;
    let bytes: i64 =
        connection.query_row("SELECT COALESCE(SUM(size),0) FROM files", [], |r| r.get(0))?;
    let ocr: i64 =
        connection.query_row("SELECT COALESCE(SUM(ocr_used),0) FROM files", [], |r| {
            r.get(0)
        })?;
    Ok(json!({
        "files": files,
        "bytes": bytes,
        "ocr_files": ocr,
        "extensions": grouped(connection, "ext", 30)?,
        "languages": grouped(connection, "lang", 10)?,
        "methods": grouped(connection, "method", 20)?,
        "top_folders_by_count": grouped(connection, "dir", 20)?,
    }))
}

/// Grouped counts for one `files` column, e.g. `("vi", 42)`. Shared with the
/// HTTP service's `/corpus/status` aggregates.
pub(crate) fn grouped(
    connection: &Connection,
    column: &str,
    limit: usize,
) -> Result<Vec<(String, i64)>> {
    let sql = format!(
        "SELECT {column},COUNT(*) FROM files GROUP BY {column} ORDER BY COUNT(*) DESC LIMIT ?1"
    );
    let mut statement = connection.prepare(&sql)?;
    let rows = statement
        .query_map([limit as i64], |row| Ok((row.get(0)?, row.get(1)?)))?
        .flatten()
        .collect();
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::FileRec;
    use crate::vision::{VisionMode, VisionResult};

    fn sample_file(path: &str) -> ProcessedFile {
        ProcessedFile {
            rec: FileRec {
                path: path.into(),
                name: "photo.jpg".into(),
                ext: ".jpg".into(),
                dir: "album".into(),
                drive: "/".into(),
                size: 10,
                mtime: 0.0,
            },
            content: "some indexed text".into(),
            tokens: vec!["some".into(), "indexed".into(), "text".into()],
            lang: "en".into(),
            method: "text".into(),
            ocr_used: false,
            pages: 0,
            sha1: None,
            chunks: Vec::new(),
            vision: None,
            elapsed_ms: 0,
        }
    }

    fn off_config() -> Config {
        let mut config = Config::default();
        config.sidecar = "none".into();
        config
    }

    #[test]
    fn database_path_addresses_a_file_or_a_directory() {
        // Service jobs name the published database; the CLI names its out dir.
        assert_eq!(
            database_path(Path::new("/out/corpus.sqlite")),
            PathBuf::from("/out/corpus.sqlite")
        );
        assert_eq!(
            database_path(Path::new("/out")),
            PathBuf::from("/out/index.sqlite")
        );
    }

    #[test]
    fn the_config_default_batch_matches_the_store_constant() {
        // An unset config must behave exactly as before commit_batch was tunable:
        // Config::default().commit_batch is default_commit_batch(), which must
        // equal the store's own COMMIT_FILES default.
        assert_eq!(Config::default().commit_batch, COMMIT_FILES);
    }

    #[test]
    fn a_smaller_commit_batch_commits_sooner() {
        // commit_batch=2 must durably commit after the 2nd file: a reader opening
        // the file mid-run (before finish) sees the committed rows. Proves the
        // setting reaches the writer's commit boundary rather than being ignored.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut config = off_config();
        config.commit_batch = 2;
        let mut store = IndexStore::open(&destination, &config, false, false).unwrap();
        store.add(&sample_file("/a/1.txt"), 0.0).unwrap();
        store.add(&sample_file("/a/2.txt"), 0.0).unwrap(); // 2nd file -> batch commits
        store.add(&sample_file("/a/3.txt"), 0.0).unwrap(); // opens a new batch

        // A SEPARATE read-only connection sees exactly the committed batch (2),
        // not the third file still in the open transaction.
        let committed: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            committed, 2,
            "commit_batch=2 must commit after the 2nd file"
        );

        store.finish().unwrap();
        let all: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(all, 3, "finish flushes the trailing partial batch");
    }

    #[test]
    fn opening_a_sqlite_destination_writes_that_file() {
        // Writing straight into the published corpus depends on `out` naming a
        // file: treating it as a directory would create `corpus.sqlite/` here.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&sample_file("/a/photo.jpg"), 0.0).unwrap();
        store.finish().unwrap();

        assert!(destination.is_file());
        let files: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        assert_eq!(files, 1);
    }

    fn chunk(index: usize) -> crate::embedding::EmbeddedChunk {
        crate::embedding::EmbeddedChunk {
            index,
            content: format!("chunk {index}"),
            vector: vec![0.5, 0.25],
        }
    }

    #[test]
    fn a_partially_failed_add_cannot_commit_a_chunkless_file() {
        // The failure that matters: the files and fts rows are already in, and
        // one of the file's chunks then fails. Committing that leaves a file
        // whose vectors are incomplete but which resume treats as done — it
        // holds at least one chunk, so `has_chunks` is true and nothing ever
        // revisits it. The row must not survive at all.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        let mut good = sample_file("/a/good.txt");
        good.chunks = vec![chunk(0)];
        store.add(&good, 0.0).unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TRIGGER fail_second_chunk BEFORE INSERT ON chunks \
                 WHEN NEW.chunk_index = 1 \
                 BEGIN SELECT RAISE(ABORT,'simulated chunk write failure'); END",
            )
            .unwrap();
        let mut broken = sample_file("/a/broken.txt");
        broken.chunks = vec![chunk(0), chunk(1)];

        let error = store
            .add(&broken, 0.0)
            .expect_err("a chunk that cannot be written fails the file");
        assert!(
            format!("{error:#}").contains("simulated chunk write failure"),
            "{error:#}"
        );
        // The run then ends the way any mid-run failure does: everything whole
        // is committed. That must not include the broken file.
        store.finish().unwrap();

        let connection = connect(&destination).unwrap();
        let paths: Vec<String> = connection
            .prepare("SELECT path FROM files ORDER BY path")
            .unwrap()
            .query_map([], |row| row.get(0))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        assert_eq!(paths, vec!["/a/good.txt".to_string()]);
        // No orphaned fts or chunk debris either.
        let fts: i64 = connection
            .query_row("SELECT COUNT(*) FROM fts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(fts, 1);
        let chunks: i64 = connection
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(chunks, 1, "only the complete file's chunk is published");
    }

    #[test]
    fn a_failed_file_leaves_earlier_committed_batches_intact() {
        // The same failure after a batch commit: the committed batch is already
        // durable and stays, which is the whole point of writing in place.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        for index in 0..3 {
            store
                .add(&sample_file(&format!("/a/file_{index}.txt")), 0.0)
                .unwrap();
        }
        store.commit().unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TRIGGER fail_chunks BEFORE INSERT ON chunks \
                 BEGIN SELECT RAISE(ABORT,'simulated chunk write failure'); END",
            )
            .unwrap();
        let mut late = sample_file("/a/late.txt");
        late.chunks = vec![chunk(0)];
        store.add(&late, 0.0).unwrap_err();
        store.finish().unwrap();

        let files: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM files", [], |row| row.get(0))
            .unwrap();
        assert_eq!(files, 3);
    }

    #[test]
    fn a_partial_corpus_reports_which_files_still_need_work() {
        // What an interrupted run leaves behind: some files complete with their
        // vector chunks, one whose extraction failed. Resume keys off exactly
        // these two columns, so both must survive a reopen accurately.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut embedded = sample_file("/a/done.txt");
        embedded.chunks = vec![crate::embedding::EmbeddedChunk {
            index: 0,
            content: "some indexed text".into(),
            vector: vec![0.5, 0.25],
        }];
        let mut failed = sample_file("/a/broken.pdf");
        failed.method = "error:poppler".into();
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&embedded, 0.0).unwrap();
        store.add(&failed, 0.0).unwrap();
        store.finish().unwrap();

        let store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        let existing = store.existing_keys().unwrap();
        let done = existing.get("/a/done.txt").unwrap();
        assert_eq!(done.method, "text");
        assert!(done.has_chunks, "a completed file is not redone on resume");
        assert_eq!(
            done.attempts, 0,
            "a finished row carries no failed attempts"
        );
        let broken = existing.get("/a/broken.pdf").unwrap();
        assert_eq!(broken.method, "error:poppler");
        assert!(
            !broken.has_chunks,
            "an unfinished file must be visible as such"
        );
        assert_eq!(
            broken.attempts, 1,
            "the failure that wrote it counts as one"
        );
    }

    #[test]
    fn off_path_writes_no_vision_rows() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store.add(&sample_file("/a/photo.jpg"), 0.0).unwrap();
        store.finish().unwrap();

        let connection = connect(temp.path()).unwrap();
        let files: i64 = connection
            .query_row("SELECT COUNT(*) FROM files", [], |r| r.get(0))
            .unwrap();
        let vision: i64 = connection
            .query_row("SELECT COUNT(*) FROM vision", [], |r| r.get(0))
            .unwrap();
        assert_eq!(files, 1);
        assert_eq!(vision, 0);
    }

    #[test]
    fn upsert_vision_round_trips_through_add() {
        let temp = tempfile::tempdir().unwrap();
        let mut file = sample_file("/a/photo.jpg");
        file.vision = Some(VisionResult {
            mode: VisionMode::Meta,
            width: Some(640),
            height: Some(480),
            phash: Some("00ff00ff00ff00ff".into()),
            elapsed_ms: Some(12),
            ..Default::default()
        });
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store.add(&file, 0.0).unwrap();
        store.finish().unwrap();

        let connection = connect(temp.path()).unwrap();
        let (mode, width, phash): (String, i64, String) = connection
            .query_row(
                "SELECT mode,width,phash FROM vision v JOIN files f ON f.id=v.file_id \
                 WHERE f.path=?1",
                params!["/a/photo.jpg"],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(mode, "meta");
        assert_eq!(width, 640);
        assert_eq!(phash, "00ff00ff00ff00ff");
    }

    fn tagged_photo(path: &str) -> ProcessedFile {
        let mut file = sample_file(path);
        file.vision = Some(VisionResult {
            mode: VisionMode::Tags,
            phash: Some("aaaaaaaaaaaaaaaa".into()),
            ..Default::default()
        });
        file
    }

    #[test]
    fn resume_drops_stale_vision_when_bytes_change() {
        let temp = tempfile::tempdir().unwrap();
        // Initial index: photo.jpg gets a vision row describing image A.
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store.add(&tagged_photo("/a/photo.jpg"), 0.0).unwrap();
        store.finish().unwrap();

        // Resume with vision OFF (vision=None) but the file's bytes changed
        // (size + mtime differ): the stale vision row must be dropped, not
        // silently re-attached to the new content.
        let mut changed = sample_file("/a/photo.jpg");
        changed.rec.size = 999;
        changed.rec.mtime = 123.0;
        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        store.add(&changed, 1.0).unwrap();
        store.finish().unwrap();

        let connection = connect(temp.path()).unwrap();
        let vision: i64 = connection
            .query_row("SELECT COUNT(*) FROM vision", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            vision, 0,
            "stale vision row must be dropped on content change"
        );
    }

    #[test]
    fn resume_keeps_vision_when_bytes_unchanged() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store.add(&tagged_photo("/a/photo.jpg"), 0.0).unwrap();
        store.finish().unwrap();

        // Resume with vision OFF and identical bytes: a lower/off tier must NOT
        // drop the existing vision row; it is carried forward to the new rowid.
        let mut same = sample_file("/a/photo.jpg");
        same.vision = None;
        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        store.add(&same, 1.0).unwrap();
        store.finish().unwrap();

        let connection = connect(temp.path()).unwrap();
        let (count, phash): (i64, String) = connection
            .query_row(
                "SELECT COUNT(*),COALESCE(MAX(phash),'') FROM vision v \
                 JOIN files f ON f.id=v.file_id WHERE f.path=?1",
                params!["/a/photo.jpg"],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(count, 1, "vision row carried forward on unchanged bytes");
        assert_eq!(phash, "aaaaaaaaaaaaaaaa");
    }

    fn face(x: i32, quality: f32) -> FaceDetection {
        FaceDetection {
            x,
            y: 5,
            width: 64,
            height: 80,
            quality,
            embedding: Some(vec![0.25, -0.5, 0.75]),
            frame: None,
        }
    }

    fn scanned_photo(path: &str, faces: Vec<FaceDetection>) -> ProcessedFile {
        let mut file = sample_file(path);
        file.vision = Some(VisionResult {
            mode: VisionMode::Tags,
            phash: Some("aaaaaaaaaaaaaaaa".into()),
            faces,
            faces_model: Some("yunet-sface".into()),
            ..Default::default()
        });
        file
    }

    fn stored_face_rows(temp: &Path) -> Vec<(i64, i64, f64, i64, String, Option<i64>)> {
        let connection = connect(temp).unwrap();
        let mut statement = connection
            .prepare(
                "SELECT face_index,x,quality,dimensions,model,frame FROM faces \
                 ORDER BY file_id,face_index",
            )
            .unwrap();
        let rows = statement
            .query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                ))
            })
            .unwrap()
            .map(Result::unwrap)
            .collect();
        rows
    }

    #[test]
    fn faces_round_trip_through_add_and_the_off_path_writes_none() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        // One scanned photo with two faces, one ordinary file with no vision.
        store
            .add(
                &scanned_photo("/a/photo.jpg", vec![face(10, 0.98), face(90, 0.91)]),
                0.0,
            )
            .unwrap();
        store.add(&sample_file("/a/notes.txt"), 0.0).unwrap();
        store.finish().unwrap();

        assert_eq!(
            stored_face_rows(temp.path()),
            vec![
                (0, 10, 0.98_f32 as f64, 3, "yunet-sface".to_string(), None),
                (1, 90, 0.91_f32 as f64, 3, "yunet-sface".to_string(), None),
            ]
        );
        let connection = connect(temp.path()).unwrap();
        let stamped: i64 = connection
            .query_row(
                "SELECT COUNT(*) FROM vision WHERE faces_model='yunet-sface'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stamped, 1, "the scan is stamped on the vision row");
        // The 128-d-shaped blob is little-endian f32, readable exactly as the
        // chunk vectors are — the app's clustering pass reads it the same way.
        let blob: Vec<u8> = connection
            .query_row(
                "SELECT embedding FROM faces WHERE face_index=0",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(blob, crate::embedding::vector_to_bytes(&[0.25, -0.5, 0.75]));
    }

    #[test]
    fn a_scan_that_found_nothing_is_recorded_as_a_scan() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store
            .add(&scanned_photo("/a/landscape.jpg", Vec::new()), 0.0)
            .unwrap();
        store.finish().unwrap();
        assert!(stored_face_rows(temp.path()).is_empty());
        let connection = connect(temp.path()).unwrap();
        let model: Option<String> = connection
            .query_row("SELECT faces_model FROM vision", [], |row| row.get(0))
            .unwrap();
        assert_eq!(model.as_deref(), Some("yunet-sface"));
    }

    #[test]
    fn re_scanning_replaces_rather_than_accumulates() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store
            .add(
                &scanned_photo("/a/photo.jpg", vec![face(10, 0.98), face(90, 0.91)]),
                0.0,
            )
            .unwrap();
        store.finish().unwrap();
        // A re-scan that finds ONE face must leave one row, not two: the tail of
        // the previous scan would otherwise keep claiming a person is in this file.
        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        store
            .add(&scanned_photo("/a/photo.jpg", vec![face(10, 0.99)]), 1.0)
            .unwrap();
        store.finish().unwrap();
        let rows = stored_face_rows(temp.path());
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].0, 0);
    }

    #[test]
    fn turning_faces_off_keeps_the_rows_but_changed_bytes_drop_them() {
        let temp = tempfile::tempdir().unwrap();
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store
            .add(&scanned_photo("/a/photo.jpg", vec![face(10, 0.98)]), 0.0)
            .unwrap();
        store.finish().unwrap();

        // Resume with faces OFF but vision still ON over identical bytes: the
        // vision row is rewritten by THIS job, so the face rows only survive
        // because they are carried forward on their own condition.
        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        let mut faces_off = sample_file("/a/photo.jpg");
        faces_off.vision = Some(VisionResult {
            mode: VisionMode::Tags,
            phash: Some("aaaaaaaaaaaaaaaa".into()),
            ..Default::default()
        });
        store.add(&faces_off, 1.0).unwrap();
        store.finish().unwrap();
        assert_eq!(
            stored_face_rows(temp.path()).len(),
            1,
            "turning faces off must not delete faces"
        );

        // Same again, but the bytes changed: the faces described the old content,
        // so they go.
        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        let mut changed = sample_file("/a/photo.jpg");
        changed.rec.size = 999;
        changed.rec.mtime = 123.0;
        store.add(&changed, 2.0).unwrap();
        store.finish().unwrap();
        assert!(
            stored_face_rows(temp.path()).is_empty(),
            "stale faces must be dropped on content change"
        );
    }

    #[test]
    fn pruning_a_vanished_file_takes_its_faces_with_it() {
        let temp = tempfile::tempdir().unwrap();
        // Walker-shaped paths: `prune_missing` matches root prefixes with the
        // platform separator, so a POSIX literal would prune nothing on Windows.
        let separator = std::path::MAIN_SEPARATOR;
        let root = format!("{separator}a");
        let gone = format!("{root}{separator}photo.jpg");
        let mut store = IndexStore::open(temp.path(), &off_config(), false, false).unwrap();
        store
            .add(&scanned_photo(&gone, vec![face(10, 0.98)]), 0.0)
            .unwrap();
        store.finish().unwrap();
        assert_eq!(stored_face_rows(temp.path()).len(), 1);

        let mut store = IndexStore::open(temp.path(), &off_config(), true, false).unwrap();
        let removed = store.prune_missing(&[root], &HashSet::new()).unwrap();
        store.finish().unwrap();
        assert_eq!(removed, 1);
        assert!(
            stored_face_rows(temp.path()).is_empty(),
            "a file that is gone takes the faces attributed to it with it"
        );
    }

    /// The `vision` table exactly as the shipped tiers created it — no
    /// `faces_model`. Written out in full for the same reason as
    /// [`LEGACY_SCHEMA`]: a fixture derived from [`SCHEMA`] would track the
    /// current columns and silently stop testing the migration.
    const PRE_FACES_VISION_SCHEMA: &str = "\
CREATE TABLE vision(
  file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
  mode TEXT NOT NULL,
  width INTEGER, height INTEGER,
  phash TEXT,
  exif_json TEXT, quality_json TEXT,
  objects_json TEXT,
  tags_json TEXT,
  caption TEXT,
  embedding BLOB, embedding_model TEXT, dimensions INTEGER,
  frames INTEGER,
  elapsed_ms INTEGER, error TEXT
);
";

    #[test]
    fn opening_a_pre_faces_corpus_adds_faces_model_and_keeps_the_vision_rows() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("index.sqlite");
        {
            let connection = Connection::open(&path).unwrap();
            connection.execute_batch(LEGACY_SCHEMA).unwrap();
            connection.execute_batch(PRE_FACES_VISION_SCHEMA).unwrap();
            connection
                .execute(
                    "INSERT INTO files(id,path,method,size,mtime,indexed_at) \
                     VALUES(1,'/a/photo.jpg','text',10,0.0,1700.0)",
                    [],
                )
                .unwrap();
            connection
                .execute(
                    "INSERT INTO vision(file_id,mode,phash) VALUES(1,'tags','abcdabcdabcdabcd')",
                    [],
                )
                .unwrap();
        }

        // Opening a STORE over it migrates in place (that is where the schema is
        // applied): the column appears, the existing tier/phash survive, and
        // `faces_model` is NULL — the truthful value, and exactly what makes the
        // first faces job pick the file up.
        IndexStore::open(&path, &off_config(), true, false)
            .unwrap()
            .finish()
            .unwrap();
        let connection = connect(&path).unwrap();
        let (mode, phash, faces_model): (String, String, Option<String>) = connection
            .query_row(
                "SELECT mode,phash,faces_model FROM vision WHERE file_id=1",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(mode, "tags");
        assert_eq!(phash, "abcdabcdabcdabcd");
        assert_eq!(faces_model, None);
        // The faces table itself arrives by `CREATE TABLE IF NOT EXISTS`, like
        // every other new table here, and starts empty.
        let faces: i64 = connection
            .query_row("SELECT COUNT(*) FROM faces", [], |row| row.get(0))
            .unwrap();
        assert_eq!(faces, 0);
        drop(connection);

        // Idempotent: a second open is a no-op, not a duplicate-column error.
        IndexStore::open(&path, &off_config(), true, false)
            .unwrap()
            .finish()
            .unwrap();
        let connection = connect(&path).unwrap();
        let columns = existing_columns(&connection, "vision").unwrap();
        assert!(columns.contains("faces_model"));
        assert_eq!(
            columns.iter().filter(|name| *name == "faces_model").count(),
            1
        );
    }

    /// The `files` table exactly as every corpus on disk was created: no attempt
    /// columns, because the release that wrote them had none. Written out in full
    /// rather than derived from [`SCHEMA`] — the whole point of the migration is
    /// the gap between the two, and a fixture that tracks the current schema
    /// would close that gap silently.
    const LEGACY_SCHEMA: &str = "\
CREATE TABLE files(
  id INTEGER PRIMARY KEY, path TEXT UNIQUE, drive TEXT, dir TEXT, name TEXT, ext TEXT,
  size INTEGER, mtime REAL, lang TEXT, method TEXT, ocr_used INTEGER, pages INTEGER,
  chars INTEGER, sha1 TEXT, indexed_at REAL
);
CREATE TABLE chunks(
  id INTEGER PRIMARY KEY,
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  chunk_index INTEGER NOT NULL, content TEXT NOT NULL, embedding BLOB NOT NULL,
  dimensions INTEGER NOT NULL, model TEXT NOT NULL, UNIQUE(file_id, chunk_index)
);
";

    /// A pre-migration corpus holding one row of every shape the live corpora
    /// hold: a finished file with its chunk, the three unfinished shapes that are
    /// re-attempted on every resume, and a terminal `excluded:` row.
    fn legacy_corpus(destination: &Path) {
        let connection = Connection::open(destination).unwrap();
        connection.execute_batch(LEGACY_SCHEMA).unwrap();
        for (id, path, method) in [
            (1, "/a/done.txt", "text"),
            (2, "/a/broken.pdf", "error:poppler"),
            (3, "/a/photo.heic", "name-only-partial"),
            (4, "/a/build.o", "name-only"),
            (5, "/a/~$memo.docx", "excluded:office-lock"),
        ] {
            connection
                .execute(
                    "INSERT INTO files(id,path,method,size,mtime,indexed_at) \
                     VALUES(?1,?2,?3,10,0.0,1700.0)",
                    params![id, path, method],
                )
                .unwrap();
        }
        connection
            .execute(
                "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                 VALUES(1,0,'text',X'00',2,'m')",
                [],
            )
            .unwrap();
    }

    fn attempt_columns(destination: &Path) -> Vec<(String, i64, Option<f64>)> {
        let connection = connect(destination).unwrap();
        let mut statement = connection
            .prepare("SELECT path,attempts,last_attempt_at FROM files ORDER BY id")
            .unwrap();
        let rows = statement
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .map(Result::unwrap)
            .collect();
        rows
    }

    #[test]
    fn opening_a_legacy_corpus_adds_the_attempt_columns_and_stamps_the_unfinished_rows() {
        // The migration a live corpus gets on its first open by this build. The
        // unfinished rows — the ~69% re-attempted on every resume — are stamped as
        // spent, so the resume right after the deploy converges instead of
        // re-burning them MAX_ATTEMPTS more times. Finished and `excluded:` rows
        // are left at zero: neither has ever failed.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        legacy_corpus(&destination);

        IndexStore::open(&destination, &off_config(), true, false)
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(
            attempt_columns(&destination),
            vec![
                ("/a/done.txt".into(), 0, None),
                (
                    "/a/broken.pdf".into(),
                    i64::from(MAX_ATTEMPTS),
                    Some(1700.0)
                ),
                (
                    "/a/photo.heic".into(),
                    i64::from(MAX_ATTEMPTS),
                    Some(1700.0)
                ),
                ("/a/build.o".into(), i64::from(MAX_ATTEMPTS), Some(1700.0)),
                ("/a/~$memo.docx".into(), 0, None),
            ]
        );
    }

    #[test]
    fn the_migration_is_idempotent_across_reopens() {
        // Every open runs it, so it has to be free the second time AND must not
        // re-stamp: a row that has since succeeded is back at zero attempts, and a
        // backfill that fired again would declare it spent.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        legacy_corpus(&destination);
        IndexStore::open(&destination, &off_config(), true, false)
            .unwrap()
            .finish()
            .unwrap();

        // The repair a later resume performs: the file finally extracted, so its
        // row is finished and its counter reset.
        connect(&destination)
            .unwrap()
            .execute(
                "UPDATE files SET method='pdf',attempts=0,last_attempt_at=NULL \
                 WHERE path='/a/broken.pdf'",
                [],
            )
            .unwrap();

        for _ in 0..2 {
            IndexStore::open(&destination, &off_config(), true, false)
                .unwrap()
                .finish()
                .unwrap();
        }

        let after = attempt_columns(&destination);
        assert_eq!(
            after[1],
            ("/a/broken.pdf".into(), 0, None),
            "a repaired row must not be re-stamped by a later open"
        );
        assert_eq!(
            after[2],
            (
                "/a/photo.heic".into(),
                i64::from(MAX_ATTEMPTS),
                Some(1700.0)
            ),
            "and the rows stamped once must not climb on every open"
        );
    }

    #[test]
    fn a_corpus_this_build_created_needs_no_migration() {
        // The other half of idempotency: SCHEMA already carries the columns, so a
        // fresh corpus must come out of open with nothing stamped — the backfill
        // fires on the column's creation, and here it was never absent.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        let mut failed = sample_file("/a/broken.pdf");
        failed.method = "error:poppler".into();
        store.add(&failed, 1700.0).unwrap();
        store.finish().unwrap();

        IndexStore::open(&destination, &off_config(), true, false)
            .unwrap()
            .finish()
            .unwrap();

        assert_eq!(
            attempt_columns(&destination),
            vec![("/a/broken.pdf".into(), 1, Some(1700.0))],
            "the row's own single failure, not a backfill"
        );
    }

    #[test]
    fn attempts_accumulate_on_failure_and_reset_on_success() {
        // What makes the cap reachable at all. A file that keeps failing counts
        // up; the run that finally reads it puts the counter back to zero so a
        // later failure gets the full budget again.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut failed = sample_file("/a/broken.pdf");
        failed.method = "error:poppler".into();
        failed.elapsed_ms = 4200;
        for expected in 1..=3 {
            let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
            store.add(&failed, 1700.0).unwrap();
            store.finish().unwrap();
            assert_eq!(
                attempt_columns(&destination)[0].1,
                expected,
                "each failing attempt counts once"
            );
        }
        let elapsed: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT elapsed_ms FROM files", [], |row| row.get(0))
            .unwrap();
        assert_eq!(elapsed, 4200, "an error row carries what it cost");

        let mut fixed = sample_file("/a/broken.pdf");
        fixed.method = "pdf".into();
        fixed.chunks = vec![chunk(0)];
        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        store.add(&fixed, 1800.0).unwrap();
        store.finish().unwrap();
        assert_eq!(attempt_columns(&destination)[0].1, 0);
    }

    #[test]
    fn changed_bytes_open_a_fresh_attempt_budget() {
        // The row's failures belong to the bytes that produced them. A file that
        // has since been rewritten is a different file at the same path, so it
        // starts its own count rather than inheriting a spent one.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut failed = sample_file("/a/broken.pdf");
        failed.method = "error:poppler".into();
        for _ in 0..3 {
            let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
            store.add(&failed, 1700.0).unwrap();
            store.finish().unwrap();
        }
        assert_eq!(attempt_columns(&destination)[0].1, 3);

        let mut rewritten = failed.clone();
        rewritten.rec.size = 999;
        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        store.add(&rewritten, 1800.0).unwrap();
        store.finish().unwrap();
        assert_eq!(
            attempt_columns(&destination)[0].1,
            1,
            "new bytes, first failure"
        );
    }

    /// A corpus with a shadow index over `dimensions`-wide vectors, built from
    /// whatever `chunks` already holds. Returns the destination.
    fn indexed_corpus(destination: &Path, dimensions: usize) {
        let mut connection = connect(destination).unwrap();
        crate::vec0::build(
            &mut connection,
            crate::embedding::EMBEDDING_MODEL,
            dimensions,
            |_, _| {},
        )
        .unwrap();
    }

    /// `(rows in the shadow index, recorded state)` for a corpus on disk.
    fn shadow(destination: &Path) -> (i64, crate::vec0::IndexState) {
        let connection = connect(destination).unwrap();
        let rows = connection
            .query_row(
                &format!("SELECT COUNT(*) FROM {}", crate::vec0::SHADOW_TABLE),
                [],
                |row| row.get(0),
            )
            .unwrap();
        (rows, crate::vec0::state(&connection).unwrap().unwrap())
    }

    /// A file carrying `count` chunks whose vectors are 2 floats wide, matching
    /// [`chunk`].
    fn embedded_file(path: &str, count: usize) -> ProcessedFile {
        let mut file = sample_file(path);
        file.chunks = (0..count).map(chunk).collect();
        file
    }

    #[test]
    fn an_index_job_keeps_an_existing_shadow_index_in_step_with_the_chunks() {
        // Incremental maintenance, end to end through the writer that jobs use:
        // new chunks are mirrored as they are written, a re-indexed file's old
        // vectors go with its old rows, and the recorded counts move with both
        // so a reader can still prove the index covers the corpus.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&embedded_file("/a/one.txt", 2), 0.0).unwrap();
        store.finish().unwrap();
        indexed_corpus(&destination, 2);
        assert_eq!(shadow(&destination).0, 2);

        // A later job adds a file and re-indexes the first with fewer chunks.
        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        store.add(&embedded_file("/a/two.txt", 3), 1.0).unwrap();
        let mut shrunk = embedded_file("/a/one.txt", 1);
        shrunk.rec.size = 999; // changed bytes, so the row is genuinely replaced
        store.add(&shrunk, 1.0).unwrap();
        store.finish().unwrap();

        let (rows, state) = shadow(&destination);
        assert_eq!(rows, 4, "3 new + 1 replacement, the old 2 removed");
        assert_eq!(state.vectors, 4);
        assert_eq!(state.chunks, 4);
        // The witness a reader checks: recorded chunk count == live chunk count.
        let live: i64 = connect(&destination)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM chunks", [], |row| row.get(0))
            .unwrap();
        assert_eq!(live as usize, state.chunks);
    }

    #[test]
    fn a_pruned_file_takes_its_vectors_out_of_the_shadow_index() {
        // The other deletion path. Without it a k-NN keeps nominating chunks of
        // a file that no longer exists, and every such candidate is dropped by
        // the re-score — silently shortening result pages.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        // Walker-shaped paths: `prune_missing` matches root prefixes with the
        // platform separator, so a POSIX literal would prune nothing on Windows.
        let separator = std::path::MAIN_SEPARATOR;
        let root = format!("{separator}a");
        let kept = format!("{root}{separator}kept.txt");
        let gone = format!("{root}{separator}gone.txt");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&embedded_file(&kept, 1), 0.0).unwrap();
        store.add(&embedded_file(&gone, 2), 0.0).unwrap();
        store.finish().unwrap();
        indexed_corpus(&destination, 2);

        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        let pruned = store
            .prune_missing(&[root], &HashSet::from([kept]))
            .unwrap();
        store.finish().unwrap();

        assert_eq!(pruned, 1);
        let (rows, state) = shadow(&destination);
        assert_eq!(rows, 1);
        assert_eq!(state.vectors, 1);
        assert_eq!(state.chunks, 1);
    }

    #[test]
    fn a_rolled_back_file_leaves_the_shadow_index_counts_where_it_found_them() {
        // `add` rolls a failed file back, and the index' counts live in memory
        // as well as in the transaction. A count stranded by the failure would
        // be stamped onto the NEXT file and mark the index stale for good.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&embedded_file("/a/good.txt", 1), 0.0).unwrap();
        store.finish().unwrap();
        indexed_corpus(&destination, 2);

        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        store
            .connection
            .execute_batch(
                "CREATE TRIGGER fail_second_chunk BEFORE INSERT ON chunks \
                 WHEN NEW.chunk_index = 1 \
                 BEGIN SELECT RAISE(ABORT,'simulated chunk write failure'); END",
            )
            .unwrap();
        store
            .add(&embedded_file("/a/broken.txt", 2), 1.0)
            .unwrap_err();
        store
            .connection
            .execute_batch("DROP TRIGGER fail_second_chunk")
            .unwrap();
        store.add(&embedded_file("/a/later.txt", 1), 1.0).unwrap();
        store.finish().unwrap();

        let (rows, state) = shadow(&destination);
        assert_eq!(rows, 2, "the failed file left nothing behind");
        assert_eq!(state.vectors, 2);
        assert_eq!(state.chunks, 2);
        let connection = connect(&destination).unwrap();
        assert!(matches!(
            crate::vec0::usable(&connection, crate::embedding::EMBEDDING_MODEL, 2).unwrap(),
            crate::vec0::Usable::Ready(_)
        ));
    }

    #[test]
    fn a_corpus_without_a_shadow_index_gains_neither_table_nor_marker() {
        // The default. An index job on an unindexed corpus writes exactly what
        // it wrote before the index existed — no table, no `meta` key, and
        // nothing in the write path that fires.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let separator = std::path::MAIN_SEPARATOR;
        let root = format!("{separator}a");
        let only = format!("{root}{separator}one.txt");
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&embedded_file(&only, 2), 0.0).unwrap();
        store
            .prune_missing(&[root], &HashSet::from([only]))
            .unwrap();
        store.finish().unwrap();

        let connection = connect(&destination).unwrap();
        assert!(!crate::vec0::present(&connection).unwrap());
        assert!(crate::vec0::state(&connection).unwrap().is_none());
        assert!(matches!(
            crate::vec0::usable(&connection, crate::embedding::EMBEDDING_MODEL, 2).unwrap(),
            crate::vec0::Usable::Absent
        ));
    }

    #[test]
    fn a_kept_row_still_records_the_attempt_that_was_thrown_away() {
        // Keep-on-failure writes nothing, so without this the run that re-read,
        // re-OCR'd and re-embedded the file and then discarded the result leaves
        // no trace of having spent anything. The row's own content and
        // `indexed_at` must not move — they describe what is stored.
        let temp = tempfile::tempdir().unwrap();
        let destination = temp.path().join("corpus.sqlite");
        let mut good = sample_file("/a/report.pdf");
        good.method = "pdf".into();
        good.chunks = vec![chunk(0)];
        let mut store = IndexStore::open(&destination, &off_config(), false, false).unwrap();
        store.add(&good, 1700.0).unwrap();
        store.finish().unwrap();

        let mut store = IndexStore::open(&destination, &off_config(), true, false).unwrap();
        store
            .record_failed_attempt("/a/report.pdf", 9000, 1800.0)
            .unwrap();
        store.finish().unwrap();

        let (method, indexed_at, attempts, last, elapsed): (String, f64, i64, f64, i64) =
            connect(&destination)
                .unwrap()
                .query_row(
                    "SELECT method,indexed_at,attempts,last_attempt_at,elapsed_ms FROM files",
                    [],
                    |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                        ))
                    },
                )
                .unwrap();
        assert_eq!(method, "pdf", "the kept row keeps its content");
        assert_eq!(indexed_at, 1700.0, "and the time that content was indexed");
        assert_eq!(attempts, 1);
        assert_eq!(last, 1800.0);
        assert_eq!(elapsed, 9000, "the burned time is now measurable");
    }
}
