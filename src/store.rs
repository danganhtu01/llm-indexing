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
use crate::vision::VisionResult;

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
  indexed_at REAL
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
  elapsed_ms INTEGER, error TEXT
);
CREATE INDEX IF NOT EXISTS idx_vision_phash ON vision(phash);
"#;

/// How long a WRITER waits for a lock before giving up. An indexing job now
/// writes into the same file readers open, and a batch commit holds an exclusive
/// lock for the duration of one flush, so both sides must wait rather than fail.
pub const BUSY_TIMEOUT: Duration = Duration::from_secs(30);

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
    fs::remove_file(&database)
        .with_context(|| format!("replacing {}", database.display()))?;
    // Best effort: an orphaned journal with no database is inert, and SQLite
    // discards one whose header does not match the database it opens.
    let _ = fs::remove_file(journal_path(&database));
    Ok(())
}

pub struct IndexStore {
    out: PathBuf,
    connection: Connection,
    resume: bool,
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
}

impl IndexStore {
    pub fn open(out: &Path, config: &Config, resume: bool, artifacts: bool) -> Result<Self> {
        let database = database_path(out);
        // Artifacts and sidecars live beside the database whichever way `out`
        // addressed it.
        let root = database.parent().unwrap_or(Path::new(".")).to_path_buf();
        fs::create_dir_all(&root)?;
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
        connection.execute_batch("BEGIN IMMEDIATE")?;
        Ok(Self {
            out: root,
            connection,
            resume,
            sidecar: config.sidecar.clone(),
            jsonl,
            catalog,
            pending: 0,
            committed: Instant::now(),
            commit_batch: config.commit_batch.max(1),
            poisoned: false,
        })
    }

    pub fn existing_keys(&self) -> Result<HashMap<String, (u64, i64, String, bool)>> {
        let mut statement = self
            .connection
            .prepare(
                "SELECT f.path,f.size,f.mtime,f.method,EXISTS(SELECT 1 FROM chunks c WHERE c.file_id=f.id) \
                 FROM files f",
            )?;
        let rows = statement.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)? as u64,
                row.get::<_, f64>(2)? as i64,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)? != 0,
            ))
        })?;
        Ok(rows
            .flatten()
            .map(|(path, size, mtime, method, has_chunks)| {
                (path, (size, mtime, method, has_chunks))
            })
            .collect())
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
            self.connection
                .execute("DELETE FROM chunks WHERE file_id=?1", [id])?;
            self.connection
                .execute("DELETE FROM vision WHERE file_id=?1", [id])?;
            self.connection
                .execute("DELETE FROM fts WHERE rowid=?1", [id])?;
            self.connection
                .execute("DELETE FROM files WHERE id=?1", [id])?;
        }
        Ok(stale.len())
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
        if let Err(error) = self.write_rows(file, indexed_at) {
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

    /// Every database row one file contributes, run inside the caller's
    /// savepoint so the set lands whole or not at all.
    fn write_rows(&mut self, file: &ProcessedFile, indexed_at: f64) -> Result<()> {
        // On resume the file row is INSERT OR REPLACE'd, which mints a new rowid
        // on the UNIQUE(path) conflict, so the previous id's chunks/fts are
        // deleted here and its vision row is reconciled after the re-insert.
        let old = if self.resume {
            self.connection
                .query_row(
                    "SELECT id,size,mtime FROM files WHERE path=?1",
                    [&file.rec.path],
                    |row| {
                        Ok((
                            row.get::<_, i64>(0)?,
                            row.get::<_, i64>(1)?,
                            row.get::<_, f64>(2)?,
                        ))
                    },
                )
                .ok()
        } else {
            None
        };
        let old_id = old.map(|(id, _, _)| id);
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
                     tags_json,caption,embedding,embedding_model,dimensions,frames,elapsed_ms,error \
                     FROM vision WHERE file_id=?1",
                    [old_id],
                    |row| (0..15).map(|i| row.get::<_, rusqlite::types::Value>(i)).collect(),
                )
                .optional()?,
            _ => None,
        };
        if let Some(old_id) = old_id {
            self.connection
                .execute("DELETE FROM chunks WHERE file_id=?1", [old_id])?;
            self.connection
                .execute("DELETE FROM fts WHERE rowid=?1", [old_id])?;
        }
        self.connection.execute(
            "INSERT OR REPLACE INTO files(path,drive,dir,name,ext,size,mtime,lang,method,ocr_used,pages,chars,sha1,indexed_at) \
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14)",
            params![file.rec.path, file.rec.drive, file.rec.dir, file.rec.name, file.rec.ext,
                file.rec.size as i64, file.rec.mtime, file.lang, file.method, file.ocr_used as i64,
                file.pages as i64, file.content.chars().count() as i64, file.sha1, indexed_at])?;
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
            self.connection.execute(
                "INSERT INTO chunks(file_id,chunk_index,content,embedding,dimensions,model) \
                 VALUES(?1,?2,?3,?4,?5,?6)",
                params![
                    id,
                    chunk.index as i64,
                    chunk.content,
                    crate::embedding::vector_to_bytes(&chunk.vector),
                    chunk.vector.len() as i64,
                    crate::embedding::EMBEDDING_MODEL,
                ],
            )?;
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
            }
            (None, Some(old_id)) => {
                // Belt-and-braces: on a foreign_keys=OFF build the old row would
                // survive under the stale id, so clear it before re-attaching.
                self.connection
                    .execute("DELETE FROM vision WHERE file_id=?1", [old_id])?;
                let unchanged = old.is_some_and(|(_, size, mtime)| {
                    size == file.rec.size as i64 && mtime as i64 == file.rec.mtime as i64
                });
                if let (true, Some(values)) = (unchanged, carried_vision) {
                    let mut row: Vec<rusqlite::types::Value> = Vec::with_capacity(16);
                    row.push(rusqlite::types::Value::Integer(id));
                    row.extend(values);
                    self.connection.execute(
                        "INSERT OR REPLACE INTO vision(file_id,mode,width,height,phash,exif_json,\
                         quality_json,objects_json,tags_json,caption,embedding,embedding_model,\
                         dimensions,frames,elapsed_ms,error) \
                         VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
                        rusqlite::params_from_iter(row),
                    )?;
                }
            }
            (None, None) => {}
        }
        Ok(())
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
        if self.sidecar != "none"
            && !file.content.trim().is_empty()
            && !matches!(file.method.as_str(), "text" | "name-only")
            && !file.method.starts_with("error:")
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
             objects_json,tags_json,caption,embedding,embedding_model,dimensions,frames,elapsed_ms,error) \
             VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16)",
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
            ],
        )?;
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
        assert_eq!(committed, 2, "commit_batch=2 must commit after the 2nd file");

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
        let (_, _, method, has_chunks) = existing.get("/a/done.txt").unwrap();
        assert_eq!(method, "text");
        assert!(has_chunks, "a completed file is not redone on resume");
        let (_, _, method, has_chunks) = existing.get("/a/broken.pdf").unwrap();
        assert_eq!(method, "error:poppler");
        assert!(!has_chunks, "an unfinished file must be visible as such");
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
}
