use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

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

pub struct IndexStore {
    out: PathBuf,
    connection: Connection,
    resume: bool,
    sidecar: String,
    jsonl: Option<BufWriter<File>>,
    catalog: Option<csv::Writer<File>>,
    pending: usize,
}

impl IndexStore {
    pub fn open(out: &Path, config: &Config, resume: bool, artifacts: bool) -> Result<Self> {
        fs::create_dir_all(out)?;
        let connection = Connection::open(out.join("index.sqlite"))?;
        connection
            .execute_batch(SCHEMA)
            .context("creating SQLite FTS5 schema")?;
        let jsonl = if artifacts {
            let file = OpenOptions::new()
                .create(true)
                .write(true)
                .append(resume)
                .truncate(!resume)
                .open(out.join("manifest.jsonl"))?;
            Some(BufWriter::new(file))
        } else {
            None
        };
        let catalog_path = out.join("catalog.csv");
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
            out: out.to_path_buf(),
            connection,
            resume,
            sidecar: config.sidecar.clone(),
            jsonl,
            catalog,
            pending: 0,
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

    pub fn prune_missing(&mut self, current: &HashSet<String>) -> Result<usize> {
        let mut statement = self.connection.prepare("SELECT id,path FROM files")?;
        let stale = statement
            .query_map([], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?
            .flatten()
            .filter(|(_, path)| !current.contains(path))
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
        self.pending += 1;
        if self.pending >= 500 {
            self.connection.execute_batch("COMMIT; BEGIN IMMEDIATE")?;
            self.pending = 0;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
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
    Connection::open(path).context("opening index database")
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
