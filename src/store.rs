use std::collections::{HashMap, HashSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rusqlite::{params, Connection};
use serde_json::{json, Value};

use crate::config::Config;
use crate::model::{ProcessedFile, SearchHit};
use crate::normalize::{fold, words, Normalizer};

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
                .execute("DELETE FROM fts WHERE rowid=?1", [id])?;
            self.connection
                .execute("DELETE FROM files WHERE id=?1", [id])?;
        }
        Ok(stale.len())
    }

    pub fn add(&mut self, file: &ProcessedFile, indexed_at: f64) -> Result<()> {
        if self.resume {
            if let Ok(old_id) = self.connection.query_row(
                "SELECT id FROM files WHERE path=?1",
                [&file.rec.path],
                |row| row.get::<_, i64>(0),
            ) {
                self.connection
                    .execute("DELETE FROM chunks WHERE file_id=?1", [old_id])?;
                self.connection
                    .execute("DELETE FROM fts WHERE rowid=?1", [old_id])?;
            }
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
