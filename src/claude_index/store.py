"""Index sinks and readers.

Writes four open, engine-agnostic artifacts:
  * index.sqlite   - SQLite FTS5 full-text index (primary; query via SQL/bm25)
  * manifest.jsonl - one JSON object per file (ingest into ES/OpenSearch/etc.)
  * catalog.csv    - flat catalog (open in Excel / Explorer)
  * sidecar/*.txt  - extracted text next to a mirror tree, so Windows Explorer
                     content-search can find scanned/binary docs by their text.

The FTS tokenizer folds diacritics (``remove_diacritics 2``) and keeps ``_`` as a
token char so Vietnamese compounds (``ngan_hang``) stay searchable as one term.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
from pathlib import Path

from .normalize import fold

SCHEMA = """
CREATE TABLE IF NOT EXISTS files(
  id INTEGER PRIMARY KEY,
  path TEXT UNIQUE, drive TEXT, dir TEXT, name TEXT, ext TEXT,
  size INTEGER, mtime REAL, lang TEXT, method TEXT, ocr_used INTEGER,
  pages INTEGER, chars INTEGER, sha1 TEXT, indexed_at REAL
);
CREATE INDEX IF NOT EXISTS idx_files_dir ON files(dir);
CREATE INDEX IF NOT EXISTS idx_files_ext ON files(ext);
CREATE VIRTUAL TABLE IF NOT EXISTS fts USING fts5(
  name, path, content, tokens,
  tokenize="unicode61 remove_diacritics 2 tokenchars '_'"
);
"""


class IndexStore:
    """Writer. Use as a context manager or call close() when done."""

    def __init__(self, out_dir, cfg, resume=False):
        self.cfg = cfg
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.sidecar_mode = cfg.get("sidecar", "mirror")
        self.db = sqlite3.connect(str(self.out / "index.sqlite"))
        self.db.executescript(SCHEMA)
        # resume: append to existing manifest/catalog instead of truncating them
        jsonl_path = self.out / "manifest.jsonl"
        csv_path = self.out / "catalog.csv"
        csv_append = resume and csv_path.exists() and csv_path.stat().st_size > 0
        self._jsonl = open(jsonl_path, "a" if (resume and jsonl_path.exists()) else "w",
                           encoding="utf-8")
        self._csv_f = open(csv_path, "a" if csv_append else "w",
                           newline="", encoding="utf-8-sig")
        self._csv = csv.writer(self._csv_f)
        if not csv_append:
            self._csv.writerow(["path", "name", "ext", "size", "mtime", "lang",
                                "method", "ocr_used", "chars"])
        self._n = 0

    def existing_keys(self):
        """path -> (size, int(mtime)) for already-indexed files (resume skip-set)."""
        return {p: (s, int(m))
                for p, s, m in self.db.execute("SELECT path, size, mtime FROM files")}

    def add(self, rec, text, tokens, lang, method, ocr_used, pages, sha1, indexed_at):
        chars = len(text)
        if self.resume:
            # drop any prior FTS row for this path so a re-add doesn't orphan it
            old = self.db.execute("SELECT id FROM files WHERE path=?", (rec.path,)).fetchone()
            if old is not None:
                self.db.execute("DELETE FROM fts WHERE rowid=?", (old[0],))
        cur = self.db.execute(
            "INSERT OR REPLACE INTO files(path,drive,dir,name,ext,size,mtime,lang,"
            "method,ocr_used,pages,chars,sha1,indexed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (rec.path, rec.drive, rec.dir, rec.name, rec.ext, rec.size, rec.mtime,
             lang, method, int(ocr_used), pages, chars, sha1, indexed_at),
        )
        fid = cur.lastrowid
        self.db.execute(
            "INSERT INTO fts(rowid,name,path,content,tokens) VALUES(?,?,?,?,?)",
            (fid, rec.name, rec.path, text, " ".join(tokens)),
        )
        self._jsonl.write(json.dumps({
            "path": rec.path, "name": rec.name, "ext": rec.ext, "dir": rec.dir,
            "drive": rec.drive, "size": rec.size, "mtime": rec.mtime, "lang": lang,
            "method": method, "ocr_used": ocr_used, "pages": pages, "chars": chars,
            "snippet": text[:400],
        }, ensure_ascii=False) + "\n")
        self._csv.writerow([rec.path, rec.name, rec.ext, rec.size, f"{rec.mtime:.0f}",
                            lang, method, int(ocr_used), chars])
        # sidecars only for real extracted content (not plaintext, name-only, or errors)
        if (self.sidecar_mode != "none" and text.strip()
                and method not in ("text", "name-only")
                and not method.startswith("error")):
            self._write_sidecar(rec, text)
        self._n += 1
        if self._n % 500 == 0:
            self.db.commit()

    def _write_sidecar(self, rec, text):
        try:
            if self.sidecar_mode == "inplace":
                target = Path(rec.path + ".txt")
            else:
                rel = os.path.splitdrive(rec.path)[1].lstrip("\\/")
                target = (self.out / "sidecar" / rec.drive.replace(":", "") / rel)
                target = target.with_name(target.name + ".txt")
                target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "w", encoding="utf-8", errors="ignore") as f:
                f.write(text)
        except Exception:
            pass

    def close(self):
        self.db.commit()
        for f in (self._jsonl, self._csv_f):
            try:
                f.close()
            except Exception:
                pass
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# --------------------------- read side ---------------------------

def connect(out_dir) -> sqlite3.Connection:
    return sqlite3.connect(str(Path(out_dir) / "index.sqlite"))


def build_match(normalizer, query: str) -> str:
    """Build a broad FTS5 MATCH expression (OR of raw + enriched terms)."""
    terms = set()
    for w in re.findall(r"[^\W_]+", query.lower(), re.UNICODE):
        terms.add(w)
        terms.add(fold(w))
    terms.update(normalizer.query_tokens(query))
    quoted = ['"%s"' % t.replace('"', '""') for t in terms if t]
    return " OR ".join(quoted) if quoted else '""'


def search(db, normalizer, query, limit=20, fuzzy=False):
    match = build_match(normalizer, query)
    rows = []
    try:
        rows = db.execute(
            "SELECT f.path, f.dir, f.lang, f.method, f.size, "
            "snippet(fts,2,'[',']',' … ',12) "
            "FROM fts JOIN files f ON f.id = fts.rowid "
            "WHERE fts MATCH ? ORDER BY bm25(fts) LIMIT ?",
            (match, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []
    if not rows and fuzzy:
        rows = _fuzzy_name(db, query, limit)
    return rows


def top_folders(db, normalizer, query, n=10):
    match = build_match(normalizer, query)
    try:
        return db.execute(
            "SELECT f.dir, COUNT(*) c FROM fts JOIN files f ON f.id = fts.rowid "
            "WHERE fts MATCH ? GROUP BY f.dir ORDER BY c DESC LIMIT ?",
            (match, n),
        ).fetchall()
    except sqlite3.OperationalError:
        return []


def _fuzzy_name(db, query, limit):
    from rapidfuzz import fuzz, process
    rows = db.execute("SELECT path, dir, lang, method, size, name FROM files").fetchall()
    if not rows:
        return []
    names = {i: r[5] for i, r in enumerate(rows)}
    out = []
    for _, score, idx in process.extract(query, names, scorer=fuzz.WRatio, limit=limit):
        r = rows[idx]
        out.append((r[0], r[1], r[2], r[3], r[4], f"~{score:.0f}% name match"))
    return out
