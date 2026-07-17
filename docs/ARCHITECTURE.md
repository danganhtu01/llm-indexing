# Architecture

## Pipeline

```text
paths -> walker -> extractor/OCR -> language normalization -> SQLite FTS5
```

The walker applies symlink, directory, extension, and size rules. Rayon workers
extract and normalize files in parallel; a single writer owns SQLite and commits
in batches. File name and path tokens remain searchable when content extraction
is unsupported or fails.

Extraction is implemented in Rust for text, email, and ZIP/XML Office formats.
The process invokes Poppler for PDFs and Tesseract for OCR. `ocr: auto` preserves
a PDF text layer and rasterizes only when that layer is empty. OCR work is capped
by `ocr_max_pages`.

## Retrieval

Normalization combines lowercased words, Unicode diacritic folding, English
Snowball stems, Vietnamese maximum-matching compounds, and editable EN/VI
abbreviation expansions. Raw content and enriched tokens are both stored in FTS.

```sql
files(id, path UNIQUE, drive, dir, name, ext, size, mtime,
      lang, method, ocr_used, pages, chars, sha1, indexed_at)
fts USING fts5(name, path, content, tokens,
               tokenize="unicode61 remove_diacritics 2 tokenchars '_'")
```

The `search` and `top-folder` commands normalize queries with the same rules.
Results are ranked with `bm25()`; folder aggregation groups matching file rows.

## Service Boundary

The Axum service exposes health, submission, and job-status endpoints. A bounded
Tokio channel serializes indexing jobs so concurrent requests cannot exhaust the
host. Request bodies, pending jobs, worker threads, and retained job history are
bounded.

All input paths are canonicalized and must remain under configured read-only
roots. Output is restricted to a plain `.sqlite` filename beneath the configured
output root. Each job builds in a temporary output directory and renames the
completed database into place. Failed jobs leave the prior database intact.

The image runs as UID/GID 10001 with no Linux capabilities, a read-only root
filesystem under Compose, read-only inputs, and a writable output mount.

## Language Extensions

OCR language selection is data-driven through `ocr_langs`; installing an
additional Tesseract trained-data package enables character recognition for that
language. Retrieval quality beyond raw Unicode matching requires adding language
detection and normalization rules in `normalize.rs`, plus relevant dictionaries
under `data/`. No protocol or SQLite schema change is required.
