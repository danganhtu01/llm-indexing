# Architecture
## Pipeline

```text
mounted tree
  -> confined walker
  -> document/archive/media extraction
  -> exhaustive OCR + Whisper transcription
  -> EN/VI normalization
  -> SQLite files + FTS5 + embedded chunks
```

The walker rejects symlink escapes and applies directory/extension rules. Rayon
workers extract content in parallel; one writer owns SQLite and commits in
batches. Archives are unpacked through `bsdtar` only after safe relative-path
validation, with a four-level recursion limit and 10,000-entry bound.

### Extraction completeness

Normal `auto`, `on` and `off` modes retain configurable byte, character and OCR
page limits. `exhaustive` bypasses those caps, rasterizes every PDF page at 250
DPI, OCRs it even when Poppler found a text layer, and combines both results.
Modern Office/ODF files contribute XML text and embedded-image OCR. Audio and
video are decoded by FFmpeg to 16 kHz mono PCM and transcribed with the pinned
multilingual Whisper small model; exhaustive video processing also OCRs sampled
frames every 30 seconds.

An extraction method ending in `-partial`, beginning `error:`, or equal to
`name-only` is counted as incomplete. Empty extraction is partial. Unsupported
or failed content is therefore visible to the caller and is not treated as a
complete searchable document.

## Retrieval schema

```sql
files(id, path UNIQUE, drive, dir, name, ext, size, mtime,
      lang, method, ocr_used, pages, chars, sha1, indexed_at)
fts USING fts5(name, path, content, tokens,
               tokenize="unicode61 remove_diacritics 2 tokenchars '_'")
chunks(id, file_id, chunk_index, content, embedding BLOB, dimensions)
```

Normalization combines lowercased words, Unicode diacritic folding, English
Snowball stems, Vietnamese maximum-matching compounds and editable abbreviation
expansions. FTS queries use the same normalization and BM25 ranking.

Complete content is split into overlapping 1,200-character chunks. FastEmbed's
`multilingual-e5-small` produces 384 float32 values stored as a SQLite BLOB.
Vector retrieval embeds the query with the E5 query prefix and ranks chunks by
cosine similarity. The current corpus size is intentionally served by a bounded
in-process scan, keeping the database portable and avoiding a second vector DB.

## Incremental consistency

Resume skips an unchanged path only when its size/mtime match, its extraction is
complete, required exhaustive methods are present, and it has vector chunks.
An optional, validated `include_paths` set narrows extraction to an exact list
selected by the caller; the full walk still drives deletion pruning.
The writer replaces each changed file's FTS row and chunks atomically. At the end
of a successful tree walk, records absent from the source tree are pruned from
all three tables. Job metrics expose files, OCR files, errors, incomplete files,
embedded chunks, removed files and elapsed time. A cooperative cancellation flag
is checked around extraction and embedding; cancellation drops the temporary
database and preserves the last atomically published corpus.

## Service boundary

Axum exposes health, job submission/status/cancellation and semantic search. A bounded Tokio
channel serializes jobs so concurrent requests cannot exhaust the host. Input
paths must canonicalize under configured read-only roots. Output is restricted
to a plain `.sqlite` filename under the output root; a temporary build is renamed
only after success, preserving the previous database on failure.

The image runs as an unprivileged UID, drops all capabilities, uses a read-only
root filesystem, mounts input read-only and writes only the output mount. The
Whisper and FastEmbed artifacts are downloaded and checksum-verified at image
build time, allowing the live engine to remain on the internal no-egress network.
