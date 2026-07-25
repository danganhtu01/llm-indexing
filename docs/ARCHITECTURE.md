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
workers extract content in parallel and hand each finished file to a single
writer over a bounded channel; the writer embeds it, stores it and commits in
batches. Nothing is buffered until the end, so peak memory is the channel depth
rather than the whole corpus, and a killed run keeps everything already
committed. Archives are unpacked through `bsdtar` under the built-in `C.UTF-8`
locale only after safe relative-path validation, preserving Vietnamese/Unicode
entry names without installing mutable locale data. Archive traversal retains a
four-level recursion limit and 10,000-entry bound.

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
complete searchable document. A method beginning `excluded:` is the opposite: a
decision not to process the file — an Office lock file, an extension no extractor
in this build handles — so it is terminal and resume never revisits it.

## Retrieval schema

```sql
files(id, path UNIQUE, drive, dir, name, ext, size, mtime,
      lang, method, ocr_used, pages, chars, sha1, indexed_at,
      attempts, last_attempt_at, elapsed_ms)
fts USING fts5(name, path, content, tokens,
               tokenize="unicode61 remove_diacritics 2 tokenchars '_'")
chunks(id, file_id, chunk_index, content, embedding BLOB, dimensions, model)
vision(file_id, mode, width, height, phash, exif_json, quality_json,
       objects_json, tags_json, caption,
       embedding BLOB, embedding_model, dimensions, frames, elapsed_ms, error)
meta(key, value)

-- OPTIONAL, created only by `llm-index vector-index`:
chunks_vec USING vec0(embedding float[384] distance_metric=cosine)
```

Normalization combines lowercased words, Unicode diacritic folding, English
Snowball stems, Vietnamese maximum-matching compounds and editable abbreviation
expansions. FTS queries use the same normalization and BM25 ranking.

Complete content is split into overlapping 1,200-character chunks. FastEmbed's
`multilingual-e5-small` produces 384 float32 values (a 1,536-byte
little-endian BLOB) and the model that produced them is stamped on every row.
Vector retrieval embeds the query with the E5 query prefix and ranks chunks by
cosine similarity, over `GET /corpus/search?mode=semantic` and the
`vector-search` CLI subcommand alike. `vision.embedding` is a CLIP *image*
vector and is deliberately NOT part of that search: it lives in a different
embedding space from the e5 text vectors, and a cosine between the two spaces
is a number with no meaning. Ranking therefore compares only rows whose
`chunks.model` matches the model the query was embedded with, and counts the
rest rather than mixing them in.

### Cost of the vector scan

The ranking is an exhaustive streaming scan — every stored vector is scored,
so the top-k is exact — with a bounded top-k heap, no per-row allocation, and
`content` fetched only for the winners. There is no ANN index, and the corpus
stays a single portable file with no second vector database. Measured with
`scan_latency_over_a_real_corpus` (release build, Windows workstation with a
GPU describe job running, so the spread is real machine load):

| vectors | corpus file | cold read | warm scan |
|---|---|---|---|
| 100,000 | 415 MB | 0.71 s | 0.24 s |
| 1,000,000 | 4.17 GB | 6.20 s | 3.89 s |
| 2,684,125 (live) | 15.6 GB | 54.3 s | 13.7 s |

The pass is bound by SQLite page reads, not arithmetic: scoring a million
vectors takes ~0.2 s single-threaded against ~2.3 s to read them, which is why
the scan is deliberately not parallelised — rayon would buy a few percent and
cost contention with the extraction pool a concurrent index job is using.

`sqlite-vec` was measured, not assumed: 0.1.9 builds cleanly against this
crate's rusqlite 0.32 (bundled SQLite 3.46.0) and its `vec_distance_cosine()`
reads the existing BLOBs with no schema change — but at 4.09-6.81 s per million
it is the same I/O-bound pass, and it agreed with this scan's ranking to the
last decimal on a million live vectors. Its `vec0` half is a corpus-format
change: it needs shadow tables WRITTEN into the corpus, which the read-only
`/corpus/*` surface cannot do. That is what the optional shadow index below is,
and why building one is a CLI subcommand rather than a route.

### The optional vec0 shadow index

`llm-index vector-index` adds a `chunks_vec` virtual table holding a second copy
of every `chunks.embedding` BLOB. Semantic search then asks it for the k nearest
candidates instead of reading the whole `chunks` table. It is off until an
operator builds it, per corpus, and nothing creates it implicitly.

It buys latency and only latency. The k-NN nominates candidates; their scores are
recomputed from `chunks.embedding` with the scan's own `cosine_bytes` and ordered
by the scan's own comparison, so a `score` is the same number and the top-k is the
same list whichever path ran. `scan_latency_over_a_real_corpus` asserts that over
the live corpus, not just over fixtures, and `/corpus/search` reports which path
served each request (`path: "scan" | "vec0"`).

Measured on BOTH live corpora with `scan_latency_over_a_real_corpus` (release
build, copies of the live files, each pair back to back in one session on one
workstation — a GPU recovery job was running throughout, which is why the spreads
are wide and why the *ratio* within a row is worth more than any single number):

| vectors | corpus | scan reads | scan, warm passes | k-NN reads | `vec0` k-NN, warm passes |
|---|---|---|---|---|---|
| 869,267 | 5.35 GB | 5.35 GB | 13.9 - 22.9 s (4) | 1.34 GB | 1.32 - 5.57 s (4) |
| 2,684,125 | 15.6 GB | 15.6 GB | 45.6 - 74.9 s (9) | 4.12 GB | 3.9 - 16.1 s (9) |

**This is not the sub-second answer.** `vec0` 0.1.9 has no ANN structure: the
speedup is layout, not algorithm — vectors packed into contiguous chunk blobs
instead of one row per chunk, so a query reads only the vectors and skips
`content` entirely. It still visits every vector, and the cost is still the bytes.
Best observed is 1.32 s at 869 k vectors and 3.9 s at 2.68 M, so on this hardware
the index buys roughly an order of magnitude and crosses into interactive
territory somewhere below a million vectors — not at the scale of the largest
live corpus. Sub-second at 2.68 M would need quantised vectors (`int8`/`bit`),
which change which rows come back: a different feature with a different bar, and
deliberately not this one.

The absolute scan numbers above are three to four times the 13.7 s in the earlier
table. Nothing regressed: that row was measured on a differently warmed page
cache. Each row here comes from one session so its two halves can be compared
with each other, which is the only comparison this section is making.

Cost of having one:

| | 869,267 vectors | 2,684,125 vectors |
|---|---|---|
| first build from the stored BLOBs | 84.4 s, 0 skipped | 84.6 s, 0 skipped |
| corpus growth | 5.35 -> 6.70 GB (+1.34 GB) | 15.6 -> 19.8 GB (+4.19 GB) |
| documents re-embedded | none | none |

The growth is exactly one `f32` copy per vector; a build reads `chunks` and
writes only `chunks_vec`, so nothing is re-embedded and no model is loaded. A
`--rebuild` over the existing 2.68 M-vector index took 207.8 s — slower than the
first build because it reuses the pages the dropped index freed and overwritten
pages are journaled — with a peak rollback journal of 4.1 MB.

A build commits every 50,000 vectors and writes its `meta.vec0_index` marker
last, so an interrupted one leaves a part-filled table that no query will use and
`--rebuild` starts over. That batching is what keeps the journal at megabytes: a
single-transaction rebuild would have to journal the pre-image of every page the
dropped index freed and the new one reuses, i.e. a journal the size of the index.
`files`, `fts`, `chunks` and `vision` are read-only to a build.

**Staleness.** Index jobs maintain the index as they write, but only jobs run by
a build that has this feature: every earlier release writes `chunks` rows
underneath it, and so does anything editing the corpus directly. `vec0::usable`
therefore re-proves the index on every query against two witnesses recorded in
`meta.vec0_index` — `meta.last_job_started_at` (moved by any job, whether or not
it maintained the index) and the `chunks` row count (moved by an edit that
bypassed the pipeline). Either mismatch falls back to the scan and puts the
reason in `index_note`; neither is expensive, since both are indexed lookups
against a k-NN that reads gigabytes.

**Old readers.** The virtual table is inert to a build without `sqlite-vec`.
SQLite records a virtual table's declaration in `sqlite_master` and instantiates
its module only when a statement names the table, so an older `llm-index` reads,
writes, migrates and prunes a corpus that has an index, and fails only on the one
query it would never issue (`no such module: vec0`). `tests/old_binary.rs`
asserts exactly that by cancelling this process' module registration and then
running the older build's statements.

## Incremental consistency

Resume skips an unchanged path only when its size/mtime match, its extraction is
complete, required exhaustive methods are present, and it has vector chunks.
An unfinished row is also skipped once it has failed three times, since
re-extracting a file this build cannot read costs the same on every resume and
produces the same row. `files.attempts` counts those failures and
`files.last_attempt_at`/`files.elapsed_ms` record when the last one ran and what
it cost. Changed bytes, an available upgrade, a changed embedding model, a moved
extraction-capability revision (`meta.extractor_revision`, derived from the
extension tables) and an explicit `retry_errors` each reopen a capped row.
An optional, validated `include_paths` set narrows extraction to an exact list
selected by the caller; the full walk still drives deletion pruning.
The writer replaces each changed file's FTS row and chunks atomically. At the end
of a successful tree walk, records absent from the source tree are pruned from
all three tables — except when the walk found nothing at all, which is far more
often an unmounted or mistyped root than a tree whose every file was deleted,
and pruning is no longer reversible now that it lands in the published corpus.
Rebuilding from empty is what `overwrite` is for. Job metrics expose files, OCR files, errors, incomplete files,
embedded chunks, removed files, capped files and elapsed time. A cooperative cancellation flag
is checked around extraction and embedding; a cancelled job commits what it
finished and leaves it in the destination corpus, so resubmitting with `resume`
continues from there.

### Durability

The job writes into the destination database itself, committing every 100 files
or 30 seconds, whichever comes first. A crash, a kill or a cancellation
therefore costs at most the current batch instead of the entire run, and resume
sees that partial corpus: a file row without vector chunks, or with a
`-partial`/`error:`/`name-only` method, is redone. The batch bounds are the
whole tradeoff — smaller means more fsyncs, larger means more extraction and OCR
work thrown away by a kill.

Each file goes in under its own savepoint, so a file whose rows fail part-way
through is rolled back rather than left in the transaction for the next commit
to publish. Without that, a chunk insert failing after the `files` and `fts`
rows landed would commit a file holding some of its vectors — and resume treats
any file with at least one chunk as done, so nothing would ever revisit it. A
rolled-back file is simply absent, which is what makes resume redo it. If a
per-file rollback itself fails the store is poisoned and the open transaction is
discarded whole rather than committed blind; earlier batches are unaffected.

Journal mode stays on SQLite's rollback-journal default. WAL would serve
concurrent readers better but leaves `-wal`/`-shm` sidecars, and the corpus is
copied and served as a bare single file. The writer carries a 30-second busy timeout so a reader
can never abort its commit; the read-only `/corpus` connections carry a much
shorter one (3 seconds), because a consumer polling during a long index wants a
prompt, honest "busy, retry" rather than a stall lasting the writer's whole
commit window. A reader that loses that race is reported as busy, never as
damaged: a batch commit spilling its page cache escalates to an EXCLUSIVE lock,
and treating that as corruption would flag every read taken during ordinary
indexing. The cost of that choice is that a writer killed mid-transaction leaves a
hot rollback journal, which a read-only connection cannot replay — SQLite
refuses the database outright. The read surface therefore recovers one itself,
with a brief read-write open, before serving reads.

## Service boundary

Axum exposes health, job submission/status/cancellation and semantic search. A bounded Tokio
channel serializes jobs so concurrent requests cannot exhaust the host. Input
paths must canonicalize under configured read-only roots. Output is restricted
to a plain `.sqlite` filename under the output root and is written in place. An
existing corpus is refused unless the job sets `resume` (continue into it) or
`overwrite` (delete it and start clean); `resume` wins when both are set.
Because there is no staged build to swap in, an interrupted `overwrite` leaves a
partial new corpus rather than the superseded one — which makes *when* the
deletion happens the whole safety contract. It is deferred to the last moment
before the store opens, after the config, the vision models and the embedding
model have all loaded, so that the predictable operator errors fail with the old
corpus still intact. The residual window is `IndexStore::open` itself: a failure
between the delete and the first write (unwritable output directory, a schema
that will not create) still costs the previous corpus.

Consumer apps used to open `corpus.sqlite` directly to render a directory tree
or preview a document. `GET /corpus/tree`, `GET /corpus/documents/{id}/text`,
`GET /corpus/status` and `GET /corpus/search` (see `docs/HTTP_API.md`) serve
that read-only join instead, so no consumer needs to decode the SQLite schema
itself. `/corpus/tree`
walks one named allowed input root (validated against the same allowed-roots
model as `/index`) and joins it against the published database by each file's
exact absolute path — precise where a by-name join could collide across
directories. All four routes degrade to an empty/zeroed result when the corpus
database hasn't been written yet, but only then: a database that exists and
cannot be read answers `503`, never a zero, because a consumer handed `0` over a
corpus holding thousands of rows cannot tell that from an empty one. Since jobs
write in place, `/corpus/status` also reports `writing` while a job targets that
output — the successor to the guarantee the old atomic publication gave for
free, that a visible corpus was a finished one.

The image runs as an unprivileged UID, drops all capabilities, uses a read-only
root filesystem, mounts input read-only and writes only the output mount. The
Whisper and FastEmbed artifacts are downloaded and checksum-verified at image
build time, allowing the live engine to remain on the internal no-egress network.
