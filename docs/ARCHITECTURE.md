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
chunks_vec   USING vec0(embedding float[384] distance_metric=cosine)  -- exact
chunks_vec_q USING vec0(embedding bit[384])                           -- quantised
             -- or int8[384] distance_metric=cosine, whichever tier was built
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
`/corpus/*` surface cannot do. That is what the optional shadow indexes below
are, and why building one is a CLI subcommand rather than a route.

### The optional vec0 shadow indexes

`llm-index vector-index` adds a virtual table holding a derived copy of every
`chunks.embedding` BLOB. Semantic search then asks it for the k nearest
candidates instead of reading the whole `chunks` table. Both are off until an
operator builds them, per corpus, and nothing creates either implicitly.

A corpus can carry two, in two SLOTS with different promises:

| slot | table | tier | who reads it | promise |
|---|---|---|---|---|
| exact | `chunks_vec` | `float` | `mode=semantic` (the default) and `mode=semantic_fast` as a fallback | the scan's own top-k, same scores, same order |
| quantised | `chunks_vec_q` | `int8` or `bit` | `mode=semantic_fast` only | an approximation, measured below |

`rank_chunks` looks in the exact slot and nowhere else, so building a fast index
can never change what the default query path returns. `rank_chunks_fast` is the
only function that reads the quantised slot, and `/corpus/search` reports both
`path` (which one ran) and `exact` (whether the answer is the scan's).

Every path re-scores its candidates from `chunks.embedding` with the scan's own
`cosine_bytes` and orders them by the scan's own comparison, so a `score` is a
true cosine against the stored vector whichever index nominated the row. What
quantisation changes is which rows were scored at all — never what a score means.

#### The exact tier

It buys latency and only latency: `scan_latency_over_a_real_corpus` asserts the
same hits, in the same order, with the same scores to the bit, over the live
corpora rather than over fixtures. Measured on BOTH (release build, copies of the
live files, each pair back to back in one session on one workstation — a GPU
recovery job was running throughout, which is why the spreads are wide and why
the *ratio* within a row is worth more than any single number):

| vectors | corpus | scan reads | scan, warm passes | k-NN reads | `vec0` k-NN, warm passes |
|---|---|---|---|---|---|
| 869,267 | 5.35 GB | 5.35 GB | 13.9 - 22.9 s (4) | 1.34 GB | 1.32 - 5.57 s (4) |
| 2,684,125 | 15.6 GB | 15.6 GB | 45.6 - 74.9 s (9) | 4.12 GB | 3.9 - 16.1 s (9) |

**This is not the sub-second answer.** `vec0` 0.1.9 has no ANN structure: the
speedup is layout, not algorithm — vectors packed into contiguous chunk blobs
instead of one row per chunk, so a query reads only the vectors and skips
`content` entirely. It still visits every vector, and the cost is still the bytes.

The absolute scan numbers above are three to four times the 13.7 s in the earlier
table. Nothing regressed: that row was measured on a differently warmed page
cache. Each row here comes from one session so its two halves can be compared
with each other, which is the only comparison that section is making.

#### The quantised tier

`int8` stores `round(v_i * 127 / max_j|v_j|)` — a quarter of the bytes. The
divisor is per vector, which is what keeps it cosine-safe: cosine does not see a
positive scale, so the only error is the rounding, and the rounding is as small
as 8 bits allow because every vector's largest component lands on the rail.
(`sqlite-vec`'s own `vec_quantize_int8(v,'unit')` is not used: it is affine over
a fixed `[-1,1]`, and the components of a unit-norm 384-d embedding sit around
±0.05, so it would spend about 13 of its 256 codes on the whole corpus.)

`bit` stores one bit per dimension, set when `v_i > centre_i`, compared by
Hamming distance — a thirty-second of the bytes. The `centre` is the corpus mean,
measured once per build from a 50,000-vector sample, and it is the difference
between a working index and a broken one: text embeddings carry a large shared
mean direction, so a sign taken about the ORIGIN is the same in nearly every
vector and its bit is a constant. That is not a theory — the last column of the
first table below is the same tier built without a centre, and it is noise.

Measured with `quantised_recall_over_a_real_corpus` on copies of both live
corpora, 20 real `multilingual-e5-small` query embeddings of real prompts (EN and
VI), `limit=10`, recall@10 against the exact path's own top-10, release build,
same loaded workstation:

**2,684,125 vectors**, exact path 3,787 / 4,032 / 4,375 ms (`int8` session) and
3,746 / 3,933 / 4,965 ms (`bit` session), best / median / worst:

| candidate pool | `int8` recall@10 | `int8` best - median | `bit` recall@10 | `bit` best - median | `bit` UNCENTRED recall@10 |
|---|---|---|---|---|---|
| 10 | 0.9750 | 1,748 - 1,799 ms | 0.1250 | 143 - 155 ms | 0.0100 |
| 20 | **1.0000** | 1,726 - 1,788 ms | 0.1800 | 163 - 174 ms | 0.0150 |
| 50 | **1.0000** | 1,792 - 1,845 ms | 0.2750 | 196 - 207 ms | 0.0350 |
| 100 | **1.0000** | 1,889 - 1,946 ms | 0.3700 | 261 - 272 ms | 0.0550 |
| 200 | **1.0000** | 2,022 - 2,052 ms | 0.4450 | 399 - 412 ms | 0.0600 |
| 500 | **1.0000** | 2,380 - 2,437 ms | 0.5350 | 897 - 904 ms | 0.0800 |
| 1,000 | **1.0000** | 3,116 - 3,197 ms | 0.6150 | 1,750 - 1,801 ms | 0.0900 |

The `int8` column is the second of two runs of the same sweep, hours apart. The
first, with the GPU job at full tilt, produced the SAME recall in every row and
latencies of 1,655 - 3,024 ms at pool 10 and 3,257 - 8,448 ms at pool 1,000: the
shape held, the medians did not. Recall is a property of the index; a millisecond
on this machine is a property of the afternoon.

**869,267 vectors**, exact path 1,142 / 1,209 / 1,282 ms (829 / 1,223 / 1,287 in
the `bit` run and 916 / 1,021 / 1,085 in the shipped-path run — the same path,
three times, an hour apart, which is the size of the noise on every number here):

| candidate pool | `int8` recall@10 | `int8` best - median | `bit` recall@10 | `bit` best - median |
|---|---|---|---|---|
| 10 | 0.9800 | 533 - 545 ms | 0.1550 | 45 - 47 ms |
| 20 | **1.0000** | 542 - 555 ms | 0.2650 | 47 - 48 ms |
| 50 | **1.0000** | 560 - 585 ms | 0.3750 | 59 - 61 ms |
| 100 | **1.0000** | 576 - 595 ms | 0.4900 | 82 - 86 ms |
| 200 | **1.0000** | 622 - 644 ms | 0.5700 | 128 - 130 ms |
| 500 | **1.0000** | 787 - 815 ms | 0.7400 | 285 - 290 ms |
| 1,000 | **1.0000** | 1,072 - 1,095 ms | 0.7950 | 551 - 560 ms |

The rerank is the whole design and the numbers say why. Pattern A — serve the
quantised k-NN's own top-10 — is the `pool 10` row: `int8` gets 0.9750 on the
2.68 M corpus and 0.9800 on the 869 k one, below the bar on both. Pattern B —
nominate a wider pool and let the float re-score choose — is every row below it,
and `int8` reaches 1.0000 from a pool of 20 upward and stays there.
`crate::embedding::CANDIDATE_OVERSAMPLE` is therefore 10 (a pool of 100 for a
10-hit page, 200 for the 20-hit `/corpus/search` default): the largest multiplier
still on the flat part of the latency curve, since a wider pool costs one keyed
`chunks` row read per candidate and nothing in the k-NN itself.

That shipped configuration was then measured as itself — `rank_chunks_fast` at
`limit=10`, the same call `mode=semantic_fast` makes — rather than inferred from
the sweep: **recall@10 mean 1.0000, worst 1.0000** on both corpora, at
571 / 586 / 615 ms over 869,267 vectors and 1,860 / 1,945 / 8,325 ms over
2,684,125 (best / median / worst; the 8.3 s worst is the first query of the run,
which pays the cold read the other nineteen do not).

#### What the sub-second target actually costs

The shipped pattern is `int8` + 10x oversample + float rerank. Against the goal
of **< 1 s warm at 2.68 M vectors with recall@10 >= 0.95**:

| corpus | recall@10 (mean / worst query) | best | median | under 1 s? |
|---|---|---|---|---|
| 869,267 vectors | **1.0000** / **1.0000** | 571 ms | 586 ms | **yes** |
| 2,684,125 vectors | **1.0000** / **1.0000** | 1,860 ms | 1,945 ms | **no** |

**At 2.68 M the target is missed, and the miss is not close enough to explain
away.** Both halves were measured rather than assumed:

- **`int8` cannot go faster here.** Its 1.03 GB is a quarter of the exact tier's
  4.12 GB, yet it is only ~2.1x faster, because at this size the tier is bound by
  arithmetic rather than by bytes: `sqlite-vec` 0.1.9 has SIMD paths for
  `l2_sqr_float`/`l1_float` only, so `distance_cosine_int8` is a scalar loop over
  2,684,125 x 384 elements doing three multiplies and three adds each. That is
  also why its latency barely moves with the candidate pool (1,726 ms at 20,
  2,022 ms at 200) and why the two sessions agreed on ~1.7 s as the floor while
  disagreeing on everything above it.
- **`bit` is fast enough and nowhere near accurate enough.** It answers in 143 -
  399 ms — comfortably sub-second, three to twelve times faster than `int8` — and
  its recall@10 tops out at 0.6150 with a pool of 1,000. 384 bits is 32x
  compression; on this corpus that does not put the true top-10 into any pool the
  re-score can afford. Centring multiplies its recall by three to six and does not
  change the conclusion.

So on this hardware, with this library and this 384-d model, sub-second and
>= 0.95 recall are reachable **separately and not together at 2.68 M vectors**.
They are reachable together below roughly a million, where `int8` does both. What
would move the 2.68 M line is not a bigger candidate pool — it is an int8 distance
kernel with SIMD, or an embedding with more bits per vector to binarise, or an
actual ANN structure; none of those is a knob in `sqlite-vec` 0.1.9.

#### Cost of having them

| | 869,267 vectors | 2,684,125 vectors |
|---|---|---|
| `float` build from the stored BLOBs | 53.8 s, 0 skipped | 229.7 s, 0 skipped |
| `int8` build | 72.3 s, 0 skipped | 92.8 s, 0 skipped |
| `bit` build, incl. the centre sample | 31.9 s, 0 skipped | 154.0 s, 0 skipped |
| corpus growth, `float` | 5.35 -> 6.70 GB (+1.34 GB) | 15.6 -> 19.8 GB (+4.19 GB) |
| corpus growth, `float` + `int8` | -> 7.06 GB (+1.71 GB) | -> 20.9 GB (+5.29 GB) |
| `bit` payload | 48 B x 869,267 = 42 MB | 48 B x 2,684,125 = 129 MB |
| documents re-embedded | none | none |

Read the build row for its order of magnitude and not for its ordering: each was
the next command in one session, so the first build of each corpus paid the cold
read and the rest did not, and a GPU job was competing throughout. That is why
`float` at 2.68 M (first, cold) is slower than `bit` there (third, warm) despite
writing 32 times the bytes, and why W1 recorded 84.6 s for the same `float` build
in its own session. What is stable across all of them: every build is minutes,
and none re-embeds anything. A build reads `chunks` and writes only its own
table, so `files`, `fts`, `chunks` and `vision` are read-only to it and the worst
an interrupted one costs is the time it had spent.
Reading `chunks.embedding` means walking past `chunks.content`, which is why a
build costs minutes rather than the seconds its output would suggest — and why the
`bit` centre is measured from a spread sample rather than from a second full pass.

A build commits every 50,000 vectors and writes its `meta` marker last, so an
interrupted one leaves a part-filled table that no query will use and `--rebuild`
starts over. That batching is what keeps the journal at megabytes: a
single-transaction rebuild would have to journal the pre-image of every page the
dropped index freed and the new one reuses, i.e. a journal the size of the index.
A `--rebuild` over the existing 2.68 M-vector float index took 207.8 s with a peak
rollback journal of 4.1 MB.

**Staleness.** Index jobs maintain every index they find as they write, but only
jobs run by a build that has this feature: every earlier release writes `chunks`
rows underneath them, and so does anything editing the corpus directly.
`vec0::usable` therefore re-proves an index on every query against two witnesses
recorded in its `meta` marker — `meta.last_job_started_at` (moved by any job,
whether or not it maintained the index) and the `chunks` row count (moved by an
edit that bypassed the pipeline). A `bit` index is checked against a third: its
`centre` must still be there and still be the right width, because an uncentred
bit index is not a degraded one but a random one. Any mismatch falls back — to the
exact path for `semantic`, and to the exact path with the reason in `index_note`
for `semantic_fast` — and none of the checks is expensive, since all are indexed
lookups against a k-NN that reads gigabytes.

**Old readers.** The virtual tables are inert to a build without `sqlite-vec`.
SQLite records a virtual table's declaration in `sqlite_master` and instantiates
its module only when a statement names the table, so an older `llm-index` reads,
writes, migrates and prunes a corpus that has both indexes, and fails only on the
two queries it would never issue (`no such module: vec0`). `tests/old_binary.rs`
asserts exactly that, for both slots, by cancelling this process' module
registration and then running the older build's statements.

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
embedded chunks, removed files, capped files, hashed and unhashable files, and elapsed time. A cooperative cancellation flag
is checked around extraction and embedding; a cancelled job commits what it
finished and leaves it in the destination corpus, so resubmitting with `resume`
continues from there.

### sha1 backfill lane

`files.sha1` is written by the forward path, i.e. only for a file the job
actually indexes, so rows finished before `hash` was enabled never acquire one:
resume does not re-extract a finished row, and nothing else writes that column.
The `hash_backfill` knob (default off, and requiring `hash` as well) adds a lane
that hashes exactly the rows the resume predicate declined.

Classification happens INSIDE the `records.retain` block that implements that
predicate, because the lane's population is by definition "the rows that block
declined"; deriving it separately would mean re-implementing the predicate and
eventually disagreeing with it. That block runs only under
`resume && !embed_model_changed`, which is the correct and complete condition:
both other shapes send every walked file down the forward path, which hashes it.
Capped rows are excluded — they may still be indexed once the cap lifts, and the
exclusion is what keeps `capped` a strict subset of `skipped`.

The writer is `IndexStore::set_sha1`, a bare `UPDATE files SET sha1 WHERE path`,
NOT the `INSERT OR REPLACE` the indexing path uses: that would restate `method`,
`chars`, `pages`, `indexed_at`, `attempts`, `last_attempt_at` and `elapsed_ms`
from a `ProcessedFile` the lane never built, and those are precisely the columns
the resume predicate and the attempt cap read. It rides the store's open
transaction and the same `commit_batch` checkpoint indexed files use, so a killed
backfill keeps what it hashed and the next run owes only the rest.

The lane runs before the indexing pass, sequentially, on the thread that owns the
store, and observes the cancellation flag. A 1 GiB ceiling applies to it alone,
mirroring `SHA1_MAX_BYTES` in the consuming drives-analytics app; the forward
hash path is unchanged and has no ceiling, so the two agree below it only. In
practice `max_bytes` binds first — a file over it extracts to `name-only`, an
incomplete row the lane never claims — so the ceiling is only load-bearing where
`max_bytes` has been raised past it or `ocr: exhaustive` bypasses the size
cut-off. Backfilled hashes reach SQLite only — `manifest.jsonl` and `catalog.csv`
are per-indexed-file exports and a hash-only row produces no line in either.

A claimed file that will not open or read is counted in `hash_failed`, and named
in the log up to a sample, rather than dropped. It writes no row: the stored row
is a successful extraction that is still true, and a hash that could not be taken
says nothing about it. Its own counter rather than `errors`, which means "this run
processed a file and the processing failed" — most of those are findable as
`error:` rows, though not all (the keep-on-failure branch counts the failure and
keeps the old complete row rather than replacing it, so that one has no `error:`
row behind it). A hash miss is neither: nothing was processed at all. The
accounting closes on it — for a lane that ran to completion,
`hashed + hash_failed` is the owed count it announced — and because those rows
never acquire a `sha1` the lane re-claims them on every armed run, which makes
the counter the convergence signal for the whole exercise.

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
