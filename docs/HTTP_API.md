# HTTP API
The container listens on TCP 9801. All API requests and responses are JSON.

## Submit token — gating the job-mutating routes

By default every route below is open to anything that can reach the port. That
is the right default for a standalone container, and the wrong one for an
app-managed engine: the managing app is supposed to hold absolute control over
the job surface, and an engine whose loopback port accepts direct
`POST /index` calls leaves that app honestly reporting an "engine-native job"
it can neither pause nor cancel.

`serve --submit-token <secret>` (env fallback `LLM_SUBMIT_TOKEN`; the flag
wins, and an explicitly empty value is refused at startup) closes that hole.
When set, every **job-mutating** route requires the same secret in an
`X-Submit-Token` header:

- `POST /index`
- `POST /jobs/{id}/cancel`
- `POST /runtime` and `POST /jobs/{id}/runtime`

A missing or wrong token answers `401` before the handler runs, so a refusal
has no side effects — no job row, no cancellation flag, no persisted envelope:

```json
{"status": "error", "error": "missing or invalid X-Submit-Token header; job-mutating routes on this server require the submit token", "header": "X-Submit-Token"}
```

Every read-only route — `GET /health`, `GET /settings`, the job and runtime
GETs, and the whole `/corpus/*` read surface including `/corpus/search` —
stays open, so an app's search proxy, monitor panels and read tools work
without the token. That split is what makes the token an app-held **write
credential** rather than a service password. The gate keys on the request
method: every mutation this service exposes is a POST and every GET is
read-only by construction, so a job-mutating route added later is gated by
default instead of depending on someone remembering to enrol it.

The comparison is constant-time, and without the flag the gate is not even
installed — an ungated `serve` behaves exactly as it always has.

The flag and header names are shared verbatim with the vlm-indexing engine's
identical gate (whose env fallback is `VLM_SUBMIT_TOKEN`), so an app managing
both engines configures and calls them uniformly.

## `GET /health`

Returns service version/readiness and whether a job is queued or running.

## `GET /settings`

Read-only capability discovery (`src/service.rs::build_settings`) — the
contract consumer apps (`ff-lc-app`, `da-academic`, `drives-analytics`) render
their OCR/vision settings UI from, so nothing is hardcoded client-side: OCR
bounds/installed languages and which vision tiers/sub-models this *specific*
running process can actually serve (capped by `serve --vision-max`, gated on
model files being present and hash-verified). Purely additive; touches no job
state. Every range/enum here is the same single source of truth
`ocr_opts`/`vision_opts` validation uses (`src/settings.rs`'s `OCR_DPI_RANGE` /
`OCR_PSM_RANGE` / `OCR_MAX_PAGES_RANGE` / `DETECTORS` / `TAGGERS` /
`CAPTIONERS` consts), so this endpoint and submit validation can never drift
apart.

```json
{
  "version": "0.4.0",
  "ocr": {
    "modes": ["auto", "on", "off", "exhaustive"],
    "langs_installed": ["eng", "vie"],
    "dpi": {"min": 150, "max": 1200, "default": 300},
    "psm": {"values": ["0","1","2","3","4","5","6","7","8","9","10","11","12","13"], "default": "3"},
    "preprocess_default": true,
    "max_pages": {"min": 1, "max": 500, "default": 20}
  },
  "vision": {
    "max_tier": "tags",
    "tiers_available": ["meta", "tags"],
    "detectors": [{"id": "nano", "present": true}],
    "taggers": [{"id": "clip", "present": true}],
    "captioners": [{"id": "florence2", "present": false}],
    "faces": [{"id": "yunet-sface", "present": false}],
    "defaults": {
      "detector_conf": 0.5,
      "tag_threshold": 0.22,
      "tag_top_k": 8,
      "faces": "off",
      "face_score": 0.9,
      "max_faces": 20,
      "max_frames": 12,
      "timeout_secs": 60
    }
  },
  "workers": {"default": 8, "max": 64}
}
```

- `version` is the running build's `CARGO_PKG_VERSION`.
- `ocr.langs_installed` enumerates the bundled `<data_dir>/tessdata` directory
  unioned with `tesseract --list-langs`'s own system-pack report — the exact
  resolution `TesseractOcr`/`ocr_opts.langs` validation uses
  (`installed_tessdata_langs`), never a hardcoded list.
- `ocr.psm.values` is every accepted PSM string, `"0"` through `"13"`.
- `vision.max_tier` is this process's `serve --vision-max` cap (`INDEX_VISION_MAX`
  env fallback); `tiers_available` is further filtered to tiers whose model
  files are present under `<data_dir>/vision` **and** pass the pinned SHA-256
  check (`available_tiers`/`corrupt_models`) — so an entry here is a real
  guarantee the tier will run, not just that the tier name is known.
- `vision.detectors`/`taggers`/`captioners`/`faces` list every selectable
  sub-model id (`ocr_opts`/`vision_opts`' accepted enum values, minus `off`)
  with a `present` flag backed by the same model-file existence + hash check
  (`detector_present`/`tagger_present`/`captioner_present`/`faces_present`). In
  v1 each category has exactly one model, so all its ids currently share one
  flag.
- `vision.faces` is the **opt-in, privacy-sensitive** face pair (YuNet + SFace).
  It differs from the other three in two ways worth reading carefully:
  `defaults.faces` is `"off"` on every server (nothing enables it by omission),
  and `present: false` means the capability is **absent**, not that a job will
  fail — a job that asks for faces on a box that has not run
  `fetch-data --faces` runs the rest of its tier and writes no faces. It also
  never appears in `tiers_available`: faces is a sub-model of the `tags` tier,
  not a tier, so an unstaged pair can never gate a tier off.
- `vision.defaults` and `ocr.dpi.default`/`psm.default`/`preprocess_default`/
  `max_pages.default` are read live from the loaded `Config` — the same
  `OcrSettings::from_config`/`VisionSettings::from_config` base every
  `ocr_opts`/`vision_opts` merge starts from — so `/settings` always reflects
  the actual YAML config in effect, not a static default.
- `workers.max` is the fixed ceiling (`config::MAX_WORKERS`, 64) applied to
  `Config::finalize`'s clamp; `workers.default` is this server's configured
  worker count.
- Runs the tessdata/hash probes on a blocking worker thread
  (`tokio::task::spawn_blocking`), never the async executor, since they exec
  `tesseract --list-langs` and hash up to ~100 MB of model files.

## `POST /index`

Queues one job and returns `202 Accepted` with an `id`. On a server started
with `--submit-token` this route requires the `X-Submit-Token` header — see
[Submit token](#submit-token--gating-the-job-mutating-routes).

```json
{
  "paths": ["/input"],
  "output": "corpus.sqlite",
  "ocr": "exhaustive",
  "ocr_langs": "vie+eng",
  "workers": 4,
  "include_paths": ["Customers/new.pdf", "Meetings/changed.mp4"],
  "resume": true,
  "overwrite": true,
  "retry_errors": false,
  "vision": "tags",
  "ocr_opts": {
    "dpi": 300,
    "psm": "3",
    "preprocess": true,
    "max_pages": 20,
    "langs": "vie+eng+rus"
  },
  "vision_opts": {
    "detector": "nano",
    "detector_conf": 0.5,
    "tagger": "clip",
    "tag_threshold": 0.22,
    "tag_top_k": 8,
    "captioner": "florence2",
    "faces": "off",
    "face_score": 0.9,
    "max_faces": 20,
    "max_frames": 12,
    "timeout_secs": 60
  }
}
```

`ocr` accepts `auto`, `on`, `off` or `exhaustive`. Omitted paths use the service
default. The queue returns 429 when full. Invalid JSON, out-of-root paths,
non-directory inputs and unsafe output names fail without publishing a file.
When present, `include_paths` must contain existing relative files confined
under an input root. Only those files are extracted; source deletion pruning
still uses the complete mounted tree.

`resume` and `overwrite` decide what may happen to an existing `output`, which
the job writes into directly (there is no staged copy renamed in at the end):

- neither set, and the database exists → the job fails with `output already
  exists; set resume or overwrite`, touching nothing;
- `resume` → the job opens the existing database and indexes only files that are
  new, changed, incomplete or missing vectors. This is also how a crashed or
  cancelled job is continued, since its committed files are already there;
- `overwrite` → the database is **deleted and rebuilt from empty**. The deletion
  happens only after the job has loaded its config and every model it needs, so
  the ordinary failures — bad config, missing or corrupt vision models, an
  embedding model that cannot be fetched — leave the existing corpus untouched.
  Once indexing starts there is no going back: an overwrite interrupted after
  that leaves a partial new corpus, not the one it replaced, so keep a copy
  first if the previous corpus still matters;
- both set → `resume` wins.

`retry_errors` (default `false`) applies only to a `resume`. A row that has
failed three times without finishing is left alone, because re-extracting a file
the engine cannot read costs the same on every resume and produces the same row;
setting this reopens those rows for one run. It changes which rows are attempted,
never how one is processed. Leave it off unless the reason for the failures has
been fixed OUTSIDE the engine (a drive that was not mounted, a dependency that
was not installed, files that have since been repaired) — a fix inside the engine
moves the extraction-capability revision and reopens them by itself. The
completed job reports how many rows were held back as `capped`.

A job that fails part-way leaves what it had committed, so a database now exists
where a failed job used to publish nothing — including the empty one a job that
failed before its first batch leaves behind. An `error` result carries `output`
and a `partial_corpus` note saying so. Resubmit with `resume` (continue) or
`overwrite` (start clean); submitting with neither is refused as above.

Readers may hold the corpus open while a job writes into it, and a live corpus
can be mid-run rather than complete. A read that arrives while the writer holds
its commit lock waits briefly and then answers `503` with
`{"error":"corpus database busy","retryable":true}` — retry it. That is
deliberately distinct from an unreadable corpus (below): one is a healthy
database under contention, the other is a fault.

`vision` requests a tier (`off`|`meta`|`tags`|`captions`, default `off`),
capped by the server's `serve --vision-max`; see
[`docs/VISION.md`](VISION.md) for tier semantics.

### `ocr_opts` / `vision_opts` — per-job quality overrides

Both fields are optional; every sub-field is independently optional (`None` ⇒
keep the service config's value). A submitted field is validated and, when
valid, wins over the service config for that job only — the service config
still wins over the built-in default. `OcrSettings::merge` /
`VisionSettings::merge` (`src/settings.rs`) is the single, unit-tested merge
path shared by this HTTP surface, the native `index --ocr-*`/`--vision-*` CLI
flags, and the `ocr:`/`vision:` sections of the YAML config — one struct pair,
three entry points. Absent `ocr_opts`/`vision_opts` reproduce today's behavior
byte-for-byte.

`vision_opts` only takes effect when `vision` (or the server's default) resolves
to a tier above `off`, and every numeric knob stays capped by `--vision-max`.

Validation at submit returns `400` with a field-specific message on the first
violation:

| Field | Type | Bounds | Notes |
|---|---|---|---|
| `ocr_opts.dpi` | integer | `150..=1200` | PDF page rasterization DPI. |
| `ocr_opts.psm` | string | `"0".."13"` | Tesseract page-segmentation mode, engine-style string. |
| `ocr_opts.preprocess` | bool | — | ImageMagick grayscale/deskew/contrast pre-pass. |
| `ocr_opts.max_pages` | integer | `1..=500` | Max PDF pages OCR'd per file. |
| `ocr_opts.langs` | string | must name only installed tesseract languages | `"vie+eng+rus"` style; validated against the same bundled-`tessdata` ∪ system-pack resolution `TesseractOcr` uses (see `GET /settings`). Wins over the legacy top-level `ocr_langs`. |
| `vision_opts.detector` | string | `off`\|`nano` | Object detector selection. |
| `vision_opts.detector_conf` | float | `0.05..=0.95` | Minimum detector confidence kept. |
| `vision_opts.tagger` | string | `off`\|`clip` | Zero-shot tagger selection. |
| `vision_opts.tag_threshold` | float | `0.0..=1.0` | Minimum CLIP tag score kept. |
| `vision_opts.tag_top_k` | integer | `1..=32` | Max tags kept per file. |
| `vision_opts.captioner` | string | `off`\|`florence2` | Captioner selection. |
| `vision_opts.faces` | string | `off`\|`yunet-sface` | Face detection + embedding. **Default `off`**; see the privacy note below. |
| `vision_opts.face_score` | float | `0.05..=0.99` | Minimum face-detection score kept. |
| `vision_opts.max_faces` | integer | `1..=200` | Max faces kept per file. |
| `vision_opts.max_frames` | integer | `1..=64` | Max video keyframes analysed. |
| `vision_opts.timeout_secs` | integer | `5..=1800` | Per-file vision timeout (seconds). |

`vision_opts.faces` is the one knob here that produces data about **people**
rather than files, so it behaves differently on purpose:

- it is `off` by default in config, in the merge, and on the CLI — no request
  turns it on by omission;
- its models are staged only by an explicit `llm-index fetch-data --faces`
  (`fetch-data --vision` does **not** fetch them);
- if the pair is not staged the capability is absent and the job succeeds
  without faces — an unavailable privacy feature never fails a job. A pair that
  IS staged but fails its pinned SHA-256 does fail the job, since bytes nobody
  vouched for must not compute claims about a person's identity;
- results land only in the corpus's own `faces` table
  (`file_id, face_index, x, y, width, height, quality, embedding, dimensions,
  model, frame`). They are never rendered into `fts.content`, sidecars,
  manifests, or the job summary — which reports only a `faces` COUNT — and this
  engine never transmits them anywhere.

Unknown top-level fields anywhere in the job body remain permissively ignored
(existing forward-compat serde posture) — only the fields above are validated.

## `GET /jobs/{id}`

Returns `queued`, `running`, `cancelling`, `cancelled`, `complete` or `error`.
Running jobs include live `processed` and `total` file counters. A completed job includes the
database path, file/OCR/error/incomplete counts, the `capped` count of rows resume
declined because they have failed too often, the `hashed` and `hash_failed`
counts from the sha1 backfill lane (both 0 unless `hash_backfill` is on),
embedded chunk count, the `faces` count of faces stored (0 unless the opt-in
faces sub-tier ran), removed source count, elapsed time and OCR languages.

`processed`/`total` include the backfill lane's rows, so an armed resume reports a
LARGER `total` than the same resume unarmed — the run genuinely has that much more
to do, and because both counters move together a rate or ETA derived from them
stays honest. Lane rows count in `processed` and are NOT counted in `skipped`;
`capped` remains a strict subset of `skipped`.

`hash_failed` counts rows the lane claimed whose file would not open or read
(locked, in use, or unreadable by the account running the job). Those rows are
left unchanged and still carry no `sha1`, so the lane claims them again on every
armed run — the counter is how an operator sees whether the backfill has
converged. It is deliberately NOT folded into `errors`, which counts rows written
with an `error:` method; a hash miss writes no row at all. For a run that
completed the lane, `hashed + hash_failed` equals the owed count the run
announced, so nothing the lane touched is left unattributed.

## `POST /jobs/{id}/cancel`

Requests cooperative cancellation of a queued or running job. The engine stops
before the next extraction/embedding boundary and commits the files it had
already finished, which stay in the published corpus. Poll the job until its
state becomes `cancelled`; that result carries the `output` name and reports the
partial corpus as retained. Resubmit with `"resume": true` to continue from it.
A job cancelled before it started leaves the output untouched. Gated by the
[submit token](#submit-token--gating-the-job-mutating-routes) when one is
configured, like every job-mutating route.

## Runtime stage tuning

Concurrency knobs that can be changed **while a job is running**. Values are
integers; out-of-range values are **clamped, not rejected**, and the response
always reports what actually landed. Both POSTs here mutate how jobs run, so
they sit behind the [submit token](#submit-token--gating-the-job-mutating-routes)
when one is configured; the GETs stay open.

### `GET /runtime`

```json
{"stages": {"extract": {"value": 8, "min": 1, "max": 64,
                        "live": true, "unit": "threads"}}}
```

| stage | unit | live | what it controls |
| --- | --- | --- | --- |
| `extract` | threads | yes | Extraction workers admitted concurrently. The rayon pool is built once at the 64-worker ceiling and admission is gated per file, so raising or lowering this retunes a job already in flight. |
| `embed` | instances | yes | Live `Embedder` instances. `fastembed`'s `embed` takes `&mut self`, so N concurrent embeds need N models — **each ~448 MB resident**. Default 2, grown lazily; shrinking drops instances as they are returned. |
| `ocr` | threads | **no** | `OMP_THREAD_LIMIT` for each tesseract spawn. `applies: next-file` — the value is resolved when the process is spawned, once per file, so it cannot reach a scan already being recognised. Defaults to the CPU count, which is what OpenMP would pick anyway. |

`live: true` means a change reaches work already in flight. When `live` is
`false` the stage carries an `applies` field naming the boundary instead. This
flag is meant to be trusted, which is why `ocr` reports `false`: turning it down
while a 900-page PDF is being OCR'd changes nothing until the next file, and a
`true` here would have a client render a control that looks like it did
something it did not.

`extract`, `embed` and `ocr` are the stage names shared with the other engine,
and they are the complete set — this engine advertises no extras. ONNX intra-op
width is a **config-only** setting (`embed_intra_threads` in the YAML config),
deliberately not a runtime stage: ort bakes the thread count into a `Session` at
construction, so it could never be live, and a name only this engine knows would
render in the app's Settings UI as a control whose save is rejected — taking
every other edit in the same request down with it.

### `POST /runtime`

Body `{"<stage>": <int>, ...}`. Sets the **process-wide default for future
jobs** — it does not touch jobs already running, which hold their own snapshot.
Returns `200` with the `GET /runtime` shape. An unknown stage name returns `400`
listing the valid names and applies **nothing** (a body with one typo does not
half-land).

### `POST /jobs/{id}/runtime`

Body `{"<stage>": <int>, ...}`. Applies to **that running job**, which is the
point of the feature. `404` if the job is unknown, `409` if it is already
terminal. Same response shape.

A job's stage settings are snapshotted from the process-wide defaults at submit;
an explicit `"workers"` on `POST /index` seeds that job's `extract` stage.

## Search moved out of this service

`POST /search/fts` and `POST /search/vector` used to live here but were moved
to the standalone `llm-search` repository (commit `5dcd054`, "move HTTP search
to the standalone search service") — this service is a pure indexer. It still
publishes the `chunks` embedding table those endpoints read; the CLI's
`search`/`vector-search` debug subcommands and the underlying
`store`/`normalize`/`embedding` code are unchanged here. Point keyword search
traffic at the `llm-search` service instead of this one.

`GET /corpus/search` (below) is **not** a walk-back of that split. `llm-search`
holds every chunk vector RESIDENT to serve a search-as-you-type socket, which
is a multi-gigabyte process on a corpus this size; the route below is the
streaming, nothing-resident half — one exhaustive scan per request, `O(limit)`
memory, no second service to deploy — added because on the live corpora 4.1 GB
of already-computed vectors were otherwise reachable only from the CLI.

## Corpus read surface

Consumer apps used to open `corpus.sqlite` directly to render a directory
listing or a document preview. These routes serve that instead, so no consumer
needs to know the SQLite schema. Every route accepts an optional
`output=NAME.sqlite` query param (default `corpus.sqlite`) naming which
published database to read, validated the same way as `POST /index`'s
`output` field.

The database is absent until the first job writes it, and that is not an error:
every route below answers an empty/zeroed result for a corpus that does not
exist yet. A database that **does** exist but cannot be read is different, and
answers `503` with `{"error":"corpus database unreadable","retryable":false}`
rather than a zero — a consumer must be able to tell "nothing indexed yet" from
"the rows are there but unreadable". A rollback journal left by a killed job is
recovered automatically on the first read rather than reported this way, and a
corpus merely locked by a running job answers the retryable `busy` error above,
never this one.

Because jobs write in place, these routes can be served from a corpus that is
still being built. `GET /corpus/status` reports that as `writing`.

### `GET /corpus/tree?root=NAME`

A sorted recursive walk of one allowed input root, joined by absolute path
against the published corpus database. `root` names one of the service's
configured allowed roots — its directory name, e.g. `/input` -> `input`
(`INDEX_ALLOWED_ROOTS`/`--allowed-root`). An unrecognized `root` is `400`.

Returns a JSON array of entries, directories before files, alphabetical within
each:

```json
[
  {
    "path": "Customers/statement.pdf",
    "name": "statement.pdf",
    "kind": "file",
    "depth": 1,
    "size_bytes": 40213,
    "modified_at": 1752600000,
    "document_id": 42,
    "character_count": 8172,
    "method": "pdf",
    "lang": "en",
    "snippet": "first 400 characters of the extracted text…"
  }
]
```

`path` is root-relative POSIX (`/`-separated). `kind` is `"dir"` or `"file"`.
`document_id`, `character_count`, `method`, `lang` and `snippet` are present
only on files that matched a row in the corpus database by exact absolute
path; unmatched files and every directory omit them. Symlinks are skipped,
matching the indexer's own default.

### `GET /corpus/documents/{id}/text`

Streams the extracted text for one document (`files.id`) as
`text/plain; charset=utf-8`. `404` when the database is absent or holds no
matching id; `503` when it exists but could not be read, so a read failure is
never presented as a missing document.

### `GET /corpus/status`

Cheap corpus-wide aggregates:

```json
{
  "indexed_files": 1204,
  "total_characters": 9823110,
  "total_bytes": 512300000,
  "ocr_files": 88,
  "languages": [["en", 900], ["vi", 304]],
  "methods": [["text", 1000], ["pdf-ocr", 204]],
  "writing": false
}
```

`writing` is `true` while a queued, running or cancelling job targets this
`output`: the counts are then a snapshot of a corpus still being built, and will
grow. It is the replacement for the guarantee the old rename-on-success
publication gave for free — that a corpus you could see was a finished one.

### `GET /corpus/search?q=TEXT`

Semantic search over the embeddings index jobs already wrote. `q` is embedded
with the same model the corpus rows were embedded with
(`intfloat/multilingual-e5-small`) and `chunks` is ranked by cosine similarity.

| param | default | meaning |
|---|---|---|
| `q` | — | required; blank or missing is `400` |
| `mode` | `semantic` | `semantic` (exact) or `semantic_fast` (quantised, approximate); anything else is `400` listing the accepted modes |
| `limit` | `20` | clamped to `1..=100` |
| `output` | `corpus.sqlite` | same plain-filename rule as every other route here |

```json
{
  "mode": "semantic",
  "status": "ready",
  "query": "beach at sunset",
  "limit": 20,
  "model": "intfloat/multilingual-e5-small",
  "hits": [
    {
      "path": "C:\photos\2019\IMG_4021.txt",
      "name": "IMG_4021.txt",
      "chunk_index": 0,
      "score": 0.8269,
      "content": "the chunk text that was embedded…"
    }
  ],
  "compared_chunks": 100000,
  "skipped_chunks": 0,
  "elapsed_ms": 1204,
  "path": "scan",
  "exact": true
}
```

`hits` matches `llm-search`'s `/search/vector` rows so the two search surfaces
are one shape. `score` is cosine similarity in `-1.0..=1.0`; ordering is
descending score, ties broken by ascending `chunks.id`, so the same query over
the same corpus always returns the same list in the same order.

**`mode` chooses the promise.** `semantic` (the default) is exact: the answer is
the corpus' true top-k, however long that takes. `semantic_fast` ranks through
the corpus' QUANTISED shadow index instead — sub-second where `semantic` is
seconds, at the cost of returning an approximation of the same list. It is a
separate mode rather than a tolerance knob because approximate and exact are
different promises, and a caller has to choose one deliberately.

**`path` says how the answer was produced**, and **`exact` says whether it is
the scan's own answer.** Never infer the second from the first: a
`semantic_fast` request over a corpus with no quantised index is answered
exactly, and only `exact` tells you so.

| `path` | `exact` | meaning |
|---|---|---|
| `scan` | `true` | the exhaustive cosine scan over every stored vector |
| `vec0` | `true` | a k-NN lookup against the corpus' exact `vec0` shadow index, with the candidates re-scored from the same BLOBs — same hits, same scores, same order as the scan |
| `vec0_int8` | `false` | a k-NN over the int8-quantised index, oversampled and re-scored from the float BLOBs |
| `vec0_bit` | `false` | the same over the 1-bit-quantised index |

On the two quantised paths `candidates` reports how many rows the k-NN nominated
before the float re-score picked the page out of them — the one number behind how
good the approximation is. Every `score` is a true cosine against the stored
vector on every path; what quantisation changes is which rows were scored at all.

A corpus has a shadow index only after `llm-index vector-index` has been run
against it; without one, `path` is always `scan` and nothing else changes. When a
corpus HAS one that was not used, `index_note` says why — an interrupted build,
another model's vectors, or an index that a build without index maintenance has
written behind (`--rebuild` is the repair). Asking for `semantic_fast` on a
corpus with no quantised index also fills `index_note`, with the command that
would build one. A missing `index_note` on `path: scan` means there is no index
to talk about.

**`status` is the field to branch on.** An empty `hits` is never ambiguous:

| `status` | meaning |
|---|---|
| `ready` | the ranking ran; `compared_chunks`/`skipped_chunks` say over what and `path` says how |
| `no_embeddings` | nothing to rank, and `reason` says why: no corpus written yet, a corpus with no `chunks` table, a corpus indexed without embeddings, or one whose vectors came from another model (then `other_models` names them) |
| `warming` | the query embedding model is still loading; `warming_ms` is how long it has been at it |
| `unavailable` | the model could not be loaded, or could not embed this query; `reason` carries the failure and `retrying` says whether a fresh load is already in flight |

Only two things are errors: a malformed request (`400`) and a corpus that exists
but cannot be read (`503`, the same `busy`/`unreadable` bodies the rest of this
surface uses). A corpus indexed with embedding disabled is a `200`.

**First-call cost.** A serve process that has only ever answered reads has no
embedding model loaded. The first `mode=semantic` request does not wait for it:
it arms the load, returns `status: "warming"` immediately (measured: 207 ms
wall, `warming_ms: 0`), and later requests report a growing `warming_ms` until
the model is ready — measured 5,336 ms end to end on the workhorse, logged as
`query embedding model loaded; semantic search is ready load_ms=5336`. A failed
load is reported with its reason and re-armed on the next request, so it never
latches. Consumers that want the first *user* query to be fast should fire one
throwaway search at startup.

**Per-query cost.** The scan is exhaustive, so latency scales with the corpus:
measured 0.24 s per 100 k vectors and 13.7 s over the live 2.68 M-vector /
15.6 GB corpus. A corpus with an exact `vec0` shadow index reads the vectors only
and answers roughly an order of magnitude faster — best warm passes 1.32 s at
869 k vectors and 3.9 s at 2.68 M, against 13.9 s and 45.6 s for the scan measured
back to back on the same loaded workstation.

`mode=semantic_fast` over an `int8` index is faster again, and it is the only
configuration that is ever interactive — measured over 20 real query embeddings,
`limit=10`, **recall@10 1.0000 against the exact answer at both sizes**:

| vectors | `semantic` (exact index) | `semantic_fast` (`int8`) |
|---|---|---|
| 869,267 | 916 ms best | **571 ms best, 586 ms median** |
| 2,684,125 | 3,787 ms best | 1,860 ms best, 1,945 ms median |

So sub-second arrives below about a million vectors and not at 2.68 M, where the
`int8` k-NN is bound by `sqlite-vec` 0.1.9's scalar int8 kernel rather than by
I/O. The `bit` tier IS sub-second there (143 - 399 ms) at recall@10 0.125 - 0.445,
which is why it is not what `semantic_fast` suggests building. `docs/ARCHITECTURE.md`
carries every pool, both tiers, both corpora, the build costs, and why `vec0`
0.1.9 is a faster brute force rather than an ANN. None of this is a
search-as-you-type endpoint at 2.68 M vectors.
