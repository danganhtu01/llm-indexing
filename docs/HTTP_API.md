# HTTP API
The container listens on TCP 9801. All API requests and responses are JSON.

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
    "defaults": {
      "detector_conf": 0.5,
      "tag_threshold": 0.22,
      "tag_top_k": 8,
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
- `vision.detectors`/`taggers`/`captioners` list every selectable sub-model id
  (`ocr_opts`/`vision_opts`' accepted enum values, minus `off`) with a
  `present` flag backed by the same model-file existence + hash check
  (`detector_present`/`tagger_present`/`captioner_present`). In v1 each
  category has exactly one model, so all its ids currently share one flag.
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

Queues one job and returns `202 Accepted` with an `id`.

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
| `vision_opts.max_frames` | integer | `1..=64` | Max video keyframes analysed. |
| `vision_opts.timeout_secs` | integer | `5..=1800` | Per-file vision timeout (seconds). |

Unknown top-level fields anywhere in the job body remain permissively ignored
(existing forward-compat serde posture) — only the fields above are validated.

## `GET /jobs/{id}`

Returns `queued`, `running`, `cancelling`, `cancelled`, `complete` or `error`.
Running jobs include live `processed` and `total` file counters. A completed job includes the
database path, file/OCR/error/incomplete counts, embedded chunk count, removed
source count, elapsed time and OCR languages.

## `POST /jobs/{id}/cancel`

Requests cooperative cancellation of a queued or running job. The engine stops
before the next extraction/embedding boundary, discards the temporary build and
keeps the previously published SQLite corpus unchanged. Poll the job until its
state becomes `cancelled`.

## Search moved out of this service

`POST /search/fts` and `POST /search/vector` used to live here but were moved
to the standalone `llm-search` repository (commit `5dcd054`, "move HTTP search
to the standalone search service") — this service is a pure indexer. It still
publishes the `chunks` embedding table those endpoints read; the CLI's
`search`/`vector-search` debug subcommands and the underlying
`store`/`normalize`/`embedding` code are unchanged here. Point search traffic
at the `llm-search` service instead of this one.

## Corpus read surface

Consumer apps used to open `corpus.sqlite` directly to render a directory
listing or a document preview. These routes serve that instead, so no consumer
needs to know the SQLite schema. Every route accepts an optional
`output=NAME.sqlite` query param (default `corpus.sqlite`) naming which
published database to read, validated the same way as `POST /index`'s
`output` field. The database is absent until the first job completes; every
route below then degrades to an empty/zeroed result rather than an error.

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
matching id.

### `GET /corpus/status`

Cheap corpus-wide aggregates:

```json
{
  "indexed_files": 1204,
  "total_characters": 9823110,
  "total_bytes": 512300000,
  "ocr_files": 88,
  "languages": [["en", 900], ["vi", 304]],
  "methods": [["text", 1000], ["pdf-ocr", 204]]
}
```
