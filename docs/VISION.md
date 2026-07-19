# Vision modes

Local computer-vision understanding of photos and videos: descriptions, tags,
object counts, EXIF, perceptual hashes and (opt-in) captions, written into the
corpus database alongside the existing text/OCR pipeline. Everything runs
locally — ONNX models via `ort`/`fastembed`, plus pure Rust code — with **no
cloud calls, no LLM APIs, and no network activity during indexing**. The
feature is **off by default everywhere**; `ff-lc-app`, `da-academic` and any
other existing caller see zero behavior change unless they explicitly opt in.

Authoritative design contract: `docs/VISION-SPEC.md`. GPU/hardware research
(RTX 3070 Ti stack, throughput estimates, licensing survey): `docs/VISION-RESEARCH.md`.

## Status

V1 (plumbing) plus the meta/tags/video tier bodies have landed: the
`VisionMode` type, `VisionConfig`, CLI/serve/fetch-data wiring, the `vision`
table + `upsert_vision`, the pipeline hooks (change detection, FTS content
append), and submit-time validation are all in place and tested — including the
off-path invariant (a `vision: "off"` job produces byte-identical rows to a
build with the feature absent).

| Tier | Owner | Status |
|---|---|---|
| `meta` (EXIF, phash, quality) | V2 | **live** — EXIF (camera/datetime/GPS), 64-bit DCT perceptual hash, and quality metrics (blur, exposure) are populated in pure code (no models). |
| `tags` (CLIP tags + embedding, object detector) | V3 | **live** — CLIP ViT-B/32 zero-shot tags + image embedding (via `fastembed`), and RF-DETR-Nano object counts (via `ort`). Needs models present: CLIP from `fastembed`'s cache, the detector from `fetch-data --vision`. |
| `captions` (Florence-2) | V5 | **unsupported stub (deferred to V6)** — opting into `captions` still runs `meta`+`tags`; the caption itself records an `unsupported` note in `vision.error`. See the `captions` tier note below. |
| video | V4 | **live** — scene-change keyframes (fixed-interval fallback), per-frame tags (capped at the `tags` tier), aggregation, and whisper-transcript merge. |

The object detector follows the `docs/VISION-SPEC.md` **AMENDMENT 2026-07-19**:
it is **RF-DETR-Nano (Apache-2.0)**, not Ultralytics YOLO11 (whose AGPL-3.0
code+weights are a licensing hazard for this engine). The detector module
(`src/vision/detector.rs`), the `VISION_MODELS` registry row
(`onnx-community/rfdetr_nano-ONNX`, pinned URL + SHA-256), and the
`detector_conf` config field all reflect the amended design; the interim
`yolo.rs` shim and `yolo_conf`/`yolo_iou` fields have been removed.

## Tiers

Tiers are cumulative: `off < meta < tags < captions`. Requesting a tier runs
every tier at or below it. A file's `vision.mode` column records the highest
tier that actually ran for it.

- **`off`** (default) — no vision analysis. Fully inert: no image is decoded,
  no `vision` row is written, `files.method` is exactly what it would be
  without the feature.
- **`meta`** — pure code, no models, no network:
  - EXIF (camera make/model, `DateTimeOriginal`, GPS lat/lon)
  - image dimensions
  - a 64-bit DCT perceptual hash (`phash`, 16 hex chars) for near-duplicate
    detection
  - quality metrics: blur (variance of the Laplacian) and over/under-exposure
    (luma histogram)
- **`tags`** — everything in `meta`, plus small local CV models:
  - a CLIP ViT-B/32 image embedding (via `fastembed`, already a dependency)
    plus zero-shot tag scoring against the curated vocabulary in
    `data/vision-tags.txt` (~300 labels: scenes, objects, document/screenshot
    kinds, people/groups, vehicles, food, pets, receipts/invoices,
    whiteboards, …). Vocabulary text-encoder embeddings are computed once per
    process and cached. Top `tag_top_k` tags scoring above `tag_score` are
    kept.
  - RF-DETR-Nano object detection (COCO classes) via `ort`, DETR-style
    postprocessing (no NMS pass needed), aggregated into per-label counts with
    a max confidence.
- **`captions`** — everything in `tags`, plus a best-effort one/two-sentence
  Florence-2-base caption (opt-in — it needs a ~500 MB model and is the
  slowest tier). If captions prove impractical to land at passing-test
  quality, the tier returns a clean error instead of blocking the rest of the
  release; see the Status table above.
- **video** — analysed at whichever tier `mode` requests: ffmpeg extracts
  scene-change keyframes (falling back to a fixed interval), capped at
  `max_frames` (default 12), each keyframe runs through the same image tag
  pipeline, and the results are aggregated and merged with the file's existing
  Whisper transcript (unaffected).

### The FTS `[vision]` block

Vision composes **with** OCR, not instead of it — a photographed document
still gets its OCR text, with a vision block appended so the description,
tags, objects and camera metadata are all searchable via the same
`FTS MATCH` query used for everything else:

```
[vision] caption: two people walking a dog on a beach at sunset
objects: person(2), dog(1)
tags: beach, sunset, outdoors, family
camera: Apple iPhone 15 Pro, 2024-06-01T18:22, GPS 10.79,106.70
```

Each line is independent and only appears when that tier produced non-empty
data — a `meta`-only file gets just a `camera:` line (or no block at all if
EXIF carried no camera/date/GPS), a `tags` file gets `objects:`/`tags:` but no
`caption:`, etc. A file with nothing vision-worthy to say (e.g. a decode
failure, or a tier that found no objects/tags/caption/EXIF) appends no block
at all — the plain OCR/text content is untouched.

## Configuration

### CLI flags

| Command | Flag | Default | Notes |
|---|---|---|---|
| `index` (native) | `--vision <off\|meta\|tags\|captions>` | `off` | Sets the effective tier for that run. |
| `serve` | `--vision-max <tier>` | `off` | Highest tier the server will accept from any job. Env fallback `INDEX_VISION_MAX`. Requests above the cap are rejected at submit with a `400`, keeping deployments that don't set this inert with **no compose changes required**. |
| `request` (native HTTP client) | `--vision <tier>` | `off` | Sets `JobRequest.vision` on the submitted job. |
| `fetch-data` | `--vision` | off (flag) | Fetches vision model artifacts instead of dictionaries/OCR data; see below. |

`serve --vision-max` and a job's requested `--vision`/`JobRequest.vision`
compose as a hard cap: the effective tier for a job is
`min(requested, vision_max)`, enforced both at submit (a clear `400` telling
the caller their tier exceeds the server's max) and again defensively when the
job actually runs.

### `JobRequest.vision`

The HTTP job payload (`POST /index`) gains an optional `vision` field:

```json
{ "paths": ["…"], "output": "corpus.sqlite", "vision": "tags" }
```

- Absent or `null` → `off`.
- An unrecognized string → `400` at submit (`unknown vision tier '…' (expected
  off, meta, tags, or captions)`).
- A tier above the server's `--vision-max` → `400` at submit.
- A tier that needs models (`tags`/`captions`) whose files are not present
  under `<data_dir>/vision` → `400` at submit
  (`vision models missing; run llm-index fetch-data --vision`) rather than a
  per-file surprise partway through the job.

### `VisionConfig` (YAML `vision:` block / `Config::vision`)

All fields are optional in YAML — unspecified keys keep their default, so
existing `config.yaml` files are unaffected:

| Field | Default | Meaning |
|---|---|---|
| `max` | `off` | Effective tier ceiling for a run; overridden by `index --vision` / `serve`+`JobRequest` (native `index` sets it directly, service mode sets it per job after validation). |
| `models_dir` | `vision` | Directory model files live under; resolved relative to `data_dir` when not absolute (i.e. `<data_dir>/vision` by default). |
| `tag_score` | `0.22` | Minimum CLIP zero-shot tag score to keep. |
| `tag_top_k` | `8` | Maximum number of tags kept per file. |
| `detector_conf` | `0.5` | Minimum object-detector confidence to keep a detection (RF-DETR-Nano, DETR-style postprocessing, no NMS). There is no IoU/NMS threshold — RF-DETR emits one object per query, so no NMS pass exists. |
| `max_frames` | `12` | Maximum keyframes analysed per video. |
| `timeout_secs` | `60` | Per-file vision timeout (seconds) for non-caption tiers. |
| `caption_timeout_secs` | `300` | Per-file vision timeout (seconds) for the captions tier. |
| `max_pixels` | `250,000,000` | Images above this pixel count are rejected before a full decode (`vision.error = "decode-limit"`). |
| `max_alloc_bytes` | `1073741824` (1 GiB) | Caps a single decode allocation (`image` crate `Limits`); tripping it also records `decode-limit`. |

## Fetching models: `fetch-data --vision`

```bash
llm-index fetch-data --vision [--data-dir data] [--force]
```

Downloads the pinned model artifacts into `<data_dir>/vision/`, verifying each
download's SHA-256 against the hash pinned in source **before** writing it to
disk; a mismatch is a hard error (nothing partially-verified is ever left on
disk). Existing files are left alone unless `--force` is passed. This is the
**only** place vision models are ever downloaded — never automatically during
`index`/`serve`.

| Tier | Model | Runtime | Approx size | License | How it's obtained |
|---|---|---|---|---|---|
| `tags` | CLIP ViT-B/32 (image encoder + paired text encoder) | `fastembed` (bundles ONNX Runtime) | ~350 MB | MIT (OpenAI CLIP) | `fastembed`'s own model cache on first use — **not** via `fetch-data --vision` |
| `tags` | RF-DETR-Nano (`onnx-community/rfdetr_nano-ONNX`, COCO object detector, DETR-style, no NMS) | `ort` (reusing `fastembed`'s bundled version where possible) | ~20–40 MB | Apache-2.0 | `fetch-data --vision`, pinned SHA-256 |
| `captions` | Florence-2-base (encoder + decoder ONNX graphs) | `ort`, greedy decode, ≤64 tokens | ~500 MB | MIT (Microsoft Florence-2-base) | not pinned in v1 — captions ships as an unsupported stub (see below) |

The tag vocabulary (`data/vision-tags.txt`, ~300 curated labels) ships in the
repo rather than being downloaded.

The source registry (`src/vision/mod.rs::VISION_MODELS`) pins the RF-DETR-Nano
detector with a real URL + SHA-256, so `fetch-data --vision` downloads and
verifies it. The two Florence-2 rows are intentionally left with no URL/hash
while the captions tier is the v1 unsupported stub; `fetch-data --vision`
prints a `skipping … — download URL not yet pinned` note for them, which is
expected and not an error.

## Consumer compatibility

Existing consumers (`ff-lc-app`, `da-academic`, `llm-search`) require **zero**
changes to keep working exactly as before:

- The `vision` table is additive (`CREATE TABLE IF NOT EXISTS`, like every
  other schema evolution here) — pre-existing databases upgrade transparently
  on open.
- `chunks` (the 384-dimensional `multilingual-e5-small` text vectors
  `llm-search` reads) is untouched. CLIP's 512-dimensional image embeddings
  live only in `vision.embedding`/`vision.dimensions`/`vision.embedding_model`
  — mixing them into `chunks` would corrupt its vector math, so they never
  are. A future text→image search path can read `vision` directly.
- **`files.method` values are unchanged** by vision — consumers that key
  behavior on `method` see identical values whether vision ran or not; vision
  presence/absence lives entirely in the separate `vision` table.
- With `vision` left at its default `off` (every existing deployment, since
  `serve --vision-max` also defaults to `off`), the pipeline never decodes an
  image for vision purposes, never writes a `vision` row, and the FTS content
  for every file is byte-identical to a build without this feature — this is
  covered by an explicit regression test.
- Deploying this release requires no compose/env changes for existing
  callers: `serve --vision-max` (and `INDEX_VISION_MAX`) both default to
  `off`, so a caller has to explicitly raise the server's cap **and** request
  a non-`off` tier before anything changes.
- Job completion summaries gain one additive field, `vision_files` (count of
  files a vision tier ran or recorded an error for) — existing fields are
  unchanged.
- Resume/incremental indexing extends the existing change-detection rules: a
  vision-eligible file is reprocessed only when the requested tier is higher
  than the tier recorded in its `vision.mode` (or it has no `vision` row yet).
  Lowering the requested tier on a later run never deletes previously-recorded
  vision rows — turning vision off (or down) for a job is always non-destructive.

## Performance

Pure-code `meta` tier work (EXIF parse, DCT phash, Laplacian/histogram
quality) is negligible next to file I/O and image decode. For the model-backed
tiers, on CPU:

- `tags` (CLIP embedding + zero-shot scoring + object detection): roughly
  **50–150 ms per image**.
- `captions` (Florence-2): **on the order of seconds per image** — this is why
  it is opt-in and gated behind its own tier rather than folded into `tags`.
- Video cost scales with the number of keyframes actually sampled (capped at
  `max_frames`, default 12) run through the same per-image `tags` pipeline,
  plus ffmpeg's own decode/scene-detection time.

These are CPU, single-image, un-batched estimates for capacity planning — they
are not a substitute for measuring against real hardware and library
composition.

**GPU:** this release does not wire up a GPU execution provider — everything
above runs on CPU via `ort`'s default EP. For a concrete GPU deployment plan
(CUDA/TensorRT execution providers, per-tier VRAM budgets, batching, and
wall-time estimates for 100k–500k-file libraries on an RTX 3070 Ti class card),
see the separate research report: `docs/VISION-RESEARCH.md`. That report also
covers a broader model survey (e.g. SigLIP2, RAM++, face recognition) that is
**not** part of this v1 scope — the approved tiers here are exactly the ones
in `docs/VISION-SPEC.md` (CLIP + RF-DETR-Nano for tags, Florence-2-base for
captions).

## Security model

- **No network at index time.** All inference (CLIP, RF-DETR-Nano,
  Florence-2) runs locally through `ort`/`fastembed`; the pure-code `meta`
  tier obviously makes no network calls either. The *only* network activity
  vision ever introduces is the explicit, operator-run
  `fetch-data --vision` step — never anything triggered by `index` or a
  `serve` job.
- **Pinned artifacts.** Every downloadable model has its HTTPS URL and
  expected SHA-256 hard-coded in source, next to each other
  (`src/vision/mod.rs::VISION_MODELS`), resolved and pinned by the worker who
  lands that tier — never resolved dynamically or auto-discovered at runtime.
  `fetch-data --vision` verifies the SHA-256 of what it downloaded before
  writing it to disk, and refuses (hard error) on a mismatch.
- **Submit-time model check.** A job requesting a tier whose models are not
  present under `<data_dir>/vision` is rejected as a whole at submit (`400`),
  rather than discovering the gap file-by-file mid-job.
- **Decode hardening.** Images are decoded through the `image` crate with
  explicit `Limits`: a cheap dimension probe rejects an oversized image before
  any pixel buffer is allocated (`max_pixels`, default ~250 megapixels), and a
  hard cap on a single decode allocation (`max_alloc_bytes`, default 1 GiB)
  catches anything the dimension probe misses. Either limit trips
  `vision.error = "decode-limit"`; an otherwise-corrupt/unreadable file trips
  `"decode-error"`. In both cases that one file is skipped and the rest of the
  job continues — vision analysis never panics and never aborts a run.
- **Per-file timeouts.** `timeout_secs` (default 60) and, separately,
  `caption_timeout_secs` (default 300, since captions are the slowest tier) are
  intended to bound worst-case per-file time. *(Implementation note: these
  config fields are defined and threaded through `VisionConfig`, but v1 does
  **not** yet wire them to an enforcement point around model inference / ffmpeg
  — no code path reads them yet. This is deferred hardening; in practice the
  live tiers are fast (meta is pure code, tags is a single forward pass) and
  the slowest tier, captions, is the unsupported stub.)*
- **Determinism.** Greedy decoding, fixed thresholds, no RNG — a given file
  and config produce the same vision result every time.
- **ffmpeg invocation (video).** A fixed argv (`-nostdin`, explicit filters),
  output into a fresh temp directory that is cleaned up on every exit path
  (success, error, or timeout) — no shell interpolation of file paths.
