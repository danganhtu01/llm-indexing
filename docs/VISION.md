# Vision modes

Local computer-vision understanding of photos and videos: descriptions, tags,
object counts, EXIF, perceptual hashes and (opt-in) captions, written into the
corpus database alongside the existing text/OCR pipeline. Everything runs
locally — ONNX models via `ort`/`fastembed`, plus pure Rust code — with **no
cloud calls, no LLM APIs, and no network activity during indexing**. The
feature is **off by default everywhere**; `ff-lc-app`, `da-academic` and any
other existing caller see zero behavior change unless they explicitly opt in.

Authoritative design contract: `docs/VISION-SPEC.md`. GPU/hardware research
(RTX 3070 Ti stack, throughput estimates, licensing survey):
`docs/VISION-RESEARCH.md`. Per-job overrides of the vision knobs below
(`detector`, `detector_conf`, `tagger`, `tag_threshold`, `tag_top_k`,
`captioner`, `max_frames`, `timeout_secs`) — plus the matching OCR knobs — are
the wave-2 settings surface: `docs/SETTINGS-SPEC.md` (design contract),
[`docs/HTTP_API.md`](HTTP_API.md#ocr_opts--vision_opts--per-job-quality-overrides)
(`ocr_opts`/`vision_opts` on `POST /index`, validation table, matching
`--ocr-*`/`--vision-*` CLI flags) and
[`docs/HTTP_API.md#get-settings`](HTTP_API.md#get-settings) (capability
discovery: bounds, installed OCR languages, model presence).

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
| `faces` sub-tier (YuNet + SFace) | F1 | **live, opt-in, OFF by default** — face boxes + 128-d embeddings into a separate `faces` table. Not a tier: a sub-model of `tags`, gated by its own `vision.faces` toggle and its own `fetch-data --faces` staging step. See [Faces](#faces-opt-in-privacy-sensitive). |

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

Faces is deliberately **not** a fifth tier — see [Faces](#faces-opt-in-privacy-sensitive)
for why a ladder is the wrong shape for it.

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
| `fetch-data` | `--faces` | off (flag) | Fetches the opt-in YuNet+SFace pair. Independent of `--vision` in both directions; usable together. |
| `index` (native) | `--vision-faces <off\|yunet-sface>` | `off` | Face detection + embedding for that run. |
| `index` (native) | `--vision-face-score <0.05..=0.99>` | `0.9` | Minimum face-detection score kept. |
| `index` (native) | `--vision-max-faces <1..=200>` | `20` | Max faces kept per file. |

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
| `faces` | `off` | Face detection + embedding: `off` or `yunet-sface`. The one sub-model whose default is `off` rather than its model id — see [Faces](#faces-opt-in-privacy-sensitive). |
| `face_score` | `0.9` | Minimum YuNet detection score to keep a face. |
| `max_faces` | `20` | Maximum faces kept per file (per file, not per video keyframe). |
| `max_frames` | `12` | Maximum keyframes analysed per video. |
| `timeout_secs` | `60` | Per-file vision timeout (seconds) for non-caption tiers. |
| `caption_timeout_secs` | `300` | Per-file vision timeout (seconds) for the captions tier. |
| `max_pixels` | `250,000,000` | Images above this pixel count are rejected before a full decode (`vision.error = "decode-limit"`). |
| `max_alloc_bytes` | `1073741824` (1 GiB) | Caps a single decode allocation (`image` crate `Limits`); tripping it also records `decode-limit`. |

## Faces (opt-in, privacy-sensitive)

Local face detection and recognition-grade embeddings: **YuNet**
(`face_detection_yunet_2023mar`) finds faces and their five landmarks,
**SFace** (`face_recognition_sface_2021dec`) turns each aligned crop into a
128-dimensional vector. Both are OpenCV-Zoo ONNX artifacts under
**Apache-2.0** — the only fully-clean face pair in the
`docs/VISION-RESEARCH.md` §6 survey; the more accurate InsightFace `buffalo_l`
packs carry non-commercial weights and are never shipped here. They run on the same `ort` the object
detector already uses, so no second runtime is introduced.

### Off by default, at every layer

A face embedding is a biometric identifier for a person who did not opt in —
including people who merely appear in the background of someone else's photo.
Everything else this engine writes is a fact about a *file*; this is a fact
about a *person*. So the feature is opt-in four times over, and each layer
defaults to off:

| Layer | Default | To turn on |
|---|---|---|
| Model staging | pair not downloaded | `llm-index fetch-data --faces` (`--vision` does **not** fetch it) |
| Config | `vision.faces: off` | `vision.faces: yunet-sface` |
| Per-job | `vision_opts.faces` absent ⇒ `off` | `"vision_opts": {"faces": "yunet-sface"}` |
| CLI | `--vision-faces` unset ⇒ `off` | `--vision-faces yunet-sface` |

Faces also needs the `tags` tier or higher, because it is a model-backed
sub-model and `meta` is defined as pure code. It is **not** a fifth tier, and
that is the whole design decision: tiers are cumulative, so a `faces` rung above
`captions` would mean anyone requesting the top tier silently starts enrolling
faces. A separate toggle beside `detector`/`tagger`/`captioner` is the shape
that lets a deployment run the richest vision tier and still never look at a
face.

### Absent models = capability absent, never an error

The `tags` tier fails a job at submit when its models are missing. Faces
deliberately does the opposite: if the pair is not staged, the capability is
absent, the job runs the rest of its tier unchanged, no faces are written, and
`vision.error` is untouched (the file is **not** held back from its tier, so
nothing is stranded). The job logs one `faces requested but the yunet/sface pair
is not staged` warning so the silence is explained, and `GET /settings` reports
`vision.faces[].present: false` so an app can grey the control out instead of
offering something the box cannot do.

The one case that DOES fail a job is a pair that is present but fails its pinned
SHA-256 — corrupt, truncated or swapped. An absent pair is a choice not to have
the capability; a wrong pair is bytes nobody vouched for computing claims about
who someone is.

### What is stored, and where it stays

Faces land in their own table, never in the searchable text:

```sql
CREATE TABLE IF NOT EXISTS faces(
  file_id INTEGER NOT NULL REFERENCES files(id) ON DELETE CASCADE,
  face_index INTEGER NOT NULL,    -- position in the detector's best-first order
  x INTEGER NOT NULL, y INTEGER NOT NULL,
  width INTEGER NOT NULL, height INTEGER NOT NULL,
  quality REAL NOT NULL,          -- detector score in [0,1] - the FIQA gate
  embedding BLOB, dimensions INTEGER,   -- 128 little-endian f32, as `chunks`
  model TEXT NOT NULL,            -- 'yunet-sface'
  frame INTEGER,                  -- video: keyframe ordinal; NULL for stills
  PRIMARY KEY(file_id, face_index)
);
```

- **Local-only.** Inference is in-process `ort`; there is no network at index
  time (the models arrive only through the operator-run fetch). The vectors are
  written into the per-drive corpus on the machine that indexed the file. This
  engine never transmits them.
- **Never searchable text.** Faces are the one vision output excluded from the
  `[vision]` FTS block, from sidecar `.txt` files and from manifests — adding
  faces to a result leaves its content block byte-identical (asserted by test).
- **Never in a summary beyond a count.** The job summary gains one additive
  field, `faces` (total faces stored). No paths, boxes or vectors.
- **Deleted with the file.** `ON DELETE CASCADE` plus an explicit delete in the
  prune path: a file that vanishes takes the faces attributed to it with it.
- **`vision.faces_model`** records which pair scanned a file. It is the evidence
  a scan *happened*, not that a face was found — a landscape photo gets the
  stamp and no rows. Old corpora gain the column by `ALTER TABLE` on open, with
  no backfill: NULL is already the truthful "never scanned".

### Quality gating and determinism

- `face_score` (default **0.9**, OpenCV's own default) is the detection floor.
  High on purpose: a false face enters a person's cluster and has to be disowned
  by hand, which costs more than a miss.
- Faces whose shorter side is under **24 original-image pixels** are dropped
  before embedding — below that the aligned 112x112 crop is mostly upsampling
  artefact, and per digiKam 8.6's published lesson a blurry or tiny face poisons
  a cluster far more than a missed face costs. Fixed, not a knob.
- NMS IoU is fixed at **0.3** (OpenCV's default) for the same reason: no
  per-corpus right answer, and a knob would only make two jobs incomparable.
- `max_faces` (default **20**) bounds a file — for a video the whole file, not
  each keyframe.
- **Determinism.** Same file + same model files ⇒ same faces, same order, same
  bytes. No augmentation, no multi-crop voting, no RNG: the image is
  letterboxed into YuNet's fixed 640x640 input with a fixed filter, decoded
  against fixed priors, gated at fixed thresholds, ordered by score with
  positional tie-breaks, and each crop is aligned by a closed-form similarity
  transform onto fixed reference landmarks. Identical bytes are guaranteed for a
  given build, model file and ONNX Runtime execution provider — the same caveat
  every other tier here carries.

### Video

Keyframes already run through the per-image pipeline, so faces come along for
free; the aggregate concatenates each keyframe's faces and stamps `frame` with
the keyframe ordinal. It is deliberately **not** de-duplicated: one person
walking through five keyframes is five rows. Deciding those five are one person
is cross-frame identity work, which belongs to the app, not the engine. No
keyframe *timestamp* is recorded today, so a video face is usable for "who
appears in this file" but not yet for seeking to the moment — that needs
timestamped keyframes, which this workstream deliberately did not build.

### Resume

Turning faces on is an upgrade the tier ladder cannot see — the requested tier
does not change — so without a rule of its own a corpus already at `tags` would
skip every file and the job would appear to do nothing. A vision-eligible file
is therefore reprocessed when the enabled pair is not the pair recorded in its
`vision.faces_model`, and like every other upgrade this is never subject to the
attempt cap. Turning faces **off** is non-destructive in both directions: it
reprocesses nothing, and existing face rows are carried forward across the
re-index exactly as vision rows are (they are dropped only when the file's own
bytes change, at which point they describe content that is gone).

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
| `faces` (opt-in) | YuNet `face_detection_yunet_2023mar` (`yunet.onnx`) | `ort` | ~230 KB | Apache-2.0 (opencv/opencv_zoo) | **`fetch-data --faces` only**, pinned URL (tag `4.10.0`) + SHA-256 |
| `faces` (opt-in) | SFace `face_recognition_sface_2021dec` (`sface.onnx`) | `ort` | ~37 MB | Apache-2.0 (opencv/opencv_zoo) | **`fetch-data --faces` only**, pinned URL (tag `4.10.0`) + SHA-256 |

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

- The `vision` and `faces` tables are additive (`CREATE TABLE IF NOT EXISTS`,
  like every other schema evolution here) — pre-existing databases upgrade
  transparently on open, and `vision` gains its one new column
  (`faces_model TEXT`) by `ALTER TABLE ADD COLUMN` with no backfill. `faces` is
  an ordinary table, so a build that has never heard of it reads, writes, joins
  and migrates a corpus that has one (asserted in `tests/old_binary.rs`).
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
- Job completion summaries gain two additive fields, `vision_files` (count of
  files a vision tier ran or recorded an error for) and `faces` (count of faces
  stored, 0 unless the opt-in faces sub-tier ran) — existing fields are
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
