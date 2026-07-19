# llm-indexing — Vision modes v1 (owner-approved scope, 2026-07-19)

Add local computer-vision understanding of photos and videos to the indexer:
descriptions/tags/objects/EXIF/perceptual-hashes written into the corpus DB, not
just filename metadata. Everything runs locally (ONNX / pure code) — no cloud, no
LLM APIs, no network at index time. Default OFF everywhere; existing consumers
(ff-lc-app, da-academic, llm-search, drives-analytics) see zero behavior change
until a caller opts in.

Approved tiers:
- **meta** (pure code): EXIF (camera, datetime, GPS), dimensions, 64-bit DCT
  perceptual hash (near-duplicate detection), quality metrics (blur via variance of
  Laplacian, over/under-exposure via histogram).
- **tags** (small local CV models): CLIP image embedding + zero-shot tags against a
  curated vocabulary (via fastembed, ALREADY a dependency — it bundles ONNX Runtime
  and supports ImageEmbedding + the paired clip text encoder), and YOLO11 object
  detection (counts per label) via the `ort` crate.
- **captions** (opt-in, best-effort): Florence-2-base ONNX one/two-sentence caption.
- **video**: ffmpeg (already a runtime dep) scene-change keyframes (fixed-interval
  fallback), capped at `max_frames` (default 12), per-frame tags pipeline,
  aggregated + merged with the existing whisper transcript.

Tier order: `off < meta < tags < captions`; each tier includes the ones below.

## 1. Surface & security model

- `JobRequest` gains `vision: Option<String>` (default `"off"`), validated against
  the tier set at submit (`400` on unknown value).
- `serve` gains `--vision-max <tier>` **default `off`** (env fallback
  `INDEX_VISION_MAX`): requests above the cap are rejected at submit with a clear
  error. This keeps the fflc/da-academic deployments inert with no compose changes.
- `index` (native CLI) gains `--vision <tier>`.
- Job submit pre-flight: when the requested tier needs models, verify the model
  files exist and hash-match → job-level 400 (`vision models missing; run
  llm-index fetch-data --vision`) rather than per-file surprises.
- Models are fetched ONLY by `fetch-data --vision` (new): HTTPS downloads with
  **pinned SHA-256 checksums** hard-coded in the source next to the URL (worker
  resolves the real URLs + hashes and pins them; document each model's license in
  docs/VISION.md). No auto-download anywhere else.
- Image decode hardening: use the `image` crate with explicit `Limits` (cap ~250
  megapixels / 1 GiB alloc); over-limit or corrupt files record
  `vision.error='decode-limit'`/`'decode-error'` for that file and continue.
- Per-file vision timeout: `vision_timeout_secs` (default 60; captions 300). ffmpeg
  invoked with a fixed argv (`-nostdin`, explicit filters, output into a fresh
  tempdir, cleaned up on every path).
- Determinism: greedy decode, fixed thresholds, no RNG.

## 2. Storage (additive; schema stays consumer-safe)

New table (in `SCHEMA` const, `IF NOT EXISTS`, so existing DBs upgrade on open —
consistent with `resume`):

```sql
CREATE TABLE IF NOT EXISTS vision(
  file_id INTEGER PRIMARY KEY REFERENCES files(id) ON DELETE CASCADE,
  mode TEXT NOT NULL,             -- highest tier that ran for this file
  width INTEGER, height INTEGER,
  phash TEXT,                     -- 16-hex 64-bit DCT hash
  exif_json TEXT, quality_json TEXT,
  objects_json TEXT,              -- [{"label","count","max_conf"}]
  tags_json TEXT,                 -- [{"tag","score"}] top-k over threshold
  caption TEXT,
  embedding BLOB, embedding_model TEXT, dimensions INTEGER,  -- CLIP image vector
  frames INTEGER,                 -- video: keyframes analyzed
  elapsed_ms INTEGER, error TEXT);
CREATE INDEX IF NOT EXISTS idx_vision_phash ON vision(phash);
```

**Deliberate choices:**
- Image embeddings go HERE, not into `chunks` — `chunks` holds 384-dim e5 text
  vectors consumed by llm-search; mixing 512-dim CLIP vectors there would corrupt
  its vector math. llm-search grows a text→image path later by reading `vision`.
- `files.method` values are UNCHANGED (consumers key on them). Vision presence
  lives in the `vision` table; prune vision rows alongside file pruning (the
  existing `ON DELETE CASCADE` handles it — verify files deletion uses DELETE).
- FTS: the composed `fts.content` for a file gains an appended vision block (so
  everything is instantly text-searchable):

```
[vision] caption: two people walking a dog on a beach at sunset
objects: person(2), dog(1)
tags: beach, sunset, outdoors, family
camera: Apple iPhone 15 Pro, 2024-06-01T18:22, GPS 10.79,106.70
```

- OCR composes WITH vision (a photographed document gets both OCR text and the
  vision block).
- Incremental (`resume`): extend the change-detection rules — a file is reprocessed
  when the requested tier exceeds what its `vision.mode` recorded (or no row).
  Lowering the tier never deletes existing vision rows.
- Job completion summary gains `vision_files` count.

## 3. Models (worker pins exact artifacts)

| Purpose | Model | Runtime | Approx size |
|---|---|---|---|
| Image embedding + tag scoring | CLIP ViT-B/32 (image + paired text encoder) | fastembed (existing dep) | ~350 MB |
| Object detection | YOLO11n or YOLO11s ONNX (COCO-80) | `ort` (same version tree fastembed uses — check `cargo tree`) | 10–20 MB |
| Captions (opt-in) | Florence-2-base ONNX (encoder/decoder) | `ort`, greedy decode, ≤64 tokens | ~500 MB |

Tag vocabulary: repo file `data/vision-tags.txt` (~300 curated labels: scene types,
common objects, document kinds, screenshots, people/groups, vehicles, food, pets,
UI/screens, receipts/invoices, whiteboards, …). Text-encoder embeddings of the
vocabulary computed once per process and cached.

New crate deps (pin exact versions; nothing else): `image`, `kamadak-exif`, `ort`
(only if fastembed's bundled version can't be reused directly), small math helpers
in-tree (DCT for pHash is ~40 lines — implement, don't add a dep).

## 4. Integration points (from the current tree — verify while implementing)

- `src/extract.rs:143-154` image handling (currently OCR-only) — vision orchestrator
  hooks in beside OCR, both contribute to the content string.
- `src/pipeline.rs:47-81` change detection (add vision-upgrade rule), `:174,222-233`
  NFC/content composition (append vision block), `:193-197` method assignment
  (UNCHANGED).
- `src/store.rs:14-49` SCHEMA const (add vision DDL), add `upsert_vision` +
  prune-with-files.
- `src/config.rs` add `VisionConfig` (tier cap for native mode, model paths under
  `<data_dir>/vision`, thresholds: tag_score ≥0.22 top-8, yolo conf ≥0.4 iou 0.45,
  max_frames 12, timeouts, max_pixels).
- `src/service.rs:42-69` JobRequest + submit validation (`--vision-max`),
  `src/main.rs:124-144` ServeArgs / `:431-449` fetch-data extension.
- `src/ocr.rs` untouched.

## 5. Worker ownership map

| Worker | Owns exclusively |
|---|---|
| V1 plumbing | src/vision/mod.rs + types.rs (VisionMode enum, orchestrator API, per-tier trait/stubs), config.rs additions, JobRequest/serve/index/fetch-data wiring, store.rs vision DDL + upsert, pipeline.rs hooks (change detection + content append), submit pre-flight validation |
| V2 tier-meta | src/vision/{exif.rs, phash.rs, quality.rs} + their tests |
| V3 tier-tags | src/vision/{clip.rs, yolo.rs} + data/vision-tags.txt + their tests |
| V4 video | src/vision/video.rs (keyframes, aggregation, transcript merge point) + tests |
| V5 captions | src/vision/caption.rs (Florence-2 greedy; if genuinely not landable to passing-test quality, a clean `Err(unsupported)` stub + docs note — do NOT block the release on it) |
| V6 docs | README vision section, docs/VISION.md (usage, config, models+licenses, fetch-data, perf guidance CPU vs GPU, consumer compat notes) |

Stub-first discipline: V1 lands the full compiling skeleton (todo!()/no-op tiers)
before V2–V5 fill bodies; V2–V5 never touch shared files (mod.rs/types.rs owned by
V1; if an interface is wrong, adapt inside your own files and flag it).

## 6. Tests & acceptance

- Unit: pHash known-pairs (identical/resized→small hamming; unrelated→large), EXIF
  fixture parse, tag cosine scoring with stub vectors, NMS/letterbox math, keyframe
  selection logic, change-detection upgrade rule, submit-validation (--vision-max,
  missing models), decode-limit path.
- Model-dependent integration tests: `#[ignore]`-gated or skip-if-model-absent so
  CI stays green without downloads; a script/justfile target runs them locally
  after `fetch-data --vision`.
- Compat: existing test suite must pass untouched; add a test asserting a
  `vision:"off"` job on the fixture corpus produces the SAME rows as before the
  feature (guards the off-path).
- Smoke (on this box): fetch models, run a real `tags` job over generated fixture
  images + (if host ffmpeg exists) a synthesized clip; assert vision rows, fts
  block searchable via FTS MATCH 'vision', embedding blob length == 4*dimensions,
  phash dedup pair detected; then a `captions` job if V5 landed.
- Performance note for docs: CPU ~50–150 ms/image for tags; captions seconds/image
  on CPU — recommend GPU guidance doc (research report will cover RTX specifics).
