# llm-indexing
Rust-native English/Vietnamese content indexer. It walks mounted folder trees,
extracts and OCRs documents, transcribes audio/video locally with Whisper, and
writes a portable SQLite database containing FTS5 text plus multilingual vector
embeddings.

The current release is `0.4.0` and requires Rust 1.88+. SQLite, Whisper and
FastEmbed are embedded in the service. Poppler, Tesseract, FFmpeg, antiword and
libarchive are runtime tools inside the container; PostgreSQL is not required by
this component.

## Capabilities

- Plain text, source files, EML/Emlx/Wdseml and legacy `.doc`
- PDF text extraction plus page-by-page raster OCR
- Images and embedded Office/ODF media OCR
- DOCX, XLSX, PPTX and OpenDocument XML text extraction
- ZIP, RAR, 7z, TAR and gzip traversal with safe-path and recursion limits
  (including UTF-8/Vietnamese archive entry names)
- Local Whisper transcription for audio and video; video frame OCR is sampled
  in exhaustive mode
- English stemming, Vietnamese segmentation, abbreviation expansion and
  diacritic-insensitive FTS5 search
- 384-dimensional `multilingual-e5-small` embeddings and cosine vector search
- Resume/change detection, removal pruning, authentic incomplete/error counts,
  folder aggregation, manifests, reports and optional sidecars
- Bounded HTTP job queue with input/output path confinement, live per-file
  counters and cooperative cancellation
- Optional local vision analysis of photos/video (EXIF, perceptual hash,
  quality, CLIP tags, object detection, best-effort captions), off by default
  — see [Vision (photos & video)](#vision-photos--video)

`ocr: exhaustive` removes the normal byte, character and PDF-page caps. It OCRs
every PDF page even when a text layer exists, inspects embedded Office images,
and processes media/archives. Empty, failed or partial extraction is explicitly
marked incomplete; it is never silently represented as a successfully indexed
filename. Office lock files (`~$…`) remain deliberate exclusions.

## Container

The image bakes in the pinned Whisper small multilingual model and the FastEmbed
model, so indexing and vector retrieval need no internet access at runtime.

```bash
mkdir -p input output
sudo chown 10001:10001 output
docker compose build --network host
docker compose up -d

docker compose exec indexing llm-index request \
  --url http://127.0.0.1:9801 \
  --output corpus.sqlite \
  --ocr exhaustive \
  --resume \
  --overwrite

docker run --rm -v "$PWD/output:/output:ro" llm-indexing:rust \
  vector-search "discussion about payment controls" \
  --index /output/corpus.sqlite
```

Set `INDEX_INPUT` and `INDEX_OUTPUT` before starting Compose. Input is read-only;
the service writes only a plain `.sqlite` filename under `/output`. Port 9801 is
bound to localhost by the standalone Compose file and is internal-only in the
`ff-lc-app` deployment.

## Native CLI

```bash
cargo build --release --locked
./target/release/llm-index prefetch-models --config config.yaml
./target/release/llm-index index ./documents --out index_out \
  --ocr exhaustive --ocr-langs vie+eng --resume
./target/release/llm-index search "know your customer" --index index_out
./target/release/llm-index vector-search "customer due diligence" \
  --index index_out/index.sqlite
./target/release/llm-index top-folder "hoa don" --index index_out --limit 10
```

Copy `config.example.yaml` to override OCR, extraction, Whisper, embedding,
worker, sidecar and skip settings.

## Vision (photos & video)

Local, offline computer-vision understanding of photos and video — EXIF,
perceptual hash and quality metrics (`meta`), CLIP zero-shot tags plus object
detection (`tags`), and a best-effort caption (`captions`) — written into a
new `vision` table and appended to each file's searchable FTS content. No
network access at index time; models are fetched only via an explicit,
pinned-checksum step. Default `off` everywhere: existing databases, `method`
values and FTS content are byte-identical until a caller opts in.

```bash
./target/release/llm-index fetch-data --vision           # one-time, pinned SHA-256
./target/release/llm-index index ./photos --out index_out --vision tags
./target/release/llm-index serve --vision-max tags        # allow jobs up to `tags`
```

Full tier reference, config knobs (`--vision`, `--vision-max` /
`INDEX_VISION_MAX`, `VisionConfig`), the model/license table, consumer
compatibility notes and performance/security details:
[`docs/VISION.md`](docs/VISION.md).

## Output and incremental behavior

Native mode writes `index.sqlite`, `manifest.jsonl`, `catalog.csv`, analysis
reports and optional sidecars. Service mode publishes only the requested SQLite
database after a successful job.

Resume uses path, size and mtime and also repairs records missing vectors or
marked partial/error. It reprocesses older PDF methods when exhaustive OCR is
requested. Source files removed from the mounted tree are pruned from `files`,
FTS and vector chunks. The job result reports `incomplete`, `embedded_chunks`
and `removed`, allowing callers to show authentic state.

Service callers may additionally provide a confined `include_paths` list of
relative file paths. The engine still scans the mounted tree to prune deletions,
but extraction, OCR, transcription and embedding are restricted to exactly that
list. This is how `ff-lc-app` guarantees that a button press processes only its
database-derived new, changed or incomplete rows.

MIT licensed.
