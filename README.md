# llm-indexing

Rust-native document indexing service for English and Vietnamese corpora. It
walks mounted directories, extracts text, OCRs images and text-less PDFs, and
publishes a portable SQLite FTS5 database.

The application code is exclusively Rust. SQLite is embedded through
`rusqlite`; Tesseract and Poppler are runtime executables used for OCR and PDF
processing. PostgreSQL is not required for this self-contained component.

## Capabilities

- Text and source files with automatic character-set detection
- PDF text extraction with OCR fallback
- Tesseract OCR for images and scanned PDFs (`vie+eng` by default)
- DOCX, XLSX, PPTX, EML, Wdseml, and Emlx extraction
- English stemming, Vietnamese compound segmentation, abbreviation expansion,
  and diacritic-insensitive matching
- SQLite FTS5 output, resumable indexing, ranked search, folder aggregation,
  JSON/Markdown analysis, JSONL/CSV manifests, and optional sidecars
- Bounded asynchronous HTTP job queue with input/output path confinement

Russian and other languages can be added by installing the corresponding
Tesseract language package and extending normalization data. The request
contract already accepts an `ocr_langs` value such as `rus+eng`.

## Container

```bash
mkdir -p input output
sudo chown 10001:10001 output
docker compose up --build -d

# Submit a job and wait for completion.
docker compose exec indexing llm-index request \
  --url http://127.0.0.1:9801 \
  --output corpus.sqlite \
  --overwrite

# Query the generated database without a running service.
docker run --rm -v "$PWD/output:/output:ro" llm-indexing:rust \
  search "ngan hang" --index /output/corpus.sqlite
```

Set `INDEX_INPUT` and `INDEX_OUTPUT` to host directories before starting
Compose. Input is mounted read-only; the service writes only plain `.sqlite`
filenames below `/output`. The HTTP port binds to localhost by default.

### HTTP contract

`GET /health` returns service readiness. `POST /index` queues a job:

```json
{
  "paths": ["/input"],
  "output": "corpus.sqlite",
  "ocr": "auto",
  "ocr_langs": "vie+eng",
  "workers": 4,
  "resume": false,
  "overwrite": true
}
```

The response is `202 Accepted` with an `id`. Poll `GET /jobs/{id}` until the
status is `complete` or `error`. A completed job publishes the SQLite file only
after indexing succeeds.

## Native CLI

Requires Rust 1.85+, Tesseract with the desired language data, and Poppler.

```bash
cargo build --release --locked
./target/release/llm-index fetch-data --data-dir data --dictionaries-only

llm-index index ./documents --out index_out --ocr auto --ocr-langs vie+eng
llm-index index ./documents --out index_out --resume
llm-index search "know your customer" --index index_out
llm-index top-folder "hoa don" --index index_out --limit 10
llm-index analyze --index index_out --json analysis.json
```

Copy `config.example.yaml` and pass `--config config.yaml` to override extraction,
OCR, worker, size, sidecar, and skip settings.

## Output

Native `index` mode creates `index.sqlite`, `manifest.jsonl`, `catalog.csv`,
analysis reports, and optional text sidecars. Service mode intentionally
publishes only the requested SQLite database, making the container artifact
stable and easy for another application to consume or archive.

MIT licensed.
