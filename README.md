# claude-indexing

Fast, low-cost **bilingual (English + Vietnamese) full-text indexer** for large
drives. Walks a tree, extracts text from documents/spreadsheets/slides/PDFs/**email**
and **OCRs scans & images**, normalizes with up-to-date EN/VI dictionaries
(word forms + abbreviations), and writes **engine-agnostic indexes** that any
search tool — including Windows File Explorer — can read. It also reports the
**folder with the most matches** and a structural analysis of the tree.

## Why it's fast / low-cost
- Python 3.10+ with **SQLite FTS5** (built in — no server, no license) as the
  primary index. Rust-backed `rapidfuzz` for fuzzy fallback.
- Threaded extraction; OCR only runs when a page has no text layer.
- Pure-Python language tooling (Hunspell via `spylls`, Snowball, dictionary
  max-matching for Vietnamese) — no heavy ML models.

## Outputs (all open, portable formats)
| File | Purpose |
|---|---|
| `index.sqlite` | **Primary** FTS5 full-text index. Query with SQL + `bm25()` from any language. |
| `manifest.jsonl` | One JSON object per file — drop into Elasticsearch/OpenSearch/pandas. |
| `catalog.csv` | Flat catalog — open in Excel or search in Explorer. |
| `sidecar/**/*.txt` | Extracted/OCR text mirrored per file so **Windows Explorer content-search** finds scanned & binary docs. |
| `reports/analysis.md` + `.json` | Structure, file types, naming conventions, language mix, busiest folders. |

## Search richness (EN + VI)
- **Diacritic-insensitive**: `ngan hang` matches `ngân hàng` (FTS folds + folded tokens).
- **English word forms**: `launder` matches `laundering`, `laundered` (Hunspell lemma → Snowball stem).
- **Vietnamese compounds**: maximum-matching over a 74K-word lexicon indexes
  `ngân_hàng`, `giao_dịch` as single terms while keeping the syllables.
- **Abbreviations**: `KYC` ⇄ `know your customer`, `NH` → `ngân hàng`
  (editable lists in `data/abbreviations_*.txt`).

## Install (Windows)
```powershell
# one-shot: Tesseract + GitHub CLI + venv + deps + dictionaries/OCR data
powershell -ExecutionPolicy Bypass -File scripts\install_windows.ps1
```
Manual:
```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python scripts\fetch_data.py     # dictionaries + vie/eng traineddata
winget install --id UB-Mannheim.TesseractOCR -e  # OCR engine
```

## Usage
```powershell
# Index a drive (auto-OCR scans; sidecar defaults to mirror = Explorer-searchable)
claude-index index E:\ --out index_out --ocr auto

# Resume an interrupted run — re-walks, skips files already indexed (same size+mtime)
claude-index index E:\ --out index_out --ocr auto --resume

# Full-text search; prints ranked hits + the folder with the most matches
claude-index search "ngan hang giao dich" --index index_out

# Just the densest-match folders
claude-index top-folder "hoa don" --index index_out --n 10

# Structure / naming / type report
claude-index analyze --index index_out --md report.md

# Index ALL plugged-in drives at once (overnight; auto-resumes on crash/interruption)
.\index-all-drives.ps1            # -> %SystemDrive%\index_out  (override: -Out 'D:\idx')
```

## Docker socket service

Like `llm-redaction`, the container accepts jobs over a resident HTTP socket and
atomically publishes a portable SQLite FTS5 corpus. The indexer initiates no
runtime network connections and mounts source data read-only.

```bash
mkdir -p input output
docker compose build
docker compose up -d

# Index all mounted input directories into ./output/corpus.sqlite.
docker compose exec indexer claude-index-request \
  --url http://127.0.0.1:9801 --path /input --output corpus.sqlite --ocr auto
```

The request returns after `GET /jobs/{id}` reaches `complete`; use `--no-wait`
to keep job polling in the caller. The finished database appears in `./output`.

Set `INDEX_INPUT=/absolute/source` and `INDEX_OUTPUT=/absolute/output` in the
environment or a `.env` file to mount other host directories. The default image
ships Tesseract OCR for Vietnamese and English (`vie+eng`). To add Russian:

```bash
TESSERACT_LANG_PACKAGES="tesseract-ocr-eng tesseract-ocr-vie tesseract-ocr-rus" \
OCR_LANGS="vie+eng+rus" docker compose build
```

The service binds `127.0.0.1:9801` by default. Other containers can call
`http://indexer:9801` when attached to the same internal Docker network. See
[`docs/SOCKET-PROTOCOL.md`](docs/SOCKET-PROTOCOL.md) for the asynchronous API.

Sidecars default to **`mirror`** — a parallel `.txt` tree under the output dir
that Windows Explorer can content-search (source drives stay untouched). Use
`--sidecar inplace` to write text next to each source file, or `--sidecar none`
to skip (faster on huge mail trees). The mirror tree lives at
`index_out\sidecar\<drive>\` (one subfolder per drive); add the `index_out\sidecar`
folder to Windows *Indexing Options* to content-search it from Explorer.

## Configuration
Copy `config.example.yaml` → `config.yaml` (auto-loaded) or pass `--config`.
Controls languages, OCR mode, skip lists, size caps, workers, Tesseract paths,
dictionary/abbreviation paths. See comments in that file.

## Repo layout
```
src/claude_index/   walker · extract · ocr · normalize · store · socket server/client · cli
data/               abbreviations_*.txt (committed); dict/ + tessdata/ (fetched, gitignored)
scripts/            install_windows.ps1 · fetch_data.py
index-all-drives.ps1   overnight all-drives runner (path-independent; auto-resume)
docs/               architecture, SQLite schema, and socket protocol
tests/              EN/VI search smoke test + socket integration test
```

## Status
v0.2 — containerized HTTP job service; the indexing core was previously validated
on a real 80K-file drive (Thunderbird email backup) and on
EN+VI OCR. See `docs/ARCHITECTURE.md` for architecture and design decisions.

MIT licensed.
