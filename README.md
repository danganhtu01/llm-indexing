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
src/claude_index/   walker · extract · ocr · lang · normalize · dictionaries · store · analyze · cli
data/               abbreviations_*.txt (committed); dict/ + tessdata/ (fetched, gitignored)
scripts/            install_windows.ps1 · fetch_data.py
index-all-drives.ps1   overnight all-drives runner (path-independent; auto-resume)
docs/ARCHITECTURE.md   pipeline + schema + extension points
tests/test_smoke.py    end-to-end EN/VI/abbrev/top-folder check
```

## Status
v0.1 — validated on a real 80K-file drive (Thunderbird email backup) and on
EN+VI OCR. See `docs/ARCHITECTURE.md` for architecture and design decisions.

MIT licensed.
