# Dev Handoff — claude-indexing

Concise state-of-the-world for the next developer/LLM. Pair with `docs/ARCHITECTURE.md`.

## Status: working v0.1
- End-to-end pipeline runs and is validated:
  - Smoke test (`tests/test_smoke.py`): EN stemming, VI diacritic-folding,
    abbreviation expansion, top-folder — all pass.
  - Real run: indexed **E:\ = 80,638 files / 15.9 GB** (99% Thunderbird
    `.wdseml` email). Search + busiest-folder + analysis verified.
  - OCR validated end-to-end (Tesseract `vie+eng`, best-quality traineddata).

## Environment (this machine: danga)
- Repo: `X:\GitHub\claude-indexing\`  ·  venv: `.venv\` (Python 3.11.9)
- Tesseract: `C:\Program Files\Tesseract-OCR\tesseract.exe` (v5.4).
  `vie`/`eng` traineddata live in `data\tessdata\` (pointed to via
  `TESSDATA_PREFIX` — **no admin write to Program Files needed**).
- GitHub CLI installed at `C:\Program Files\GitHub CLI\gh.exe`.

## Architecture (module map)
```
walker.py     os.scandir recursion; skip lists; reparse-point loop guard.
extract.py    dispatch by extension -> text/code, email(.wdseml/.eml),
              docx/xlsx/pptx, pdf(text|OCR), image(OCR). Returns text+method.
ocr.py        Tesseract backend (pluggable, self-disables if missing).
lang.py       EN/VI detection via diacritic density + dict coverage.
dictionaries.py  Hunspell(spylls) known/lemma, Snowball stem, VI max-match
                 segmentation (Viet74K), abbreviation maps. Degrades gracefully.
normalize.py  token enrichment: raw + folded + EN stems + VI compounds(_) + abbrev.
store.py      sinks: SQLite FTS5 (+JSONL/CSV/sidecar); readers: search, top_folders.
analyze.py    structure/type/naming/language report (dict + Markdown).
cli.py        index | search | top-folder | analyze (argparse, ThreadPool).
```
Flow: `walk → extract(+OCR) → detect_lang → normalize.enrich → store.add`.
Workers extract+normalize; the main thread is the sole SQLite writer (no lock
contention). Inflight futures bounded to `workers*8` for steady memory.

## Key decisions / tradeoffs
- **Python** (not Rust/Go): richest OCR/NLP ecosystem and lowest dev cost; speed
  recovered via SQLite FTS5 (C) + threads + bounded queue.
- **SQLite FTS5 = primary index**: zero-dependency, SQL-queryable by "any
  engine", `unicode61 remove_diacritics 2 tokenchars '_'` gives free
  diacritic-folding and keeps VI compounds as single tokens. Optional Tantivy
  backend is noted but not required.
- **Dictionary max-matching for Vietnamese** instead of ML segmenters (pyvi/
  underthesea pull scikit-learn) — keeps install light and uses the "richest
  dictionary" directly.
- **Sidecar `.txt`** is how Windows Explorer becomes searchable over scanned/
  binary content. **Default `mirror`** (parallel `.txt` tree under the output
  dir; source drives stay clean); `inplace` writes next to each source file;
  `none` skips (fewer inodes on huge mail trees).

## Data dependencies (fetched, NOT committed — see `.gitignore`)
`scripts/fetch_data.py` (idempotent) downloads into `data/`:
- `dict/en_US.{dic,aff}`, `dict/vi_VN.{dic,aff}` — Hunspell (wooorm/dictionaries).
- `dict/vi_words.txt` — Viet74K compound lexicon.
- `tessdata/{vie,eng}.traineddata` — Tesseract best.

## Known limitations / TODO
- Legacy binary formats not parsed: `.doc/.xls/.ppt`, `.msg` (Outlook) → name-only.
  Add via `olefile`/`textract`/`extract-msg`.
- Email **attachment images aren't OCR'd** yet (text parts only). Hook into
  `_email()` to rasterize/OCR `image/*` parts.
- No **incremental re-index**: `add()` uses `INSERT OR REPLACE` on path, but the
  walker re-extracts everything. Add an mtime/size skip check against `files`.
- VI lemma normalization is segmentation-only (VI is largely non-inflecting) —
  fine, but synonym expansion (WordNet/Vietnamese WordNet) is a future lever.
- Optional **Tantivy** backend + a tiny query web UI are unimplemented.

## Run tests
```powershell
.\.venv\Scripts\python tests\test_smoke.py        # or: python -m pytest -q
```

## Re-index this machine's test drive
```powershell
.\.venv\Scripts\claude-index index E:\ --out index_out --ocr auto
.\.venv\Scripts\claude-index search "ngan hang" --index index_out
```
