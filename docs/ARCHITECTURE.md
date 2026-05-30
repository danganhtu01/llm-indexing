# Architecture

## Pipeline
```
paths ─► walker.walk ─► extract.extract ─► lang.detect_lang ─► normalize.enrich ─► store.add ─► sinks
            │                │                                                          │
       skip lists,      per-type text                                          SQLite FTS5 (primary)
       reparse guard    + OCR fallback                                         JSONL · CSV · sidecar .txt
```
- **walker** — iterative `os.scandir` (low memory), honors `skip_dirs`/`skip_exts`,
  never follows reparse points (junction/symlink loop guard). Yields `FileRec`.
- **extract** — extension dispatch. Text/code (charset-detected), email
  (`.wdseml/.eml`: headers + plain/HTML body + attachment names), Office
  (docx/xlsx/pptx), PDF (text layer; OCR pages when the layer is empty), images
  (OCR). Always falls back to indexing name + path tokens.
- **lang** — samples ~2 KB; Vietnamese diacritic density + Hunspell coverage →
  `vi | en | mixed | und`.
- **normalize** — builds the high-recall token bag (see below).
- **store** — writes all sinks; provides `search` / `top_folders` readers.

## Token enrichment (the core of recall)
For each document `enrich(text, lang)` emits a de-duplicated bag combining:
1. **raw words** (Unicode word runs, lowercased);
2. **diacritic-folded** variants (`fold`: NFD strip + đ→d) — diacritic-insensitive;
3. **English stems/lemmas** — Hunspell lemma if known (handles irregulars), else
   Snowball stem;
4. **Vietnamese compounds** — forward maximum-matching against the Viet74K
   lexicon, joined with `_` (e.g. `ngân_hàng`) plus a folded copy (`ngan_hang`);
5. **abbreviation expansions** — from `data/abbreviations_{en,vi}.txt`.

The bag goes into the FTS `tokens` column; the original text goes into `content`
(for phrase search + snippets). Queries are normalized the same way and OR-ed,
then ranked by `bm25()`.

## Index schema (`index.sqlite`)
```sql
files(id, path UNIQUE, drive, dir, name, ext, size, mtime,
      lang, method, ocr_used, pages, chars, sha1, indexed_at)
fts  USING fts5(name, path, content, tokens,
                tokenize="unicode61 remove_diacritics 2 tokenchars '_'")
-- fts.rowid == files.id
```
`tokenchars '_'` keeps Vietnamese compounds as single tokens; `remove_diacritics 2`
makes even the raw `content` column diacritic-insensitive at query time.

Example raw query:
```sql
SELECT f.path, snippet(fts,2,'[',']','…',12)
FROM fts JOIN files f ON f.id = fts.rowid
WHERE fts MATCH '"ngan" OR "hang" OR "ngan_hang"'
ORDER BY bm25(fts) LIMIT 20;
```

## Folder-with-most-matches
`top_folders` runs the same MATCH, `GROUP BY files.dir`, `ORDER BY COUNT(*) DESC`.
`search`/`top-folder` CLI print the winning folder and runners-up.

## Extension points
- **New file type** — add the extension to a set in `extract.py` and a small
  `_handler(path, max_chars)`; route it in `extract()`.
- **New OCR backend** — implement a class with `available` + `image_to_text(img)`
  (mirror `ocr.TesseractOCR`); e.g. RapidOCR/EasyOCR. Select it in `cli`.
- **New language** — add a Hunspell dict + (optional) wordlist in `fetch_data.py`,
  extend `lang.detect_lang` and `normalize.enrich` branches.
- **Alternate index** — Tantivy/Elasticsearch can be fed directly from
  `manifest.jsonl`; or add a sink alongside `IndexStore`.

## Performance notes
- Extraction is threaded (`--workers`, default 8); SQLite writes are serialized
  on the main thread; commits batch every 500 rows.
- OCR is the costliest step and only runs on images or text-less PDF pages
  (`ocr: auto`). Cap pages with `ocr_max_pages`.
- Memory is bounded: streaming walk + inflight cap (`workers*8`) + per-file
  `max_chars` (default 1M) and `max_bytes` (default 100 MB → name-only).
