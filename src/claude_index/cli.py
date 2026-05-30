"""Command-line interface: index | search | top-folder | analyze."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from . import __version__
from .analyze import analyze as run_analyze
from .analyze import render_markdown
from .config import REPO_ROOT, Config
from .dictionaries import Dictionaries
from .extract import extract
from .lang import detect_lang
from .normalize import Normalizer
from .ocr import TesseractOCR
from .store import IndexStore, build_match, connect, search, top_folders
from .walker import walk


# ----------------------------- worker -----------------------------

def _sha1(path: str) -> str:
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _process(rec, cfg, ocr, dicts, normalizer, do_hash):
    ex = extract(rec.path, rec.ext, rec.size, cfg, ocr)
    text = ex["text"]
    name_blob = rec.name + " " + rec.dir
    content = text if text.strip() else name_blob
    token_src = " ".join((text, rec.name, rec.path.replace("\\", " ").replace("/", " ")))
    lang = detect_lang(text if text.strip() else rec.name, dicts)
    tokens = normalizer.enrich(token_src, lang)
    sha1 = _sha1(rec.path) if do_hash else None
    return (rec, content, tokens, lang, ex["method"], ex["ocr_used"], ex["pages"], sha1)


# ----------------------------- commands -----------------------------

def cmd_index(args):
    overrides = {
        "ocr": args.ocr, "sidecar": args.sidecar, "workers": args.workers,
        "max_bytes": args.max_bytes,
    }
    cfg = Config.load(args.config, overrides)
    out = Path(args.out)
    print(f"claude-index {__version__}  →  indexing {len(args.paths)} path(s) into {out}")
    dicts = Dictionaries(cfg)
    normalizer = Normalizer(dicts)
    ocr = TesseractOCR(cfg)
    print(f"  OCR: {'enabled (' + ocr.langs + ')' if ocr.available else 'unavailable'}"
          f"  ·  workers: {cfg['workers']}  ·  sidecar: {cfg['sidecar']}")

    counts = {"files": 0, "bytes": 0, "ocr": 0, "errors": 0, "skipped": 0}
    started = time.time()
    maxq = max(8, cfg["workers"] * 8)

    def handle(fut, store, pbar):
        rec, content, tokens, lang, method, ocr_used, pages, sha1 = fut.result()
        store.add(rec, content, tokens, lang, method, ocr_used, pages, sha1, time.time())
        counts["files"] += 1
        counts["bytes"] += rec.size
        counts["ocr"] += 1 if ocr_used else 0
        counts["errors"] += 1 if method.startswith("error") else 0
        pbar.update(1)

    with IndexStore(out, cfg, resume=args.resume) as store, \
            ThreadPoolExecutor(max_workers=cfg["workers"]) as pool, \
            tqdm(unit="file", desc="indexing") as pbar:
        seen = store.existing_keys() if args.resume else {}
        if args.resume:
            print(f"  resume: {len(seen):,} files already indexed — skipping unchanged")
        inflight = set()
        for rec in walk(args.paths, cfg):
            if seen:
                prev = seen.get(rec.path)
                if prev is not None and prev[0] == rec.size and prev[1] == int(rec.mtime):
                    counts["skipped"] += 1
                    pbar.update(1)
                    continue
            inflight.add(pool.submit(_process, rec, cfg, ocr, dicts, normalizer, cfg["hash"]))
            if len(inflight) >= maxq:
                done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
                for fut in done:
                    handle(fut, store, pbar)
        for fut in inflight:
            handle(fut, store, pbar)

    elapsed = time.time() - started
    rate = counts["files"] / elapsed if elapsed else 0
    print(f"\nIndexed {counts['files']:,} files ({counts['bytes'] / 1e9:.2f} GB) in "
          f"{elapsed:.1f}s ({rate:.0f} files/s).  OCR used on {counts['ocr']:,}; "
          f"errors {counts['errors']:,}.")
    if counts["skipped"]:
        print(f"Resume: skipped {counts['skipped']:,} files already indexed.")

    # analysis report
    db = connect(out)
    analysis = run_analyze(db)
    db.close()
    reports = out / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    label = ", ".join(args.paths)
    (reports / "analysis.md").write_text(render_markdown(analysis, label), encoding="utf-8")
    (reports / "analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"Analysis written: {reports / 'analysis.md'}")
    busiest = analysis.get("top_folders_by_count", [])
    if busiest:
        print("Densest folders (overall):")
        for folder, c in busiest[:5]:
            print(f"  {c:>6,}  {folder}")
    abs_out = out.resolve()
    print(f"\n📁 Index databases live in: {abs_out}")
    contents = "index.sqlite · manifest.jsonl · catalog.csv · reports/"
    if cfg["sidecar"] != "none":
        contents += f" · sidecar/ (Explorer-searchable .txt, {cfg['sidecar']} mode)"
    print(f"   {contents}")
    return 0


def _normalizer_for(args):
    cfg = Config.load(args.config, None)
    return Normalizer(Dictionaries(cfg))


def cmd_search(args):
    db = connect(args.index)
    nrm = _normalizer_for(args)
    rows = search(db, nrm, args.query, limit=args.limit, fuzzy=args.fuzzy)
    if not rows:
        print("No matches.")
        return 0
    for i, (path, d, lang, method, size, snip) in enumerate(rows, 1):
        print(f"{i:>2}. {path}")
        print(f"    [{lang}/{method}] {snip.strip()[:160]}")
    folders = top_folders(db, nrm, args.query, n=args.limit)
    if folders:
        best, c = folders[0]
        print(f"\n📁 Folder with most matches: {best}  ({c} match(es))")
        if len(folders) > 1:
            print("   Runners-up:")
            for folder, cnt in folders[1:6]:
                print(f"     {cnt:>4}  {folder}")
    db.close()
    return 0


def cmd_top_folder(args):
    db = connect(args.index)
    nrm = _normalizer_for(args)
    folders = top_folders(db, nrm, args.query, n=args.n)
    db.close()
    if not folders:
        print("No matches.")
        return 0
    for folder, c in folders:
        print(f"{c:>6}  {folder}")
    return 0


def cmd_analyze(args):
    db = connect(args.index)
    analysis = run_analyze(db)
    db.close()
    md = render_markdown(analysis, args.index)
    if args.json:
        Path(args.json).write_text(
            json.dumps(analysis, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    if args.md:
        Path(args.md).write_text(md, encoding="utf-8")
    print(md)
    return 0


# ----------------------------- parser -----------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="claude-index",
        description="Fast bilingual (EN/VI) full-text indexer with OCR and folder analytics.")
    p.add_argument("--version", action="version", version=f"claude-index {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    default_out = str(REPO_ROOT / "index_out")

    pi = sub.add_parser("index", help="walk paths and build the index")
    pi.add_argument("paths", nargs="+", help="drives/folders to index, e.g. E:\\")
    pi.add_argument("--out", default=default_out, help="output dir (default: ./index_out)")
    pi.add_argument("--ocr", choices=["auto", "on", "off"], default=None)
    pi.add_argument("--sidecar", choices=["mirror", "inplace", "none"], default=None,
                    help="Windows Explorer .txt sidecars (default: mirror)")
    pi.add_argument("--workers", type=int, default=None)
    pi.add_argument("--max-bytes", type=int, default=None, dest="max_bytes")
    pi.add_argument("--config", default=None)
    pi.add_argument("--resume", action="store_true",
                    help="continue a prior run: skip files already indexed (same size+mtime)")
    pi.set_defaults(func=cmd_index)

    ps = sub.add_parser("search", help="full-text search the index")
    ps.add_argument("query")
    ps.add_argument("--index", default=default_out)
    ps.add_argument("--limit", type=int, default=20)
    ps.add_argument("--fuzzy", action="store_true", help="fuzzy filename fallback")
    ps.add_argument("--config", default=None)
    ps.set_defaults(func=cmd_search)

    pt = sub.add_parser("top-folder", help="print folders with the most matches")
    pt.add_argument("query")
    pt.add_argument("--index", default=default_out)
    pt.add_argument("--n", type=int, default=10)
    pt.add_argument("--config", default=None)
    pt.set_defaults(func=cmd_top_folder)

    pa = sub.add_parser("analyze", help="structure / naming / type report")
    pa.add_argument("--index", default=default_out)
    pa.add_argument("--md", default=None, help="write Markdown report to file")
    pa.add_argument("--json", default=None, help="write JSON report to file")
    pa.set_defaults(func=cmd_analyze)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
