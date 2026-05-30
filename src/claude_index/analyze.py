"""Structural analysis over an index: types, sizes, languages, extraction
coverage, folder hot-spots, and filename naming conventions.

Pure SQL + Python over the ``files`` table; emits a dict plus a Markdown report.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
import statistics
from collections import Counter

from .extract import category_for
from .lang import VN_DIACRITIC_RE

_PASCAL = re.compile(r"^([A-Z][a-z0-9]+){2,}$")
_CAMEL = re.compile(r"^[a-z]+([A-Z][a-z0-9]*)+$")
_SNAKE = re.compile(r"^[a-z0-9]+(_[a-z0-9]+)+$")
_KEBAB = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)+$")
_DATE = re.compile(r"(19|20)\d{2}[-_.]?(0[1-9]|1[0-2])|"
                   r"\b(0[1-9]|[12]\d|3[01])[-_.](0[1-9]|1[0-2])[-_.](19|20)?\d{2}\b")
_NAME_TOK = re.compile(r"[0-9A-Za-zÀ-ỹ]+")


def _method_group(m: str) -> str:
    if m.startswith("error"):
        return "error"
    if m in ("docx", "xlsx", "pptx"):
        return "office"
    if m.startswith("pdf"):
        return "pdf"
    if m == "ocr":
        return "ocr"
    if m == "email":
        return "email"
    if m == "text":
        return "text"
    return "name-only"


def analyze(db) -> dict:
    rows = db.execute(
        "SELECT path, dir, name, ext, size, mtime, lang, method, ocr_used FROM files"
    ).fetchall()
    n = len(rows)
    res: dict = {"totals": {"files": n, "folders": 0, "drives": 0, "bytes": 0}}
    if n == 0:
        return res

    cat_count, cat_bytes = Counter(), Counter()
    ext_count, ext_bytes = Counter(), Counter()
    dir_count, dir_bytes = Counter(), Counter()
    lang_count, method_count = Counter(), Counter()
    name_tokens, base_count, year_count, naming = Counter(), Counter(), Counter(), Counter()
    drives, sizes, mtimes, depths = set(), [], [], []
    largest, ocr_used, vn_names, name_len_total = [], 0, 0, 0

    for path, d, name, ext, size, mtime, lang, method, ocr in rows:
        size = size or 0
        cat = category_for(ext)
        cat_count[cat] += 1; cat_bytes[cat] += size
        ext_count[ext or "(noext)"] += 1; ext_bytes[ext or "(noext)"] += size
        dir_count[d] += 1; dir_bytes[d] += size
        drives.add(os.path.splitdrive(path)[0])
        sizes.append(size); largest.append((size, path))
        if mtime:
            mtimes.append(mtime)
            year_count[_dt.datetime.fromtimestamp(mtime).year] += 1
        lang_count[lang] += 1
        method_count[_method_group(method)] += 1
        ocr_used += 1 if ocr else 0
        base = os.path.splitext(name)[0]
        base_count[name] += 1
        name_len_total += len(name)
        depths.append(path.replace("/", "\\").count("\\"))
        for tok in _NAME_TOK.findall(base):
            if len(tok) >= 2:
                name_tokens[tok.lower()] += 1
        if " " in name: naming["space"] += 1
        if _SNAKE.match(base): naming["snake"] += 1
        if _CAMEL.match(base): naming["camel"] += 1
        if _PASCAL.match(base): naming["pascal"] += 1
        if _KEBAB.match(base): naming["kebab"] += 1
        if base.isascii() and base.isupper() and len(base) > 1: naming["allcaps"] += 1
        if _DATE.search(name): naming["date"] += 1
        if VN_DIACRITIC_RE.search(name): vn_names += 1

    largest.sort(reverse=True)
    pct = lambda x: round(100.0 * x / n, 1)
    res["totals"] = {"files": n, "folders": len(dir_count), "drives": len(drives),
                     "bytes": sum(sizes)}
    res["by_category"] = [(c, cnt, cat_bytes[c]) for c, cnt in cat_count.most_common()]
    res["by_ext"] = [(e, cnt, ext_bytes[e]) for e, cnt in ext_count.most_common(25)]
    res["sizes"] = {"total_bytes": sum(sizes), "mean": int(statistics.mean(sizes)),
                    "median": int(statistics.median(sizes)),
                    "largest": [(p, s) for s, p in largest[:10]]}
    res["times"] = {
        "oldest": min(mtimes) if mtimes else None,
        "newest": max(mtimes) if mtimes else None,
        "by_year": sorted(year_count.items()),
    }
    res["languages"] = dict(lang_count)
    res["extraction"] = {**dict(method_count), "ocr_used": ocr_used}
    res["depth"] = {"max": max(depths), "avg": round(statistics.mean(depths), 1)}
    res["top_folders_by_count"] = dir_count.most_common(15)
    res["top_folders_by_size"] = dir_bytes.most_common(15)
    res["naming"] = {
        "space_pct": pct(naming["space"]), "snake_pct": pct(naming["snake"]),
        "camel_pct": pct(naming["camel"]), "pascal_pct": pct(naming["pascal"]),
        "kebab_pct": pct(naming["kebab"]), "allcaps_pct": pct(naming["allcaps"]),
        "date_pct": pct(naming["date"]), "vietnamese_pct": pct(vn_names),
        "avg_name_len": round(name_len_total / n, 1),
        "common_tokens": name_tokens.most_common(20),
        "duplicate_basenames": [(k, v) for k, v in base_count.most_common(15) if v > 1],
    }
    return res


def _hsize(b: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if b < 1024 or unit == "TB":
            return f"{b:.1f} {unit}"
        b /= 1024


def _ts(t):
    return _dt.datetime.fromtimestamp(t).strftime("%Y-%m-%d") if t else "—"


def render_markdown(a: dict, label: str = "") -> str:
    t = a["totals"]
    if t["files"] == 0:
        return "# Index analysis\n\n_No files indexed._\n"
    L = [f"# Index analysis{(' — ' + label) if label else ''}", ""]
    L.append(f"- **Files:** {t['files']:,}  ·  **Folders:** {t['folders']:,}  ·  "
             f"**Drives:** {t['drives']}  ·  **Total size:** {_hsize(t['bytes'])}")
    d = a["depth"]
    L.append(f"- **Folder depth:** max {d['max']}, avg {d['avg']}")
    tm = a["times"]
    L.append(f"- **Modified range:** {_ts(tm['oldest'])} → {_ts(tm['newest'])}")
    lg = a["languages"]
    L.append(f"- **Language mix:** " + ", ".join(f"{k}={v:,}" for k, v in lg.items()))
    ex = a["extraction"]
    L.append(f"- **Extraction:** " + ", ".join(f"{k}={v:,}" for k, v in ex.items()))
    L.append("")

    L.append("## File types (by category)")
    L.append("| category | files | size |")
    L.append("|---|--:|--:|")
    for c, cnt, b in a["by_category"]:
        L.append(f"| {c} | {cnt:,} | {_hsize(b)} |")
    L.append("")

    L.append("## Top extensions")
    L.append("| ext | files | size |")
    L.append("|---|--:|--:|")
    for e, cnt, b in a["by_ext"][:15]:
        L.append(f"| {e} | {cnt:,} | {_hsize(b)} |")
    L.append("")

    L.append("## Busiest folders (by file count)")
    L.append("| folder | files |")
    L.append("|---|--:|")
    for folder, cnt in a["top_folders_by_count"][:10]:
        L.append(f"| {folder} | {cnt:,} |")
    L.append("")

    L.append("## Largest files")
    L.append("| size | path |")
    L.append("|--:|---|")
    for p, s in a["sizes"]["largest"]:
        L.append(f"| {_hsize(s)} | {p} |")
    L.append("")

    nm = a["naming"]
    L.append("## Naming conventions")
    L.append(f"- avg name length: **{nm['avg_name_len']}** chars")
    L.append(f"- contains spaces: **{nm['space_pct']}%**  ·  snake_case: **{nm['snake_pct']}%**  "
             f"·  camelCase: **{nm['camel_pct']}%**  ·  PascalCase: **{nm['pascal_pct']}%**  "
             f"·  kebab-case: **{nm['kebab_pct']}%**  ·  ALLCAPS: **{nm['allcaps_pct']}%**")
    L.append(f"- date in name: **{nm['date_pct']}%**  ·  Vietnamese diacritics in name: "
             f"**{nm['vietnamese_pct']}%**")
    if nm["common_tokens"]:
        L.append("- most common name tokens: " +
                 ", ".join(f"`{w}`({c})" for w, c in nm["common_tokens"][:15]))
    if nm["duplicate_basenames"]:
        L.append("- frequent duplicate filenames: " +
                 ", ".join(f"`{w}`×{c}" for w, c in nm["duplicate_basenames"][:10]))
    L.append("")
    return "\n".join(L)
