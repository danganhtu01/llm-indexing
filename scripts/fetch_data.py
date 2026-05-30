#!/usr/bin/env python3
"""Download the English/Vietnamese dictionaries, the Vietnamese compound
wordlist, and Tesseract best-quality language data. Idempotent: skips files
that already exist (use --force to re-download). Cross-platform, stdlib only.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DICT = ROOT / "data" / "dict"
TESS = ROOT / "data" / "tessdata"

RAW = "https://raw.githubusercontent.com"
FILES = {
    DICT / "en_US.dic": f"{RAW}/wooorm/dictionaries/main/dictionaries/en/index.dic",
    DICT / "en_US.aff": f"{RAW}/wooorm/dictionaries/main/dictionaries/en/index.aff",
    DICT / "vi_VN.dic": f"{RAW}/wooorm/dictionaries/main/dictionaries/vi/index.dic",
    DICT / "vi_VN.aff": f"{RAW}/wooorm/dictionaries/main/dictionaries/vi/index.aff",
    DICT / "vi_words.txt": f"{RAW}/duyet/vietnamese-wordlist/master/Viet74K.txt",
    TESS / "vie.traineddata": f"{RAW}/tesseract-ocr/tessdata_best/main/vie.traineddata",
    TESS / "eng.traineddata": f"{RAW}/tesseract-ocr/tessdata_best/main/eng.traineddata",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()
    DICT.mkdir(parents=True, exist_ok=True)
    TESS.mkdir(parents=True, exist_ok=True)
    ok = True
    for dst, url in FILES.items():
        if dst.exists() and not args.force:
            print(f"skip  {dst.name} (exists)")
            continue
        try:
            print(f"get   {dst.name} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dst)
            print(f"{dst.stat().st_size // 1024} KB")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"FAIL ({e})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
