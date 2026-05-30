"""OCR backend (Tesseract). Pluggable and self-disabling if unavailable.

Language data is located via TESSDATA_PREFIX pointing at a user-writable folder
(``data/tessdata`` by default), so no admin write to Program Files is required.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    import pytesseract
except Exception:  # pragma: no cover
    pytesseract = None

from .config import REPO_ROOT


class TesseractOCR:
    def __init__(self, cfg):
        self.langs = cfg.get("ocr_langs", "vie+eng")
        self.available = False
        if pytesseract is None:
            return

        cmd = cfg.get("tesseract_cmd") or "tesseract"
        if cmd and Path(cmd).exists():
            pytesseract.pytesseract.tesseract_cmd = cmd

        td = cfg.get("tessdata_dir") or str(REPO_ROOT / "data" / "tessdata")
        tdp = Path(td)
        if tdp.exists() and any(tdp.glob("*.traineddata")):
            os.environ["TESSDATA_PREFIX"] = str(tdp)

        try:
            pytesseract.get_tesseract_version()
            self.available = True
        except Exception:
            self.available = False

    def image_to_text(self, image) -> str:
        if not self.available:
            return ""
        try:
            return pytesseract.image_to_string(
                image, lang=self.langs, config="--oem 1 --psm 3"
            )
        except Exception:
            return ""
