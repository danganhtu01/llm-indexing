"""Configuration: baked-in defaults, optional YAML override, path resolution."""
from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../claude-indexing

DEFAULTS = {
    "languages": ["vi", "en"],
    "ocr": "auto",            # auto | on | off
    "sidecar": "mirror",      # mirror | inplace | none
    "workers": 8,
    "max_bytes": 100 * 1024 * 1024,
    "max_chars": 1_000_000,
    "hash": False,
    "ocr_max_pages": 20,
    "tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "tessdata_dir": "",
    "ocr_langs": "vie+eng",
    "dict_dir": "data/dict",
    "abbreviations": ["data/abbreviations_en.txt", "data/abbreviations_vi.txt"],
    "skip_dirs": [
        "$RECYCLE.BIN", "System Volume Information", ".git",
        "$WinREAgent", "Windows", "node_modules",
        # never index our own output or Python virtualenvs / caches
        "index_out", ".venv", "venv", "site-packages", "__pycache__",
    ],
    "skip_exts": [".sys", ".dll", ".exe", ".iso", ".vmdk", ".lock"],
    "follow_symlinks": False,
}


class Config(dict):
    """A dict of settings with helpers for path resolution and skip lookups."""

    @classmethod
    def load(cls, path: str | None = None, overrides: dict | None = None) -> "Config":
        cfg = dict(DEFAULTS)
        chosen = Path(path) if path else (REPO_ROOT / "config.yaml")
        if chosen.exists():
            with open(chosen, "r", encoding="utf-8") as f:
                cfg.update(yaml.safe_load(f) or {})
        if overrides:
            cfg.update({k: v for k, v in overrides.items() if v is not None})
        c = cls(cfg)
        c._finalize()
        return c

    def _finalize(self) -> None:
        self["dict_dir"] = str(self._abs(self["dict_dir"]))
        self["abbreviations"] = [str(self._abs(p)) for p in self["abbreviations"]]
        if self.get("tessdata_dir"):
            self["tessdata_dir"] = str(self._abs(self["tessdata_dir"]))
        self["_skip_dirs_upper"] = {d.upper() for d in self["skip_dirs"]}
        self["_skip_exts_lower"] = {e.lower() for e in self["skip_exts"]}

    @staticmethod
    def _abs(p) -> Path:
        p = Path(p)
        return p if p.is_absolute() else (REPO_ROOT / p)

    def skip_dir(self, name: str) -> bool:
        return name.upper() in self["_skip_dirs_upper"]

    def skip_ext(self, ext: str) -> bool:
        return ext.lower() in self["_skip_exts_lower"]
