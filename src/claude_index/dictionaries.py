"""EN/VI dictionaries: known-word checks, English lemmas/stems, Vietnamese
compound segmentation (maximum-matching), and abbreviation expansion.

Every loader degrades gracefully: if a data file is missing the matching
capability becomes a no-op, so the indexer still runs without downloads.
"""
from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path

import snowballstemmer

try:
    from spylls.hunspell import Dictionary as _Hunspell
except Exception:  # pragma: no cover
    _Hunspell = None

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


class Dictionaries:
    def __init__(self, cfg):
        d = Path(cfg["dict_dir"])
        self._en = self._load_hunspell(d / "en_US")
        self._vi = self._load_hunspell(d / "vi_VN")
        self._snow = snowballstemmer.stemmer("english")
        self.vi_vocab, self.vi_max_len = self._load_vi_words(d / "vi_words.txt")
        self.abbr = self._load_abbrev(cfg.get("abbreviations", []))

    # ---- loaders ----
    @staticmethod
    def _load_hunspell(prefix: Path):
        if _Hunspell is None or not prefix.with_suffix(".dic").exists():
            return None
        try:
            return _Hunspell.from_files(str(prefix))
        except Exception:
            return None

    @staticmethod
    def _load_vi_words(path: Path):
        vocab, maxlen = set(), 1
        if path.exists():
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    w = line.strip().lower()
                    if not w or w.startswith("#"):
                        continue
                    vocab.add(w)
                    maxlen = max(maxlen, w.count(" ") + 1)
        return vocab, min(maxlen, 6)

    @staticmethod
    def _load_abbrev(paths):
        m: dict[str, list[str]] = {}
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "\t" in line:
                        k, v = line.split("\t", 1)
                    elif "=" in line:
                        k, v = line.split("=", 1)
                    else:
                        continue
                    key = k.strip().lower()
                    vals = [x.strip().lower() for x in re.split(r"[;,]", v) if x.strip()]
                    if key and vals:
                        bucket = m.setdefault(key, [])
                        bucket.extend(x for x in vals if x not in bucket)
        return m

    # ---- queries ----
    @lru_cache(maxsize=200_000)
    def en_known(self, w: str) -> bool:
        if self._en is None:
            return False
        try:
            return bool(self._en.lookup(w))
        except Exception:
            return False

    @lru_cache(maxsize=200_000)
    def vi_known(self, w: str) -> bool:
        if self._vi is not None:
            try:
                if self._vi.lookup(w):
                    return True
            except Exception:
                pass
        return w in self.vi_vocab

    @lru_cache(maxsize=200_000)
    def en_stem(self, w: str) -> str:
        # Prefer a Hunspell lemma (handles irregular forms); else Snowball stem.
        if self._en is not None:
            try:
                for form in self._en.lookuper.good_forms(w):
                    stem = getattr(form, "stem", None)
                    if stem:
                        return stem.lower()
            except Exception:
                pass
        try:
            return self._snow.stemWord(w)
        except Exception:
            return w  # snowballstemmer can IndexError on rare inputs; keep raw word

    def segment_vi(self, syllables):
        """Forward maximum-matching of syllables into known compounds.

        Returns the multi-syllable compounds found (space-joined); single
        syllables are already indexed as plain tokens elsewhere.
        """
        vocab, maxlen = self.vi_vocab, self.vi_max_len
        if not vocab:
            return []
        out, i, n = [], 0, len(syllables)
        while i < n:
            chosen = 1
            for L in range(min(maxlen, n - i), 1, -1):
                if " ".join(syllables[i:i + L]) in vocab:
                    chosen = L
                    break
            if chosen > 1:
                out.append(" ".join(syllables[i:i + chosen]))
            i += chosen
        return out

    def expand_abbr(self, w: str):
        return self.abbr.get(w, [])
