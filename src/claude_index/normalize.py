"""Bilingual token enrichment for high-recall, engine-agnostic indexing.

For each text we emit a *token bag* combining: raw words, diacritic-folded
variants, English stems/lemmas, Vietnamese compounds (underscore-joined), and
abbreviation expansions. The bag is stored in the FTS ``tokens`` column so that
recall is high regardless of how the searcher spells or inflects a query.
"""
from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)  # unicode letters/digits runs


def fold(s: str) -> str:
    """ASCII-fold: strip combining marks, map đ/Đ -> d/D (diacritic-insensitive)."""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return s.replace("đ", "d").replace("Đ", "D")


class Normalizer:
    def __init__(self, dicts):
        self.d = dicts

    def enrich(self, text: str, lang: str = "und") -> list[str]:
        words = _TOKEN_RE.findall(text.lower())
        if not words:
            return []
        out = list(words)
        out.extend(f for w in words if (f := fold(w)) != w)

        if lang in ("en", "mixed", "und"):
            for w in words:
                if w.isascii() and w.isalpha() and len(w) > 3:
                    st = self.d.en_stem(w)
                    if st and st != w:
                        out.append(st)

        if lang in ("vi", "mixed", "und"):
            for c in self.d.segment_vi(words):
                joined = c.replace(" ", "_")
                out.append(joined)
                fj = fold(joined)
                if fj != joined:
                    out.append(fj)

        for w in set(words):
            for exp in self.d.expand_abbr(w):
                out.extend(_TOKEN_RE.findall(exp))

        seen, uniq = set(), []
        for t in out:
            if t and t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    def query_tokens(self, q: str, lang: str = "und") -> list[str]:
        return self.enrich(q, lang)
