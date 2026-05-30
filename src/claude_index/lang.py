"""Lightweight, low-cost English/Vietnamese language detection.

Uses Vietnamese diacritic density plus optional dictionary coverage. Samples
only the first ~2 KB of text, so it is cheap to run on every file.
"""
from __future__ import annotations

import re

# Precomposed Vietnamese vowels/consonants with tone/quality marks (+ đ).
VN_DIACRITIC_RE = re.compile(
    r"[ăâđêôơưáàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộ"
    r"ớờởỡợúùủũụứừửữựýỳỷỹỵ]",
    re.IGNORECASE,
)
WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def detect_lang(text: str, dicts=None, sample: int = 2000) -> str:
    """Return one of: 'vi', 'en', 'mixed', 'und'."""
    s = text[:sample]
    if not s.strip():
        return "und"
    words = WORD_RE.findall(s.lower())
    if not words:
        return "und"

    vn_dia = len(VN_DIACRITIC_RE.findall(s))
    vn_ratio = vn_dia / max(1, len(s))

    en_cov = vi_cov = 0.0
    if dicts is not None:
        sampled = words[:300]
        en_hits = sum(1 for w in sampled if dicts.en_known(w))
        vi_hits = sum(1 for w in sampled if dicts.vi_known(w))
        en_cov = en_hits / len(sampled)
        vi_cov = vi_hits / len(sampled)

    is_vi = vn_ratio > 0.02 or vi_cov > 0.35
    is_en = en_cov > 0.35
    if is_vi and is_en:
        return "mixed"
    if is_vi:
        return "vi"
    if is_en:
        return "en"
    if vn_dia:
        return "vi"
    return "en" if en_cov >= vi_cov and en_cov > 0 else "und"
