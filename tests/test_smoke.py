"""Smoke test: build a tiny index from sample EN/VI files and query it.

Run with the venv Python:  python tests/test_smoke.py
(Also works under pytest.)
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from claude_index.cli import main
from claude_index.dictionaries import Dictionaries
from claude_index.config import Config
from claude_index.normalize import Normalizer
from claude_index.store import connect, search, top_folders


def _build_sample(d: Path):
    (d / "sub").mkdir(parents=True, exist_ok=True)
    (d / "report_en.txt").write_text(
        "Anti money laundering compliance report. Suspicious activity detected.",
        encoding="utf-8")
    (d / "bao_cao_vi.txt").write_text(
        "Báo cáo giao dịch đáng ngờ tại ngân hàng. Khách hàng rủi ro cao.",
        encoding="utf-8")
    (d / "sub" / "notes.md").write_text(
        "KYC and CDD notes for the bank account review.", encoding="utf-8")


def test_index_and_search():
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "src"
        out = Path(tmp) / "out"
        _build_sample(src)

        rc = main(["index", str(src), "--out", str(out), "--ocr", "off", "--sidecar", "none"])
        assert rc == 0

        db = connect(out)
        nrm = Normalizer(Dictionaries(Config.load()))

        # English stemming: "launder" should find "laundering"
        assert search(db, nrm, "launder", limit=5), "EN stem search failed"
        # Vietnamese diacritic-insensitive: "ngan hang" should find "ngân hàng"
        assert search(db, nrm, "ngan hang", limit=5), "VI folded search failed"
        # Abbreviation expansion: "know your customer" should find the KYC note
        assert search(db, nrm, "know your customer", limit=5), "abbrev search failed"

        folders = top_folders(db, nrm, "bank", n=5)
        assert folders, "top_folders returned nothing"
        db.close()
        print("OK: index, EN stem, VI fold, abbrev expansion, top-folder all passed.")


if __name__ == "__main__":
    test_index_and_search()
