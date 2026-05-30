"""claude-indexing: fast bilingual (English/Vietnamese) full-text indexer.

Walks drives, extracts text (incl. OCR for scans), normalizes with EN/VI
dictionaries, and writes engine-agnostic indexes (SQLite FTS5 + JSONL + CSV +
Windows-Explorer sidecars). Also reports the folder with the most matches and a
structural analysis of the tree.
"""

__version__ = "0.1.0"
