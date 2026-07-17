"""Real HTTP-socket integration test for the container service contract."""
from __future__ import annotations

import sqlite3
import tempfile
import threading
from pathlib import Path

from claude_index.client import http_json, wait_for_job
from claude_index.server import IndexHttpServer, IndexService, JobQueue


def test_http_socket_builds_sqlite_and_rejects_unmounted_paths():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        source = root / "input"
        output = root / "output"
        source.mkdir()
        output.mkdir()
        (source / "hello.txt").write_text(
            "Vietnamese ngân hàng and English compliance corpus.", encoding="utf-8")

        service = IndexService(str(output), [str(source)], [str(source)],
                               default_ocr_langs="vie+eng", default_workers=1)
        jobs = JobQueue(service)
        server = IndexHttpServer(("127.0.0.1", 0), jobs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            status, health = http_json("GET", f"{base}/health")
            assert status == 200 and health["ok"] is True

            status, queued = http_json("POST", f"{base}/index", {
                "id": "index-1", "paths": [str(source)], "output": "corpus.sqlite",
                "ocr": "off", "workers": 1,
            })
            assert status == 202
            result = wait_for_job(base, queued["id"], poll_seconds=0.01)
            assert result["status"] == "complete", result
            assert result["files"] == 1
            database = output / "corpus.sqlite"
            assert list(output.iterdir()) == [database]
            with sqlite3.connect(database) as db:
                assert db.execute("SELECT COUNT(*) FROM files").fetchone()[0] == 1
                assert db.execute(
                    "SELECT COUNT(*) FROM fts WHERE fts MATCH 'ngan'"
                ).fetchone()[0] == 1

            status, queued = http_json("POST", f"{base}/index", {
                "id": "index-2", "paths": [str(root)], "output": "denied.sqlite",
                "ocr": "off",
            })
            assert status == 202
            denied = wait_for_job(base, queued["id"], poll_seconds=0.01)
            assert denied["status"] == "error"
            assert "INDEX_ALLOWED_ROOTS" in denied["error"]
        finally:
            server.shutdown()
            server.server_close()
            jobs.close()
            thread.join(timeout=2)
