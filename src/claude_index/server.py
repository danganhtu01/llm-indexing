"""Resident HTTP service that queues jobs and publishes SQLite corpus indexes."""
from __future__ import annotations

import argparse
import json
import os
import queue
import shutil
import signal
import sqlite3
import tempfile
import threading
import time
import uuid
from argparse import Namespace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from . import __version__
from .cli import cmd_index

DEFAULT_MAX_BODY = 1024 * 1024
DEFAULT_MAX_PENDING = 32
MAX_JOB_HISTORY = 1000


class RequestError(ValueError):
    pass


class BusyError(RequestError):
    pass


def _split_paths(value: str) -> list[str]:
    return [item for item in value.split(os.pathsep) if item]


def _within(path: Path, roots: list[Path]) -> bool:
    return any(path == root or root in path.parents for root in roots)


class IndexService:
    def __init__(self, output_root: str, allowed_roots: list[str], default_paths: list[str],
                 config: str | None = None, default_ocr_langs: str = "vie+eng",
                 default_workers: int = 4):
        self.output_root = Path(output_root).resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.allowed_roots = [Path(p).resolve(strict=True) for p in allowed_roots]
        self.default_paths = default_paths
        self.config = config
        self.default_ocr_langs = default_ocr_langs
        self.default_workers = default_workers

    def index(self, request: dict) -> dict:
        request_id = str(request.get("id") or uuid.uuid4())
        paths = request.get("paths") or self.default_paths
        if not isinstance(paths, list) or not paths or not all(isinstance(p, str) for p in paths):
            raise RequestError("paths must be a non-empty array of mounted directories")
        try:
            resolved = [Path(p).resolve(strict=True) for p in paths]
        except OSError as exc:
            raise RequestError(f"input path does not exist: {exc.filename}") from exc
        if any(not p.is_dir() for p in resolved):
            raise RequestError("every input path must be a directory")
        if any(not _within(p, self.allowed_roots) for p in resolved):
            raise RequestError("input path is outside INDEX_ALLOWED_ROOTS")

        output_name = request.get("output", "corpus.sqlite")
        if (not isinstance(output_name, str) or Path(output_name).name != output_name
                or not output_name.endswith(".sqlite")):
            raise RequestError("output must be a plain filename ending in .sqlite")
        destination = self.output_root / output_name
        resume = bool(request.get("resume", False))
        overwrite = bool(request.get("overwrite", False))
        if destination.exists() and not (resume or overwrite):
            raise RequestError("output already exists; set resume or overwrite")

        ocr = request.get("ocr", "auto")
        if ocr not in {"auto", "on", "off"}:
            raise RequestError("ocr must be auto, on, or off")
        ocr_langs = request.get("ocr_langs") or self.default_ocr_langs
        if not isinstance(ocr_langs, str) or not ocr_langs.replace("+", "").isalnum():
            raise RequestError("ocr_langs must look like vie+eng or vie+eng+rus")
        workers = request.get("workers") or self.default_workers
        if not isinstance(workers, int) or not 1 <= workers <= 64:
            raise RequestError("workers must be an integer from 1 to 64")

        work = Path(tempfile.mkdtemp(prefix=".indexing-", dir=self.output_root))
        started = time.time()
        try:
            if resume and destination.exists():
                shutil.copy2(destination, work / "index.sqlite")
            args = Namespace(
                paths=[str(p) for p in resolved], out=str(work), ocr=ocr,
                ocr_langs=ocr_langs, sidecar="none", workers=workers,
                max_bytes=None, config=self.config, resume=resume, sqlite_only=True,
            )
            rc = cmd_index(args)
            if rc != 0:
                raise RuntimeError(f"index command failed with status {rc}")
            database = work / "index.sqlite"
            with sqlite3.connect(database) as db:
                files, ocr_files, errors = db.execute(
                    "SELECT COUNT(*), COALESCE(SUM(ocr_used), 0), "
                    "COALESCE(SUM(CASE WHEN method LIKE 'error:%' THEN 1 ELSE 0 END), 0) "
                    "FROM files"
                ).fetchone()
            os.replace(database, destination)
            return {
                "id": request_id, "database": str(destination), "files": files,
                "ocr_files": ocr_files, "errors": errors,
                "elapsed_seconds": round(time.time() - started, 3),
                "ocr_langs": ocr_langs,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)


class JobQueue:
    """One worker serializes PyMuPDF/OCR/SQLite jobs; HTTP remains responsive."""

    def __init__(self, service: IndexService, max_pending: int = DEFAULT_MAX_PENDING):
        self.service = service
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.pending: queue.Queue[tuple[str, dict] | None] = queue.Queue(maxsize=max_pending)
        self.worker = threading.Thread(target=self._run, name="index-worker", daemon=True)
        self.worker.start()

    def submit(self, request: dict) -> dict:
        job_id = str(request.get("id") or uuid.uuid4())
        with self.lock:
            if job_id in self.jobs:
                raise RequestError("job id already exists")
            if len(self.jobs) >= MAX_JOB_HISTORY:
                finished = sorted(
                    (j for j in self.jobs.values() if j["status"] in {"complete", "error"}),
                    key=lambda j: j.get("completed_at", 0),
                )
                for old in finished[:max(1, len(self.jobs) - MAX_JOB_HISTORY + 1)]:
                    self.jobs.pop(old["id"], None)
            if len(self.jobs) >= MAX_JOB_HISTORY:
                raise BusyError("job history is full while all retained jobs are active")
            job = {"id": job_id, "status": "queued", "submitted_at": time.time()}
            self.jobs[job_id] = job
        request = dict(request)
        request["id"] = job_id
        try:
            self.pending.put_nowait((job_id, request))
        except queue.Full as exc:
            with self.lock:
                self.jobs.pop(job_id, None)
            raise BusyError("indexing queue is full") from exc
        return dict(job)

    def get(self, job_id: str) -> dict | None:
        with self.lock:
            job = self.jobs.get(job_id)
            return dict(job) if job else None

    def busy(self) -> bool:
        with self.lock:
            return any(j["status"] in {"queued", "running"} for j in self.jobs.values())

    def close(self):
        self.pending.put(None)
        self.worker.join(timeout=5)

    def _run(self):
        while True:
            item = self.pending.get()
            if item is None:
                return
            job_id, request = item
            with self.lock:
                self.jobs[job_id].update(status="running", started_at=time.time())
            try:
                result = self.service.index(request)
                result.update(status="complete", completed_at=time.time())
            except Exception as exc:  # a bad file/job must never kill the service
                result = {"id": job_id, "status": "error", "error": str(exc),
                          "completed_at": time.time()}
            with self.lock:
                self.jobs[job_id] = result


class _Handler(BaseHTTPRequestHandler):
    server_version = "claude-index-server"

    def _json(self, status: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/health":
            return self._json(200, {"ok": True, "service": "llm-indexing",
                                    "version": __version__, "busy": self.server.jobs.busy()})
        if path.startswith("/jobs/"):
            job = self.server.jobs.get(path.removeprefix("/jobs/"))
            return self._json(200, job) if job else self._json(404, {"error": "job not found"})
        self._json(404, {"error": "not found"})

    def do_POST(self):
        if urlsplit(self.path).path != "/index":
            return self._json(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._json(400, {"error": "invalid Content-Length"})
        if length <= 0:
            return self._json(400, {"error": "empty request"})
        if length > self.server.max_body:
            return self._json(413, {"error": "request body too large"})
        try:
            request = json.loads(self.rfile.read(length))
            if not isinstance(request, dict):
                raise RequestError("request must be a JSON object")
            job = self.server.jobs.submit(request)
            self._json(202, job)
        except BusyError as exc:
            self._json(429, {"status": "error", "error": str(exc)})
        except (json.JSONDecodeError, UnicodeDecodeError, RequestError) as exc:
            self._json(400, {"status": "error", "error": str(exc)})

    def log_message(self, fmt, *args):
        print(f"http {self.address_string()} {fmt % args}", flush=True)


class IndexHttpServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, jobs: JobQueue, max_body: int = DEFAULT_MAX_BODY):
        self.jobs = jobs
        self.max_body = max_body
        super().__init__(address, _Handler)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve indexing jobs over HTTP")
    parser.add_argument("--listen", default=os.getenv("INDEX_LISTEN", "0.0.0.0:9801"))
    parser.add_argument("--output-root", default=os.getenv("INDEX_OUTPUT_ROOT", "/output"))
    parser.add_argument("--allowed-root", action="append", dest="allowed_roots")
    parser.add_argument("--default-path", action="append", dest="default_paths")
    parser.add_argument("--config", default=os.getenv("INDEX_CONFIG") or None)
    parser.add_argument("--ocr-langs", default=os.getenv("OCR_LANGS", "vie+eng"))
    parser.add_argument("--workers", type=int, default=int(os.getenv("INDEX_WORKERS", "4")))
    parser.add_argument("--max-body", type=int, default=DEFAULT_MAX_BODY)
    parser.add_argument("--max-pending", type=int,
                        default=int(os.getenv("INDEX_MAX_PENDING", str(DEFAULT_MAX_PENDING))))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    host, port = args.listen.rsplit(":", 1)
    allowed = args.allowed_roots or _split_paths(os.getenv("INDEX_ALLOWED_ROOTS", "/input"))
    defaults = args.default_paths or _split_paths(os.getenv("INDEX_DEFAULT_PATHS", "/input"))
    service = IndexService(args.output_root, allowed, defaults, args.config,
                           args.ocr_langs, args.workers)
    jobs = JobQueue(service, args.max_pending)
    server = IndexHttpServer((host, int(port)), jobs, args.max_body)

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    print(f"claude-index-server listening on http://{server.server_address[0]}:"
          f"{server.server_address[1]}", flush=True)
    try:
        server.serve_forever(poll_interval=0.5)
    finally:
        server.server_close()
        jobs.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
