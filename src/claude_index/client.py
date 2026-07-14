"""HTTP client and built-in health probe for the indexing service."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid


def http_json(method: str, url: str, payload: dict | None = None) -> tuple[int, dict]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method=method,
        headers={"Content-Type": "application/json"} if body is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        try:
            data = json.load(exc)
        except Exception:
            data = {"status": "error", "error": str(exc)}
        return exc.code, data


def wait_for_job(base_url: str, job_id: str, poll_seconds: float = 0.5) -> dict:
    while True:
        _, job = http_json("GET", f"{base_url.rstrip('/')}/jobs/{job_id}")
        if job.get("status") in {"complete", "error"}:
            return job
        time.sleep(poll_seconds)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Submit a job to claude-index-server")
    parser.add_argument("--url", default=os.getenv("INDEX_URL", "http://127.0.0.1:9801"))
    parser.add_argument("--ping", action="store_true", help="only check server readiness")
    parser.add_argument("--no-wait", action="store_true", help="return after the job is queued")
    parser.add_argument("--path", action="append", dest="paths",
                        help="mounted directory to index; repeat for multiple roots")
    parser.add_argument("--output", default="corpus.sqlite",
                        help="SQLite filename created under the server output root")
    parser.add_argument("--ocr", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--ocr-langs", default=None,
                        help="Tesseract languages, e.g. vie+eng+rus")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = args.url.rstrip("/")
    try:
        if args.ping:
            status, response = http_json("GET", f"{base}/health")
            ok = status == 200 and response.get("ok") is True
        else:
            payload = {
                "id": str(uuid.uuid4()), "paths": args.paths, "output": args.output,
                "ocr": args.ocr, "ocr_langs": args.ocr_langs, "workers": args.workers,
                "resume": args.resume, "overwrite": args.overwrite,
            }
            status, response = http_json("POST", f"{base}/index", payload)
            ok = status == 202
            if ok and not args.no_wait:
                response = wait_for_job(base, response["id"])
                ok = response.get("status") == "complete"
    except (OSError, ValueError, urllib.error.URLError) as exc:
        response, ok = {"status": "error", "error": str(exc)}, False
    stream = sys.stdout if ok else sys.stderr
    print(json.dumps(response, ensure_ascii=False, indent=2), file=stream)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
