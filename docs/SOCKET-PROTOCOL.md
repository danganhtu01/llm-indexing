# Indexing HTTP socket

Like `llm-redaction`, the image runs a resident HTTP service. It listens on port
9801 by default and exposes a bounded JSON control plane. Source bytes stay in
read-only mounted directories; they are not uploaded through HTTP.

## Endpoints

```text
GET  /health       readiness and busy state
POST /index        queue one indexing job (maximum body: 1 MiB)
GET  /jobs/{id}    poll job state and retrieve the output database path
```

Submit a job:

```json
{"id":"job-42","paths":["/input/contracts","/input/policies"],"output":"legal-corpus.sqlite","ocr":"auto","ocr_langs":"vie+eng","workers":4,"resume":false,"overwrite":false}
```

`POST /index` returns HTTP 202:

```json
{"id":"job-42","status":"queued","submitted_at":1784000000.0}
```

The job progresses through `queued`, `running`, then `complete` or `error`.
A completed `GET /jobs/job-42` response is:

```json
{"id":"job-42","status":"complete","database":"/output/legal-corpus.sqlite","files":2381,"ocr_files":47,"errors":2,"elapsed_seconds":91.203,"ocr_langs":"vie+eng"}
```

`paths` must resolve beneath `INDEX_ALLOWED_ROOTS`. `output` is a filename, not a
path, and must end in `.sqlite`. A single background worker serializes indexing
jobs because PyMuPDF is not thread-safe; extraction within a job remains threaded.
The queue is bounded (32 pending jobs by default); excess submissions receive
HTTP 429 instead of growing memory without limit.
The database is built on the output filesystem and atomically renamed, so readers
see either the previous complete corpus or the new complete corpus. Existing
output is refused unless `resume` or `overwrite` is true.
