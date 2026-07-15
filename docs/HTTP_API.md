# HTTP API
The container listens on TCP 9801. All API requests and responses are JSON.

## `GET /health`

Returns service version/readiness and whether a job is queued or running.

## `POST /index`

Queues one job and returns `202 Accepted` with an `id`.

```json
{
  "paths": ["/input"],
  "output": "corpus.sqlite",
  "ocr": "exhaustive",
  "ocr_langs": "vie+eng",
  "workers": 4,
  "include_paths": ["Customers/new.pdf", "Meetings/changed.mp4"],
  "resume": true,
  "overwrite": true
}
```

`ocr` accepts `auto`, `on`, `off` or `exhaustive`. Omitted paths use the service
default. The queue returns 429 when full. Invalid JSON, out-of-root paths,
non-directory inputs and unsafe output names fail without publishing a file.
When present, `include_paths` must contain existing relative files confined
under an input root. Only those files are extracted; source deletion pruning
still uses the complete mounted tree.

## `GET /jobs/{id}`

Returns `queued`, `running`, `cancelling`, `cancelled`, `complete` or `error`.
Running jobs include live `processed` and `total` file counters. A completed job includes the
database path, file/OCR/error/incomplete counts, embedded chunk count, removed
source count, elapsed time and OCR languages.

## `POST /jobs/{id}/cancel`

Requests cooperative cancellation of a queued or running job. The engine stops
before the next extraction/embedding boundary, discards the temporary build and
keeps the previously published SQLite corpus unchanged. Poll the job until its
state becomes `cancelled`.

## `POST /search/vector`

Searches the current SQLite chunk embeddings. The request accepts `query`,
`index` and `limit`; the response contains path, file name, chunk index, cosine
score and chunk content. Model artifacts are local and no text leaves the box.
