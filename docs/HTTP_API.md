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

## Corpus read surface

Consumer apps used to open `corpus.sqlite` directly to render a directory
listing or a document preview. These routes serve that instead, so no consumer
needs to know the SQLite schema. Every route accepts an optional
`output=NAME.sqlite` query param (default `corpus.sqlite`) naming which
published database to read, validated the same way as `POST /index`'s
`output` field. The database is absent until the first job completes; every
route below then degrades to an empty/zeroed result rather than an error.

### `GET /corpus/tree?root=NAME`

A sorted recursive walk of one allowed input root, joined by absolute path
against the published corpus database. `root` names one of the service's
configured allowed roots — its directory name, e.g. `/input` -> `input`
(`INDEX_ALLOWED_ROOTS`/`--allowed-root`). An unrecognized `root` is `400`.

Returns a JSON array of entries, directories before files, alphabetical within
each:

```json
[
  {
    "path": "Customers/statement.pdf",
    "name": "statement.pdf",
    "kind": "file",
    "depth": 1,
    "size_bytes": 40213,
    "modified_at": 1752600000,
    "document_id": 42,
    "character_count": 8172,
    "method": "pdf",
    "lang": "en",
    "snippet": "first 400 characters of the extracted text…"
  }
]
```

`path` is root-relative POSIX (`/`-separated). `kind` is `"dir"` or `"file"`.
`document_id`, `character_count`, `method`, `lang` and `snippet` are present
only on files that matched a row in the corpus database by exact absolute
path; unmatched files and every directory omit them. Symlinks are skipped,
matching the indexer's own default.

### `GET /corpus/documents/{id}/text`

Streams the extracted text for one document (`files.id`) as
`text/plain; charset=utf-8`. `404` when the database is absent or holds no
matching id.

### `GET /corpus/status`

Cheap corpus-wide aggregates:

```json
{
  "indexed_files": 1204,
  "total_characters": 9823110,
  "total_bytes": 512300000,
  "ocr_files": 88,
  "languages": [["en", 900], ["vi", 304]],
  "methods": [["text", 1000], ["pdf-ocr", 204]]
}
```
