# HTTP API

The container listens on TCP port 9801. All requests and responses use JSON.

## `GET /health`

Returns `200` with service version and whether a job is queued or running.

## `POST /index`

Queues one indexing job and returns `202`. Fields are `paths` (mounted directory
array), `output` (plain `.sqlite` filename), `ocr` (`auto`, `on`, or `off`),
`ocr_langs`, `workers`, `resume`, and `overwrite`. Omitted paths use the service
default. Reusing a job `id` is rejected.

The queue returns `429` when full. Invalid JSON, paths outside allowed roots,
non-directory inputs, and unsafe output paths fail without publishing a file.

## `GET /jobs/{id}`

Returns `queued`, `running`, `complete`, or `error`. A complete response includes
the database path, file/OCR/error counts, elapsed time, and OCR languages.
