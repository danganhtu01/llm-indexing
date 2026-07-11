# claude-index in a container.
#
# Default target (`cli`): plain batch image — run the indexer directly:
#   docker build -t claude-index .
#   docker run --rm -v /docs:/mirror:ro -v idx:/index claude-index \
#       index /mirror --out /index --ocr auto --resume --config /app/config.container.yaml
#
# `supervised` target: same image plus an external supervisor binary injected
# via --build-arg SUPERVISOR_IMAGE=<image containing /usr/local/bin/stage-shim>,
# for orchestrators that drive batch stages over a small HTTP control plane.
# The supervisor's command/env are provided at runtime by the orchestrator.
#
# Dictionaries + tessdata are vendored at BUILD time (fetch_data.py needs
# raw.githubusercontent.com) so the runtime container needs no egress.

FROM python:3.10-slim AS cli
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-vie tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --uid 10001 --user-group --no-create-home indexer

WORKDIR /app
COPY pyproject.toml requirements.txt ./
COPY src ./src
COPY scripts ./scripts
COPY config.container.yaml ./
RUN pip install --no-cache-dir -e . \
    && python scripts/fetch_data.py \
    && chown -R indexer:indexer /app/data

ENV OMP_THREAD_LIMIT=1
USER indexer
ENTRYPOINT ["claude-index"]

# ---------------------------------------------------------------- supervised
ARG SUPERVISOR_IMAGE=scratch
FROM ${SUPERVISOR_IMAGE} AS supervisor

FROM cli AS supervised
COPY --from=supervisor /usr/local/bin/stage-shim /usr/local/bin/stage-shim
EXPOSE 9000
ENTRYPOINT ["/usr/local/bin/stage-shim"]
