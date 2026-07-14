FROM python:3.12-slim-bookworm

ARG TESSERACT_LANG_PACKAGES="tesseract-ocr-eng tesseract-ocr-vie"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TESSDATA_PREFIX=/usr/share/tesseract-ocr/5/tessdata \
    INDEX_ALLOWED_ROOTS=/input \
    INDEX_DEFAULT_PATHS=/input \
    INDEX_OUTPUT_ROOT=/output \
    OCR_LANGS=vie+eng

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       tesseract-ocr ${TESSERACT_LANG_PACKAGES} \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN python -m pip install . \
    && python scripts/fetch_data.py --dictionaries-only \
    && useradd --uid 10001 --user-group --no-create-home --shell /usr/sbin/nologin indexer \
    && mkdir -p /input /output \
    && chown -R indexer:indexer /output

USER 10001:10001

EXPOSE 9801
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["claude-index-request", "--ping", "--url", "http://127.0.0.1:9801"]

ENTRYPOINT ["claude-index-server"]
CMD ["--listen", "0.0.0.0:9801"]
