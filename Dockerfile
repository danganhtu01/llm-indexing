FROM rust:1.85-bookworm AS builder

WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY data ./data
RUN cargo build --release --locked \
    && ./target/release/llm-index fetch-data --data-dir data --dictionaries-only

FROM debian:bookworm-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 indexer \
    && useradd --uid 10001 --gid indexer --no-create-home --shell /usr/sbin/nologin indexer \
    && mkdir -p /app/data /input /output \
    && chown -R indexer:indexer /app /output

COPY --from=builder /src/target/release/llm-index /usr/local/bin/llm-index
COPY --from=builder --chown=indexer:indexer /src/data /app/data

WORKDIR /app
USER 10001:10001
EXPOSE 9801
VOLUME ["/output"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["llm-index", "request", "--ping", "--url", "http://127.0.0.1:9801"]

ENTRYPOINT ["llm-index"]
CMD ["serve", "--listen", "0.0.0.0:9801", "--output-root", "/output", "--allowed-root", "/input", "--default-path", "/input", "--ocr-langs", "vie+eng"]
