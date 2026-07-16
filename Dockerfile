FROM rust:1.88-trixie AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates cmake curl g++ pkg-config \
    && rm -rf /var/lib/apt/lists/*

ENV WHISPER_DONT_GENERATE_BINDINGS=1
WORKDIR /src
COPY Cargo.toml Cargo.lock ./
COPY src ./src
COPY data ./data
RUN cargo build --release --locked \
    && cargo test --release --locked \
    && ./target/release/llm-index fetch-data --data-dir data \
    && mkdir -p /models/fastembed \
    && curl --fail --location --retry 3 \
       https://huggingface.co/ggerganov/whisper.cpp/resolve/5359861c739e955e79d9a303bcbc70fb988958b1/ggml-small.bin \
       --output /models/ggml-small.bin \
    && echo "1be3a9b2063867b937e64e2ec7483364a79917e157fa98c5d94b5c1fffea987b  /models/ggml-small.bin" \
       | sha256sum --check \
    && ./target/release/llm-index prefetch-models --embedding-cache /models/fastembed

FROM debian:trixie-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        antiword \
        ffmpeg \
        imagemagick \
        libarchive-tools \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-vie \
        tesseract-ocr-rus \
        tesseract-ocr-deu \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 indexer \
    && useradd --uid 10001 --gid indexer --no-create-home --shell /usr/sbin/nologin indexer \
    && mkdir -p /app/data /app/models /input /output \
    && chown -R indexer:indexer /app /output

COPY --from=builder /src/target/release/llm-index /usr/local/bin/llm-index
COPY --from=builder --chown=indexer:indexer /src/data /app/data
COPY --from=builder --chown=indexer:indexer /models /app/models

WORKDIR /app
ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    WHISPER_MODEL=/app/models/ggml-small.bin \
    FASTEMBED_CACHE_DIR=/app/models/fastembed
USER 10001:10001
EXPOSE 9801
VOLUME ["/output"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["llm-index", "request", "--ping", "--url", "http://127.0.0.1:9801"]

ENTRYPOINT ["llm-index"]
CMD ["serve", "--listen", "0.0.0.0:9801", "--output-root", "/output", "--allowed-root", "/input", "--default-path", "/input", "--ocr-langs", "vie+eng"]
