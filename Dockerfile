# syntax=docker/dockerfile:1
# Verified on 2026-11-17 against hub.docker.com API:
# https://hub.docker.com/v2/repositories/nvidia/cuda/tags/12.8.1-cudnn-runtime-ubuntu22.04 -> 200
# No RunPod handler needed: this endpoint is a LOAD BALANCER (HTTP) type, RunPod proxies
# straight to the container port. server.py implements the health contract itself.
ARG CUDA_TAG=12.8.1-cudnn-runtime-ubuntu22.04
FROM nvidia/cuda:${CUDA_TAG}

# /releases/latest currently resolves to v1.0.0, which has only nightly-tag.txt.
# That release explicitly links b12359. Its real Linux x86_64 CUDA 12.8 assets:
# https://github.com/ggml-org/llama.cpp/releases/download/b12359/llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz
# https://github.com/ggml-org/llama.cpp/releases/download/b12359/cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz
# Pin the linked build and the API-reported SHA256 values for reproducibility.
ARG LLAMA_RELEASE=b12359
ARG LLAMA_ARCHIVE=llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz
ARG LLAMA_SHA256=50c937bec5bbd9f74c99138034f7403283ee21f1943f2429d7b1e4c079eafc11
ARG CUDART_ARCHIVE=cudart-llama-b12359-bin-ubuntu-cuda-12.8-x64.tar.gz
ARG CUDART_SHA256=75a12d227f44a391608992329241765343bd143ee9445b90dc62e154ee726e5d

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/ \
    LLAMA_BIN_DIR=/opt/llama.cpp \
    MODEL_FILE=/data/models/Qwen3.8-Flash-Next-Uncensored-IQ4_XS-MTP-00001-of-00003.gguf \
    MMPROJ_FILE=/data/models/mmproj-Qwen3.8-Flash-Next-Uncensored-F16.gguf \
    MODEL_ALIAS=team-hermes \
    REQUEST_TIMEOUT=1200

# One RUN layer. Keep both archives' internal directory structure intact.
# Archive contents were NOT downloaded/inspected locally: LLAMA_BIN_DIR is a search
# root, not an assumed bin/ directory. server.py recursively finds llama-server
# and its shared libraries; LLAMA_SERVER_BIN can explicitly select the binary.
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl libgomp1; \
    python3 -m pip install --no-cache-dir 'fastapi>=0.115,<1' 'uvicorn>=0.30,<1' 'httpx>=0.27,<1'; \
    mkdir -p /opt/llama.cpp; \
    curl --fail --location --retry 5 --retry-all-errors \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_RELEASE}/${LLAMA_ARCHIVE}" -o /tmp/llama.tar.gz; \
    curl --fail --location --retry 5 --retry-all-errors \
        "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_RELEASE}/${CUDART_ARCHIVE}" -o /tmp/cudart.tar.gz; \
    printf '%s  %s\n' "$LLAMA_SHA256" /tmp/llama.tar.gz "$CUDART_SHA256" /tmp/cudart.tar.gz | sha256sum --check -; \
    tar -xzf /tmp/llama.tar.gz -C /opt/llama.cpp; \
    tar -xzf /tmp/cudart.tar.gz -C /opt/llama.cpp; \
    find /opt/llama.cpp -type f -name llama-server -exec chmod +x {} \;; \
    rm -f /tmp/llama.tar.gz /tmp/cudart.tar.gz; \
    rm -rf /var/lib/apt/lists/*

# No model weights are copied into the image. The template handler is inherited.
# Its launch/import contract remains unverified because its repository is unavailable.
WORKDIR /
COPY server.py /server.py
EXPOSE 8000
CMD ["python3", "/handler/main.py"]
