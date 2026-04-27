FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git curl ca-certificates tini wget gnupg procps && \
    rm -rf /var/lib/apt/lists/*

COPY docker-assets/ptxas-arm64 /usr/local/cuda-13.0/bin/ptxas
RUN mkdir -p /usr/local/cuda-13.0/bin && \
    chmod 0755 /usr/local/cuda-13.0/bin/ptxas && \
    ln -sf /usr/local/cuda-13.0 /usr/local/cuda

RUN python3 -m pip install --break-system-packages uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY nanochat ./nanochat
COPY scripts ./scripts
COPY runs ./runs
COPY tasks ./tasks
COPY dev ./dev

RUN uv venv .venv && \
    . .venv/bin/activate && \
    uv sync --extra gpu --extra ray && \
    chown root:root /workspace/.venv/bin/py-spy && \
    chmod 4755 /workspace/.venv/bin/py-spy

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
