FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    build-essential git curl ca-certificates tini wget gnupg && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    printf 'deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/ /\n' > /etc/apt/sources.list.d/cuda.list && \
    apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-13-0 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/local/cuda-13.0 /usr/local/cuda

RUN python3 -m pip install --break-system-packages uv

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY nanochat ./nanochat
COPY scripts ./scripts
COPY runs ./runs
COPY tasks ./tasks
COPY dev ./dev

RUN uv venv .venv && . .venv/bin/activate && uv sync --extra gpu

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
