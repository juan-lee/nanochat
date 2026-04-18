FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    OMP_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates tini && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
COPY nanochat ./nanochat
COPY scripts ./scripts
COPY runs ./runs
COPY tasks ./tasks
COPY dev ./dev

RUN pip install --upgrade pip && pip install uv && uv sync --extra gpu

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
