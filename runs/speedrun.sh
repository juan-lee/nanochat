#!/bin/bash

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# The default settings target a blank 8xH100 GPU node, but the run is also configurable
# for smaller or different hardware such as a single GB10 GPU via environment overrides.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run can take hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun bash runs/speedrun.sh

set -euo pipefail

# Default intermediate artifacts directory is in ~/.cache/nanochat
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
mkdir -p "$NANOCHAT_BASE_DIR"

# Hardware/runtime knobs. Override these from env for different boxes.
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
BASE_DEPTH="${BASE_DEPTH:-24}"
BASE_TARGET_PARAM_DATA_RATIO="${BASE_TARGET_PARAM_DATA_RATIO:-8}"
BASE_DEVICE_BATCH_SIZE="${BASE_DEVICE_BATCH_SIZE:-16}"
BASE_EVAL_DEVICE_BATCH_SIZE="${BASE_EVAL_DEVICE_BATCH_SIZE:-$BASE_DEVICE_BATCH_SIZE}"
SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-$BASE_DEVICE_BATCH_SIZE}"
BASE_EXTRA_ARGS="${BASE_EXTRA_ARGS:-}"
BASE_EVAL_EXTRA_ARGS="${BASE_EVAL_EXTRA_ARGS:-}"
SFT_EXTRA_ARGS="${SFT_EXTRA_ARGS:-}"
CHAT_EVAL_EXTRA_ARGS="${CHAT_EVAL_EXTRA_ARGS:-}"
ENABLE_FP8="${ENABLE_FP8:-auto}"

if [[ "$ENABLE_FP8" == "auto" ]]; then
    if [[ "$NPROC_PER_NODE" -ge 8 ]]; then
        FP8_ARG="--fp8"
    else
        FP8_ARG=""
    fi
elif [[ "$ENABLE_FP8" == "1" || "$ENABLE_FP8" == "true" || "$ENABLE_FP8" == "yes" ]]; then
    FP8_ARG="--fp8"
else
    FP8_ARG=""
fi

# -----------------------------------------------------------------------------
# Python venv setup with uv

command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup
if [ -z "${WANDB_RUN:-}" ]; then
    WANDB_RUN=dummy
fi

# -----------------------------------------------------------------------------
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

python -m nanochat.dataset -n "${TOKENIZER_BOOTSTRAP_SHARDS:-8}"
python -m nanochat.dataset -n "${PRETRAIN_TOTAL_SHARDS:-170}" &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train ${TOK_TRAIN_ARGS:-}
python -m scripts.tok_eval ${TOK_EVAL_ARGS:-}

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_train -- \
    --depth="$BASE_DEPTH" \
    --target-param-data-ratio="$BASE_TARGET_PARAM_DATA_RATIO" \
    --device-batch-size="$BASE_DEVICE_BATCH_SIZE" \
    $FP8_ARG \
    --run="$WANDB_RUN" \
    $BASE_EXTRA_ARGS

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.base_eval -- \
    --device-batch-size="$BASE_EVAL_DEVICE_BATCH_SIZE" \
    $BASE_EVAL_EXTRA_ARGS

# -----------------------------------------------------------------------------
# SFT
curl -L -o "$NANOCHAT_BASE_DIR/identity_conversations.jsonl" https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_sft -- \
    --device-batch-size="$SFT_DEVICE_BATCH_SIZE" \
    --run="$WANDB_RUN" \
    $SFT_EXTRA_ARGS

torchrun --standalone --nproc_per_node="$NPROC_PER_NODE" -m scripts.chat_eval -- \
    -i sft \
    $CHAT_EVAL_EXTRA_ARGS

python -m nanochat.report generate
