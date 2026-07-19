#!/usr/bin/env bash
# Submit the Airplane-Mode Coach demo (headless) to a GPU node, bv_scripts style.
#
#   bv-normal.sh convention: -G grp_runtime -q normal, blaunch bash <script> [args]
# From this repo root:
#   bsub -G grp_runtime -q preemptable -gpu "num=1" -J sdft_demo \
#        -o bsub_outputs/%J.out bash scripts/bv-demo.sh [CONFIG] [ROUNDS]
#
# Compute nodes on this cluster have no internet: pre-cache the model on the
# login node first (the HF cache is shared):
#   python -c "from huggingface_hub import snapshot_download as d; d('LiquidAI/LFM2.5-230M')"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bsub_outputs

CONFIG="${1:-configs/demo_house_style.yaml}"
ROUNDS="${2:-6}"

unset HF_TOKEN || true            # the login node's token is expired; anon pulls work
export PYTHONNOUSERSITE=1         # avoid a stale ~/.local torchvision shadowing base torch
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

PY="${PYTHON:-.venv/bin/python}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -m sdft.online.cli demo --config "$CONFIG" --rounds "$ROUNDS"
