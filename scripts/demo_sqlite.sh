#!/usr/bin/env bash
# Text→SQL against a real SQLite engine, headless on a GPU node.
#
#   bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_sqlite \
#        -o bsub_outputs/%J.out env PYTHON=.venv-gpu/bin/python bash scripts/demo_sqlite.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bsub_outputs
unset HF_TOKEN || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/inf-scaling/zwhong/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY="${PYTHON:-.venv/bin/python}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" scripts/demo_sqlite.py "$@"
