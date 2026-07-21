#!/usr/bin/env bash
# Continual-learning demo (mode A -> B -> A, fast recovery). GPU-node submit:
#   bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_cont \
#        -o bsub_outputs/%J.out env PYTHON=<gpu-venv>/bin/python bash scripts/demo_continual.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bsub_outputs
unset HF_TOKEN || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/inf-scaling/zwhong/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY="${PYTHON:-.venv/bin/python}"
"$PY" scripts/demo_continual.py "$@"
