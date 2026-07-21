#!/usr/bin/env bash
# Narrated tool-calling demo (calculator). Runs the model learning to call a
# tool, entirely on-device. On the cluster, submit to a GPU node:
#   bsub -G grp_preemptable -q preemptable -gpu "num=1" -J sdft_toolcall \
#        -o bsub_outputs/%J.out env PYTHON=<gpu-venv>/bin/python bash scripts/demo_toolcall.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bsub_outputs

unset HF_TOKEN || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/inf-scaling/zwhong/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"

PY="${PYTHON:-.venv/bin/python}"
"$PY" scripts/demo_toolcall.py --rounds "${1:-6}"
