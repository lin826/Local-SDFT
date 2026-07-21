#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p bsub_outputs
unset HF_TOKEN || true
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/proj/inf-scaling/zwhong/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
PY="${PYTHON:-.venv/bin/python}"
"$PY" scripts/demo_correct_once.py "$@"
