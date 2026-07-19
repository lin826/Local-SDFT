#!/usr/bin/env bash
# Smoke or full OpenClaw-RL ReTool eval against LFM2.5-230M.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/openclaw_rl_eval.yaml}"
NUM_EXAMPLES="${NUM_EXAMPLES:-2}"
N_SAMPLES="${N_SAMPLES:-1}"
FORMAT="${FORMAT:-auto}"
MODEL="${MODEL:-}"

extra_args=()
if [[ -n "$MODEL" ]]; then
  # Override model path via env without editing YAML (requires yq or inline python).
  extra_args+=(--config "$CONFIG")
fi

echo "Running OpenClaw-RL eval: config=$CONFIG num_examples=$NUM_EXAMPLES format=$FORMAT"

uv run python -m sdft.toolcall.openclaw_eval \
  --config "$CONFIG" \
  --num-examples "$NUM_EXAMPLES" \
  --n-samples "$N_SAMPLES" \
  --format "$FORMAT" \
  "${extra_args[@]}"

echo "Results: outputs/benchmarks/openclaw-rl/latest.json"
