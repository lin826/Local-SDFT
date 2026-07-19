#!/usr/bin/env bash
# Build data, train SDFT on OpenClaw-style tool trajectories, merge, and three-way eval.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONFIG="${CONFIG:-configs/openclaw_tooluse_sdft.yaml}"
DEMO_EVAL="${DEMO_EVAL:-configs/openclaw_demo_eval.yaml}"
AIME_EVAL="${AIME_EVAL:-configs/openclaw_rl_eval.yaml}"
AIME_N="${AIME_N:-3}"
FORMAT="${FORMAT:-openclaw}"
MERGED="${MERGED:-outputs/openclaw-tooluse-merged}"

echo "==> sync deps"
uv sync --extra toolcall

echo "==> build tool-use data"
uv run python scripts/build_openclaw_tooluse_data.py --write-sdft

echo "==> train LoRA (identity SDFT)"
uv run python -m sdft.train --config "$CONFIG" --data data/openclaw_tooluse_sdft.jsonl

echo "==> merge adapter"
uv run python -m sdft.merge --config "$CONFIG" --out "$MERGED"

echo "==> zero-shot base (demo)"
uv run python -m sdft.toolcall.openclaw_eval \
  --config "$DEMO_EVAL" --format "$FORMAT" \
  --out-dir outputs/benchmarks/openclaw-rl/demo-zero-shot

echo "==> one-shot base (demo)"
uv run python -m sdft.toolcall.openclaw_eval \
  --config "$DEMO_EVAL" --format "$FORMAT" --one-shot \
  --out-dir outputs/benchmarks/openclaw-rl/demo-one-shot

echo "==> post-SDFT (demo)"
uv run python - <<PY
from pathlib import Path
import yaml
raw = yaml.safe_load(Path("$DEMO_EVAL").read_text()) or {}
raw.setdefault("model", {})["name"] = "$MERGED"
out = Path("outputs/benchmarks/openclaw-rl/_demo_sdft_eval.yaml")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
print(f"wrote {out}")
PY
uv run python -m sdft.toolcall.openclaw_eval \
  --config outputs/benchmarks/openclaw-rl/_demo_sdft_eval.yaml --format "$FORMAT" \
  --out-dir outputs/benchmarks/openclaw-rl/demo-sdft

echo "==> optional AIME slice (n=$AIME_N)"
uv run python -m sdft.toolcall.openclaw_eval \
  --config "$AIME_EVAL" --num-examples "$AIME_N" --format "$FORMAT" \
  --out-dir outputs/benchmarks/openclaw-rl/aime-zero-shot
uv run python -m sdft.toolcall.openclaw_eval \
  --config "$AIME_EVAL" --num-examples "$AIME_N" --format "$FORMAT" --one-shot \
  --out-dir outputs/benchmarks/openclaw-rl/aime-one-shot
uv run python - <<PY
from pathlib import Path
import yaml
raw = yaml.safe_load(Path("$AIME_EVAL").read_text()) or {}
raw.setdefault("model", {})["name"] = "$MERGED"
out = Path("outputs/benchmarks/openclaw-rl/_aime_sdft_eval.yaml")
out.write_text(yaml.dump(raw, default_flow_style=False, sort_keys=False))
print(f"wrote {out}")
PY
uv run python -m sdft.toolcall.openclaw_eval \
  --config outputs/benchmarks/openclaw-rl/_aime_sdft_eval.yaml --num-examples "$AIME_N" --format "$FORMAT" \
  --out-dir outputs/benchmarks/openclaw-rl/aime-sdft

echo "Done. Check outputs/benchmarks/openclaw-rl/"
