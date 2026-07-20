#!/usr/bin/env python3
"""Run AlpacaEval-style judging on a model_outputs JSON file.

Local judge (default, Colab T4 — no OpenAI)::

  uv sync --extra alpacaeval
  # optional: pip install bitsandbytes  # CUDA 4-bit
  export JUDGE=local
  # optional: export ALPACA_EVAL_LOCAL_JUDGE=Qwen/Qwen2.5-7B-Instruct  # if 9B OOMs
  uv run python scripts/run_alpaca_eval.py \\
    --model-outputs outputs/alpacaeval/sdft/model_outputs.json \\
    --name sdft \\
    --output-dir outputs/alpacaeval/sdft

Official GPT-4-Turbo judge::

  export JUDGE=openai OPENAI_API_KEY=...
  uv run python scripts/run_alpaca_eval.py --model-outputs ... --name sdft
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdft.alpacaeval_score import (  # noqa: E402
    DEFAULT_ANNOTATORS_CONFIG,
    evaluate_model_outputs,
    resolve_judge_mode,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="AlpacaEval 2-style win-rate (local open judge or official GPT-4-Turbo)"
    )
    parser.add_argument(
        "--model-outputs",
        required=True,
        help="JSON list of {instruction, output, generator} rows",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Model name on the leaderboard (default: generator field / file stem)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for leaderboard / annotations / summary.json",
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Cap annotations (smoke / cost control)",
    )
    parser.add_argument(
        "--judge",
        default=None,
        choices=["local", "openai"],
        help="local (default via JUDGE env) or openai GPT-4-Turbo",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="HF id for local judge (default: ALPACA_EVAL_LOCAL_JUDGE / Qwen/Qwen3.5-9B)",
    )
    parser.add_argument(
        "--annotators-config",
        default=DEFAULT_ANNOTATORS_CONFIG,
        help=f"OpenAI annotator config (default: {DEFAULT_ANNOTATORS_CONFIG})",
    )
    args = parser.parse_args(argv)

    mode = resolve_judge_mode(args.judge)
    summary = evaluate_model_outputs(
        args.model_outputs,
        name=args.name,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        annotators_config=args.annotators_config,
        judge=mode,
        judge_model=args.judge_model,
    )
    metrics = summary["metrics"]
    print(json.dumps(summary, indent=2))
    print()
    print(
        f"[{mode}] {metrics['name']}: win_rate={metrics.get('win_rate')}  "
        f"length_controlled_winrate={metrics.get('length_controlled_winrate')}  "
        f"n_total={metrics.get('n_total')}  avg_length={metrics.get('avg_length')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
