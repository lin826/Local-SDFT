#!/usr/bin/env python3
"""Run official AlpacaEval 2 judging on a model_outputs JSON file.

Example:
  export OPENAI_API_KEY=...
  uv sync --extra alpacaeval
  uv run python scripts/run_alpaca_eval.py \\
    --model-outputs outputs/alpacaeval/sdft/model_outputs.json \\
    --name sdft \\
    --output-dir outputs/alpacaeval/sdft
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
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Official AlpacaEval 2 win-rate / LC win-rate via alpaca_eval"
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
        "--annotators-config",
        default=DEFAULT_ANNOTATORS_CONFIG,
        help=f"Annotator config (default: {DEFAULT_ANNOTATORS_CONFIG})",
    )
    args = parser.parse_args(argv)

    summary = evaluate_model_outputs(
        args.model_outputs,
        name=args.name,
        output_dir=args.output_dir,
        max_instances=args.max_instances,
        annotators_config=args.annotators_config,
    )
    metrics = summary["metrics"]
    print(json.dumps(summary, indent=2))
    print()
    print(
        f"{metrics['name']}: win_rate={metrics.get('win_rate')}  "
        f"length_controlled_winrate={metrics.get('length_controlled_winrate')}  "
        f"n_total={metrics.get('n_total')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
