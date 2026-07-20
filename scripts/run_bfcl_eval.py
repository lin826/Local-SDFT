#!/usr/bin/env python3
"""Convenience wrapper for local BFCL-v3 subset evaluation.

Examples::

  uv run python scripts/run_bfcl_eval.py
  uv run python scripts/run_bfcl_eval.py --suite 1_2b --num-examples 16
  uv run python scripts/run_bfcl_eval.py --adapter outputs/compare/batch1-sdft --arm sdft
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

_SUITES = {
    "230m": "configs/bfcl_eval.yaml",
    "1_2b": "configs/bfcl_eval_1_2b.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(_SUITES), default="230m")
    parser.add_argument("--config", default=None, help="Override YAML config path")
    parser.add_argument("--categories", default=None)
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", default=None)
    args, extra = parser.parse_known_args()

    config = args.config or _SUITES[args.suite]
    cmd = [sys.executable, "-m", "sdft.bfcl.eval", "--config", config]
    if args.categories:
        cmd.extend(["--categories", args.categories])
    if args.num_examples is not None:
        cmd.extend(["--num-examples", str(args.num_examples)])
    if args.adapter:
        cmd.extend(["--adapter", args.adapter])
    if args.arm:
        cmd.extend(["--arm", args.arm])
    if args.out:
        cmd.extend(["--out", args.out])
    if args.model:
        cmd.extend(["--model", args.model])
    if args.dtype:
        cmd.extend(["--dtype", args.dtype])
    cmd.extend(extra)

    print("+", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
