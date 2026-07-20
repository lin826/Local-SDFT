#!/usr/bin/env python3
"""Build BFCL gold / GRPO training jsonl with a held-out eval split.

Eval = first ``--num-eval-per-cat`` rows per category (same slice as default
BFCL eval). Train = the following ``--num-train-per-cat`` rows.

Produces:
  data/compare/bfcl[_1_2b]_gold.jsonl
  data/compare/bfcl[_1_2b]_grpo.jsonl
  data/bfcl/split_manifest[_1_2b].json

SDFT teacher rewrites are produced by ``scripts/run_bfcl_baselines.py``.

Example:
  uv run python scripts/build_bfcl_train_data.py --num-train-per-cat 16 --num-eval-per-cat 32
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdft.bfcl.train_data import (  # noqa: E402
    TRAIN_CATEGORIES,
    build_grpo_row,
    build_sft_row,
    load_bfcl_train_eval_split,
    write_jsonl,
    write_split_manifest,
)
from sdft.config import ModelConfig  # noqa: E402
from sdft.utils import load_tokenizer  # noqa: E402

_SUITES = {
    "230m": {
        "model": "LiquidAI/LFM2.5-230M",
        "prefix": "bfcl",
        "manifest": "data/bfcl/split_manifest.json",
    },
    "1_2b": {
        "model": "LiquidAI/LFM2.5-1.2B-Instruct",
        "prefix": "bfcl_1_2b",
        "manifest": "data/bfcl/split_manifest_1_2b.json",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(_SUITES), default="230m")
    parser.add_argument("--num-train-per-cat", type=int, default=16)
    parser.add_argument("--num-eval-per-cat", type=int, default=32)
    parser.add_argument("--model", default=None, help="Override tokenizer/model name")
    parser.add_argument("--cache-dir", default="data/bfcl")
    args = parser.parse_args()

    suite = _SUITES[args.suite]
    model_name = args.model or suite["model"]
    prefix = suite["prefix"]

    split = load_bfcl_train_eval_split(
        categories=TRAIN_CATEGORIES,
        num_train_per_cat=args.num_train_per_cat,
        num_eval_per_cat=args.num_eval_per_cat,
        cache_dir=ROOT / args.cache_dir,
    )
    tokenizer = load_tokenizer(ModelConfig(name=model_name))

    gold_rows = []
    grpo_rows = []
    for row in split["train"]:
        sft = build_sft_row(row, tokenizer)
        if sft is None:
            continue
        gold_rows.append(sft)
        grpo = build_grpo_row(row, tokenizer)
        if grpo is not None:
            grpo_rows.append(grpo)

    gold_path = write_jsonl(ROOT / f"data/compare/{prefix}_gold.jsonl", gold_rows)
    grpo_path = write_jsonl(ROOT / f"data/compare/{prefix}_grpo.jsonl", grpo_rows)
    manifest = write_split_manifest(
        ROOT / suite["manifest"],
        split,
        extra={
            "suite": args.suite,
            "model": model_name,
            "gold_file": str(gold_path.relative_to(ROOT)),
            "grpo_file": str(grpo_path.relative_to(ROOT)),
            "gold_n": len(gold_rows),
            "grpo_n": len(grpo_rows),
        },
    )
    print(f"train={len(split['train'])} eval={len(split['eval'])} gold={len(gold_rows)}")
    print(f"wrote {gold_path}")
    print(f"wrote {grpo_path}")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
