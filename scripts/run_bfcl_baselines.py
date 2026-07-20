#!/usr/bin/env python3
"""Train BFCL gold-SFT / SDFT / GRPO and evaluate on the held-out BFCL slice.

Mirrors ``run_batch1_comparison.py`` but trains on BFCL tool-call trajectories
(not Alpaca) and scores with the local BFCL AST harness.

Split: eval = first N/category (default 32); train = next M/category (default 16).

Example:
  uv run python scripts/run_bfcl_baselines.py --suite 230m
  uv run python scripts/run_bfcl_baselines.py --suite 1_2b --num-train-per-cat 8 --max-grpo-steps 8
  uv run python scripts/run_bfcl_baselines.py --skip-train  # re-eval adapters
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from sdft.bfcl.data import extract_user_text, functions_to_tools
from sdft.bfcl.eval import _generate_one, run_bfcl_eval
from sdft.bfcl.train_data import (
    TRAIN_CATEGORIES,
    build_grpo_row,
    build_sft_row,
    load_bfcl_train_eval_split,
    write_jsonl,
    write_split_manifest,
)
from sdft.config import load_config
from sdft.peft_utils import adapter_ready
from sdft.utils import load_model, load_tokenizer, pick_device

_SUITES = {
    "230m": {
        "sdft": "configs/compare/bfcl_sdft.yaml",
        "sft_gold": "configs/compare/bfcl_sft_gold.yaml",
        "grpo": "configs/compare/bfcl_grpo.yaml",
        "eval": "configs/bfcl_eval.yaml",
        "prefix": "bfcl",
        "manifest": "data/bfcl/split_manifest.json",
        "out": "outputs/compare/bfcl_comparison.json",
        "eval_out_dir": "outputs/benchmarks/bfcl/230m_trained",
    },
    "1_2b": {
        "sdft": "configs/compare/bfcl_1_2b_sdft.yaml",
        "sft_gold": "configs/compare/bfcl_1_2b_sft_gold.yaml",
        "grpo": "configs/compare/bfcl_1_2b_grpo.yaml",
        "eval": "configs/bfcl_eval_1_2b.yaml",
        "prefix": "bfcl_1_2b",
        "manifest": "data/bfcl/split_manifest_1_2b.json",
        "out": "outputs/compare/bfcl_1_2b_comparison.json",
        "eval_out_dir": "outputs/benchmarks/bfcl/1_2b_trained",
    },
}


def _run_module(module: str, *args: str) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


@torch.inference_mode()
def _teacher_sdft_rows(
    cfg,
    train_rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Generate in-distribution tool-call rewrites with tools= (BFCL eval path)."""
    device = pick_device()
    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, device)
    model.eval()

    out_rows: list[dict[str, Any]] = []
    for i, row in enumerate(train_rows):
        sft = build_sft_row(row, tokenizer)
        if sft is None:
            continue
        user_text = extract_user_text(row["question"])
        tools = functions_to_tools(row["function"])
        text, _, _ = _generate_one(
            model,
            tokenizer,
            user_text=user_text,
            tools=tools,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )
        # Strip chat end tokens (same as eval).
        for stop in ("<|im_end|>", "<|endoftext|>"):
            if stop in text:
                text = text.split(stop)[0]
        gen = text.strip()
        if not gen:
            continue
        out_rows.append(
            {
                "id": sft["id"],
                "category": sft["category"],
                "prompt": sft["prompt"],
                "response": sft["response"],
                "sdft_response": gen,
                "user_text": sft["user_text"],
            }
        )
        if (i + 1) % 8 == 0 or i == 0:
            print(f"  teacher [{i+1}/{len(train_rows)}] {row['id']}", flush=True)
    return out_rows


def _arm_summary(result: dict[str, Any], train_s: float) -> dict[str, Any]:
    cats = result.get("categories") or {}
    return {
        "arm": result.get("arm"),
        "train_seconds": round(train_s, 1),
        "overall": result.get("overall"),
        "categories": {
            k: {"n": v["n"], "accuracy": v["accuracy"]} for k, v in cats.items()
        },
        "latency": result.get("latency"),
        "adapter": result.get("adapter"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=sorted(_SUITES), default="230m")
    parser.add_argument("--num-train-per-cat", type=int, default=16)
    parser.add_argument("--num-eval-per-cat", type=int, default=32)
    parser.add_argument("--model", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true")
    parser.add_argument("--max-grpo-steps", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    suite = _SUITES[args.suite]
    prefix = suite["prefix"]
    sdft_cfg = load_config(ROOT / suite["sdft"])
    if args.model:
        sdft_cfg.model.name = args.model
    if args.dtype:
        sdft_cfg.model.dtype = args.dtype

    split = load_bfcl_train_eval_split(
        categories=TRAIN_CATEGORIES,
        num_train_per_cat=args.num_train_per_cat,
        num_eval_per_cat=args.num_eval_per_cat,
        cache_dir=ROOT / "data/bfcl",
    )
    tokenizer = load_tokenizer(sdft_cfg.model)

    gold_rows = [r for row in split["train"] if (r := build_sft_row(row, tokenizer))]
    grpo_rows = [r for row in split["train"] if (r := build_grpo_row(row, tokenizer))]
    gold_path = ROOT / f"data/compare/{prefix}_gold.jsonl"
    sdft_path = ROOT / f"data/compare/{prefix}_sdft.jsonl"
    grpo_path = ROOT / f"data/compare/{prefix}_grpo.jsonl"
    write_jsonl(gold_path, gold_rows)
    write_jsonl(grpo_path, grpo_rows)
    write_split_manifest(
        ROOT / suite["manifest"],
        split,
        extra={
            "suite": args.suite,
            "model": sdft_cfg.model.name,
            "gold_file": str(gold_path.relative_to(ROOT)),
            "grpo_file": str(grpo_path.relative_to(ROOT)),
            "gold_n": len(gold_rows),
            "grpo_n": len(grpo_rows),
        },
    )
    print(
        f"split train={len(split['train'])} eval={len(split['eval'])} "
        f"gold={len(gold_rows)} grpo={len(grpo_rows)}"
    )

    sft_gold_cfg = load_config(ROOT / suite["sft_gold"])
    grpo_cfg = load_config(ROOT / suite["grpo"])
    if args.model:
        sft_gold_cfg.model.name = args.model
        grpo_cfg.model.name = args.model
    if args.dtype:
        sft_gold_cfg.model.dtype = args.dtype
        grpo_cfg.model.dtype = args.dtype

    arms: dict[str, Path | None] = {
        "base": None,
        "sft_gold": ROOT / sft_gold_cfg.training.output_dir,
        "sdft": ROOT / sdft_cfg.training.output_dir,
        "grpo": ROOT / grpo_cfg.grpo.output_dir,
    }
    train_times: dict[str, float] = {k: 0.0 for k in arms}

    if not args.skip_train:
        t0 = time.perf_counter()
        max_new = int(sdft_cfg.generation.max_new_tokens or 256)
        print("SDFT teacher pass…", flush=True)
        teacher_rows = _teacher_sdft_rows(sdft_cfg, split["train"], max_new_tokens=max_new)
        write_jsonl(sdft_path, teacher_rows)
        print(f"SDFT teacher wrote {len(teacher_rows)} rows in {time.perf_counter() - t0:.1f}s")

        t0 = time.perf_counter()
        _run_module(
            "sdft.train",
            "--config",
            suite["sft_gold"],
            "--data",
            str(gold_path),
            "--target",
            "gold",
            "--output-dir",
            str(arms["sft_gold"]),
        )
        train_times["sft_gold"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        _run_module(
            "sdft.train",
            "--config",
            suite["sdft"],
            "--data",
            str(sdft_path),
            "--target",
            "sdft",
            "--output-dir",
            str(arms["sdft"]),
        )
        train_times["sdft"] = time.perf_counter() - t0

        if not args.skip_grpo:
            t0 = time.perf_counter()
            grpo_args = [
                "sdft.grpo_train",
                "--config",
                suite["grpo"],
                "--data",
                str(grpo_path),
                "--output-dir",
                str(arms["grpo"]),
            ]
            if args.max_grpo_steps is not None:
                grpo_args.extend(["--max-steps", str(args.max_grpo_steps)])
            _run_module(*grpo_args)
            train_times["grpo"] = time.perf_counter() - t0

    eval_cfg = load_config(ROOT / suite["eval"])
    if args.model:
        eval_cfg.model.name = args.model
    if args.dtype:
        eval_cfg.model.dtype = args.dtype
    eval_ids = split["eval_ids"]
    eval_out_dir = ROOT / suite["eval_out_dir"]
    eval_out_dir.mkdir(parents=True, exist_ok=True)

    arm_results: list[dict[str, Any]] = []
    for name, adapter in arms.items():
        if name == "grpo" and args.skip_grpo:
            continue
        if adapter is not None and not adapter_ready(adapter):
            print(f"skip {name}: adapter missing at {adapter}")
            continue
        print(f"evaluating {name} on held-out BFCL…", flush=True)
        result = run_bfcl_eval(
            eval_cfg,
            categories=list(TRAIN_CATEGORIES),
            num_examples=args.num_eval_per_cat,
            adapter_dir=adapter,
            arm_name=name,
            example_ids=eval_ids,
        )
        out_arm = eval_out_dir / f"{name}.json"
        out_arm.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        arm_results.append(_arm_summary(result, train_times.get(name, 0.0)))
        print(
            f"  {name}: overall={result['overall']['accuracy']:.1%} "
            f"mean_lat={result['latency']['mean_s']:.2f}s"
        )

    out_path = ROOT / (args.out or suite["out"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "benchmark": "bfcl_v3_trained_baselines",
        "suite": args.suite,
        "model": sdft_cfg.model.name,
        "dtype": sdft_cfg.model.dtype,
        "num_train_per_cat": args.num_train_per_cat,
        "num_eval_per_cat": args.num_eval_per_cat,
        "train_n": len(split["train"]),
        "eval_n": len(split["eval"]),
        "categories": list(TRAIN_CATEGORIES),
        "split_note": (
            "Eval = first num_eval_per_cat / category; "
            "train = following num_train_per_cat (no id/question overlap)."
        ),
        "arms": arm_results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\n=== BFCL-trained baselines ===")
    print(
        f"{'arm':<12} {'overall':>8} {'simple':>8} {'multiple':>8} "
        f"{'parallel':>8} {'irrel':>8} {'train_s':>10}"
    )
    for r in arm_results:
        cats = r.get("categories") or {}
        overall = (r.get("overall") or {}).get("accuracy", 0.0)
        print(
            f"{r['arm']:<12} {overall:>7.1%} "
            f"{cats.get('simple', {}).get('accuracy', 0):>7.1%} "
            f"{cats.get('multiple', {}).get('accuracy', 0):>7.1%} "
            f"{cats.get('parallel', {}).get('accuracy', 0):>7.1%} "
            f"{cats.get('irrelevance', {}).get('accuracy', 0):>7.1%} "
            f"{r['train_seconds']:>10.1f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
