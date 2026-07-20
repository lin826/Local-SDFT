#!/usr/bin/env python3
"""Run batch-size-1 baselines: base / gold-SFT / SDFT / GRPO and score held-out prompts.

Mirrors the online-learning demo's per-example update style (LoRA, batch_size=1
for SFT/SDFT; GRPO uses batch_size=num_generations=2 for TRL group rollouts).

Example:
  uv run python scripts/run_batch1_comparison.py --num-train 32 --num-eval 16
  uv run python scripts/run_batch1_comparison.py --skip-train   # re-score adapters
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from peft import PeftModel

from sdft.config import load_config
from sdft.data import load_examples
from sdft.generate import generate_responses
from sdft.grpo_train import examples_to_grpo_jsonl
from sdft.peft_utils import adapter_ready
from sdft.rewards import instruction_reward
from sdft.utils import load_model, load_tokenizer, pick_device, to_model_device

REFUSAL_RE = re.compile(
    r"(?i)\b(i('m| am) sorry|i can('t|not) (assist|help)|as an ai)\b"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _run_module(module: str, *args: str) -> None:
    import os
    import subprocess

    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd = [sys.executable, "-m", module, *args]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT, env=env)


@torch.inference_mode()
def _generate_eval(
    model_name: str,
    prompts: list[str],
    *,
    adapter_dir: Path | None,
    max_new_tokens: int = 192,
    dtype: str = "float32",
) -> list[str]:
    device = pick_device()
    from sdft.config import ModelConfig

    cfg = ModelConfig(name=model_name, dtype=dtype)
    tokenizer = load_tokenizer(cfg)
    tokenizer.padding_side = "left"
    base = load_model(cfg, device)
    model = PeftModel.from_pretrained(base, str(adapter_dir)) if adapter_dir and adapter_ready(adapter_dir) else base
    model.eval()

    outs: list[str] = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = to_model_device(
            tokenizer(text, return_tensors="pt", add_special_tokens=False),
            model,
        )
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = out[:, enc["input_ids"].shape[1] :]
        outs.append(tokenizer.decode(new[0], skip_special_tokens=True).strip())
    return outs


def _score_arm(name: str, prompts: list[str], golds: list[str], generations: list[str], train_s: float) -> dict:
    rewards = instruction_reward(generations, gold=golds)
    refusals = sum(1 for g in generations if REFUSAL_RE.search(g))
    chars = [len(g) for g in generations]
    return {
        "arm": name,
        "n": len(generations),
        "train_seconds": round(train_s, 1),
        "mean_chars": round(sum(chars) / max(len(chars), 1), 1),
        "refusal_rate": round(refusals / max(len(generations), 1), 3),
        "mean_reward": round(sum(rewards) / max(len(rewards), 1), 3),
        "generations": [
            {"prompt": p, "gold": g, "output": o, "reward": r}
            for p, g, o, r in zip(prompts, golds, generations, rewards)
        ],
    }


_SUITES = {
    "230m": {
        "sdft": "configs/compare/batch1_sdft.yaml",
        "sft_gold": "configs/compare/batch1_sft_gold.yaml",
        "grpo": "configs/compare/batch1_grpo.yaml",
        "prefix": "batch1",
        "max_new_tokens": 192,
    },
    "1_2b": {
        "sdft": "configs/compare/batch1_1_2b_sdft.yaml",
        "sft_gold": "configs/compare/batch1_1_2b_sft_gold.yaml",
        "grpo": "configs/compare/batch1_1_2b_grpo.yaml",
        "prefix": "batch1_1_2b",
        "max_new_tokens": 128,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-train", type=int, default=32)
    parser.add_argument("--num-eval", type=int, default=16)
    parser.add_argument(
        "--suite",
        choices=sorted(_SUITES),
        default="230m",
        help="Model suite: 230m (default) or 1_2b (LFM2.5-1.2B-Thinking)",
    )
    parser.add_argument("--model", default=None, help="Override model.name for all arms")
    parser.add_argument("--dtype", default=None, help="Override model.dtype for eval loads")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-grpo", action="store_true", help="skip GRPO (slowest arm)")
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
    sdft_cfg.data.num_examples = args.num_train + args.num_eval
    all_examples = load_examples(sdft_cfg.data)
    train_ex = all_examples[: args.num_train]
    eval_ex = all_examples[args.num_train : args.num_train + args.num_eval]
    if len(train_ex) < args.num_train or len(eval_ex) < args.num_eval:
        raise SystemExit("not enough examples; lower --num-train/--num-eval")

    gold_path = ROOT / f"data/compare/{prefix}_gold.jsonl"
    sdft_path = ROOT / f"data/compare/{prefix}_sdft.jsonl"
    grpo_path = ROOT / f"data/compare/{prefix}_grpo.jsonl"
    _write_jsonl(
        gold_path,
        [{"prompt": e["prompt"], "response": e["response"], "sdft_response": e["response"]} for e in train_ex],
    )
    examples_to_grpo_jsonl(train_ex, grpo_path)

    sft_gold_cfg = load_config(ROOT / suite["sft_gold"])
    grpo_cfg = load_config(ROOT / suite["grpo"])
    arms: dict[str, Path | None] = {
        "base": None,
        "sft_gold": ROOT / sft_gold_cfg.training.output_dir,
        "sdft": ROOT / sdft_cfg.training.output_dir,
        "grpo": ROOT / grpo_cfg.grpo.output_dir,
    }
    train_times: dict[str, float] = {k: 0.0 for k in arms}

    if not args.skip_train:
        # SDFT teacher pass
        t0 = time.perf_counter()
        device = pick_device()
        sdft_cfg.data.num_examples = args.num_train
        sdft_cfg.generation.out_path = str(sdft_path)
        gens = generate_responses(sdft_cfg, train_ex, device)
        rows = []
        for ex, gen in zip(train_ex, gens):
            if not gen:
                continue
            rows.append({"prompt": ex["prompt"], "response": ex["response"], "sdft_response": gen})
        _write_jsonl(sdft_path, rows)
        print(f"SDFT teacher wrote {len(rows)} rows in {time.perf_counter() - t0:.1f}s")

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

    prompts = [e["prompt"] for e in eval_ex]
    golds = [e["response"] for e in eval_ex]
    results = []
    model_name = sdft_cfg.model.name
    max_new = int(suite["max_new_tokens"])
    for name, adapter in arms.items():
        if name == "grpo" and args.skip_grpo:
            continue
        if adapter is not None and not adapter_ready(adapter):
            print(f"skip {name}: adapter missing at {adapter}")
            continue
        print(f"evaluating {name}…", flush=True)
        gens = _generate_eval(
            model_name,
            prompts,
            adapter_dir=adapter,
            max_new_tokens=max_new,
            dtype=sdft_cfg.model.dtype,
        )
        results.append(_score_arm(name, prompts, golds, gens, train_times.get(name, 0.0)))

    out_default = (
        "outputs/compare/batch1_comparison.json"
        if args.suite == "230m"
        else "outputs/compare/batch1_1_2b_comparison.json"
    )
    out_path = ROOT / (args.out or out_default)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "num_train": args.num_train,
        "num_eval": args.num_eval,
        "model": model_name,
        "suite": args.suite,
        "dtype": sdft_cfg.model.dtype,
        "batch_size": 1,
        "arms": [{k: v for k, v in r.items() if k != "generations"} for r in results],
        "details": results,
    }
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("\n=== batch-size-1 comparison ===")
    print(f"{'arm':<12} {'reward':>8} {'refusal':>8} {'chars':>8} {'train_s':>10}")
    for r in results:
        print(
            f"{r['arm']:<12} {r['mean_reward']:>8.3f} {r['refusal_rate']:>8.3f} "
            f"{r['mean_chars']:>8.1f} {r['train_seconds']:>10.1f}"
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
