"""Standalone OpenClaw-RL ReTool evaluation harness for LFM2.5-230M."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

from sdft.config import Config, load_config
from sdft.toolcall.format import DEFAULT_COT_LINE
from sdft.toolcall.loop import ToolLoopConfig, run_tool_loop
from sdft.toolcall.scoring import score_openclaw_solution
from sdft.utils import load_model, load_tokenizer, pick_device


def _load_eval_rows(cfg: Config) -> list[dict[str, Any]]:
    eval_cfg = cfg.openclaw_eval
    if eval_cfg.data_file:
        rows = []
        with Path(eval_cfg.data_file).open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        ds = load_dataset(eval_cfg.dataset, split=eval_cfg.split)
        rows = [dict(row) for row in ds]

    if eval_cfg.num_examples is not None:
        rows = rows[: eval_cfg.num_examples]
    return rows


def _extract_prompt(row: dict[str, Any]) -> str:
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
        if prompt and isinstance(prompt[0], dict):
            return str(prompt[0].get("content", "")).strip()
    if isinstance(prompt, str):
        return prompt.strip()
    for key in ("question", "instruction", "input"):
        if row.get(key):
            return str(row[key]).strip()
    raise ValueError(f"Could not extract prompt from row keys: {sorted(row)}")


def _extract_label(row: dict[str, Any]) -> str:
    for key in ("label", "answer", "ground_truth"):
        if row.get(key) is not None:
            return str(row[key]).strip()
    reward = row.get("reward_model")
    if isinstance(reward, dict) and reward.get("ground_truth") is not None:
        return str(reward["ground_truth"]).strip()
    raise ValueError(f"Could not extract label from row keys: {sorted(row)}")


def run_eval(cfg: Config) -> dict[str, Any]:
    device = pick_device()
    print(f"device: {device}")
    print(f"model: {cfg.model.name}")

    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, device)
    model.eval()
    return run_eval_with_model(cfg, model, tokenizer, device)


def run_eval_with_model(
    cfg: Config,
    model,
    tokenizer,
    device: str,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run eval using a pre-loaded model (for ablation sweeps)."""
    if rows is None:
        rows = _load_eval_rows(cfg)
    print(f"loaded {len(rows)} eval examples")

    few_shot_k = cfg.openclaw_eval.few_shot_k
    cot_line = cfg.toolcall.cot_line
    loop_cfg = ToolLoopConfig(
        max_rounds=cfg.toolcall.max_rounds,
        max_new_tokens=cfg.toolcall.max_new_tokens,
        temperature=cfg.toolcall.temperature,
        top_p=cfg.toolcall.top_p,
        format=cfg.toolcall.format,
        system_prompt=cfg.toolcall.system_prompt,
        max_context_chars=cfg.toolcall.max_context_chars,
        max_obs_chars=cfg.toolcall.max_obs_chars,
        sandbox_timeout_s=cfg.toolcall.sandbox_timeout_s,
        few_shot_k=few_shot_k,
        cot_line=cot_line,
    )
    print(f"few_shot_k: {few_shot_k}")
    if cot_line:
        print(f"cot_line: {cot_line!r}")

    rng = random.Random(cfg.openclaw_eval.seed)
    results: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        prompt = _extract_prompt(row)
        label = _extract_label(row)
        sample_scores: list[float] = []
        sample_accs: list[bool] = []
        sample_details: list[dict[str, Any]] = []

        for sample_idx in range(cfg.openclaw_eval.n_samples):
            if cfg.openclaw_eval.n_samples > 1:
                loop_cfg.temperature = max(cfg.toolcall.temperature, 0.01 if sample_idx else 0.0)

            loop_result = run_tool_loop(
                model,
                tokenizer,
                prompt,
                cfg=loop_cfg,
                device=device,
            )
            solution = prompt + loop_result.response_text
            score = score_openclaw_solution(
                solution,
                label,
                strict_box_verify=cfg.openclaw_eval.strict_box_verify,
            )
            sample_scores.append(float(score["score"]))
            sample_accs.append(bool(score["acc"]))
            sample_details.append(
                {
                    "sample": sample_idx,
                    "acc": score["acc"],
                    "pred": score["pred"],
                    "tool_call_count": loop_result.tool_call_count,
                    "finish_reason": loop_result.finish_reason,
                    "response_text": loop_result.response_text,
                }
            )

        best_acc = any(sample_accs)
        results.append(
            {
                "index": idx,
                "prompt": prompt,
                "label": label,
                "mean_score": statistics.mean(sample_scores),
                "pass_at_k": best_acc,
                "samples": sample_details,
            }
        )
        print(
            f"  [{idx + 1}/{len(rows)}] pass@{cfg.openclaw_eval.n_samples}={best_acc} "
            f"tools={sample_details[0]['tool_call_count']} pred={sample_details[0]['pred']!r}",
            flush=True,
        )
        rng.randint(0, 1)

    pass_at_k = sum(1 for r in results if r["pass_at_k"]) / len(results) if results else 0.0
    mean_score = statistics.mean(r["mean_score"] for r in results) if results else 0.0
    mean_tools = statistics.mean(r["samples"][0]["tool_call_count"] for r in results) if results else 0.0

    summary = {
        "benchmark": "openclaw-rl/retool",
        "model": cfg.model.name,
        "dataset": cfg.openclaw_eval.dataset,
        "data_file": cfg.openclaw_eval.data_file,
        "num_examples": len(results),
        "n_samples": cfg.openclaw_eval.n_samples,
        "few_shot_k": few_shot_k,
        "cot_line": cot_line,
        "toolcall_format": cfg.toolcall.format,
        "pass_at_k": pass_at_k,
        "mean_score": mean_score,
        "mean_tool_calls": mean_tools,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    return summary


def _write_outputs(summary: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = out_dir / f"openclaw_rl_{stamp}.json"
    latest_path = out_dir / "latest.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    latest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {json_path}")
    print(f"wrote {latest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/openclaw_rl_eval.yaml")
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--format",
        choices=["auto", "openclaw", "lfm"],
        default=None,
        help="tool-call conversation format",
    )
    parser.add_argument(
        "--one-shot",
        action="store_true",
        help="prepend one tool-use demonstration (sets few_shot_k=1; not pass@k)",
    )
    parser.add_argument(
        "--few-shot-k",
        type=int,
        default=None,
        help="number of tool-use demos to prepend (overrides --one-shot)",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="append default one-line CoT cue to system prompt",
    )
    parser.add_argument(
        "--cot-line",
        default=None,
        help="custom one-line CoT cue (implies CoT when set)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="per-turn generation cap (default: toolcall.max_new_tokens in config)",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="max tool-loop rounds before stopping (default: toolcall.max_rounds)",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="prompt char budget per round (default: toolcall.max_context_chars)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.num_examples is not None:
        cfg.openclaw_eval.num_examples = args.num_examples
    if args.n_samples is not None:
        cfg.openclaw_eval.n_samples = args.n_samples
    if args.out_dir is not None:
        cfg.openclaw_eval.out_dir = args.out_dir
    if args.format is not None:
        cfg.toolcall.format = args.format
    if args.few_shot_k is not None:
        cfg.openclaw_eval.few_shot_k = args.few_shot_k
    elif args.one_shot:
        cfg.openclaw_eval.few_shot_k = 1
    if args.cot_line is not None:
        cfg.toolcall.cot_line = args.cot_line
    elif args.cot:
        cfg.toolcall.cot_line = DEFAULT_COT_LINE
    if args.max_new_tokens is not None:
        cfg.toolcall.max_new_tokens = args.max_new_tokens
    if args.max_rounds is not None:
        cfg.toolcall.max_rounds = args.max_rounds
    if args.max_context_chars is not None:
        cfg.toolcall.max_context_chars = args.max_context_chars

    summary = run_eval(cfg)
    print(
        f"\npass@{cfg.openclaw_eval.n_samples}={summary['pass_at_k']:.3f} "
        f"mean_score={summary['mean_score']:.3f} "
        f"mean_tool_calls={summary['mean_tool_calls']:.2f}"
    )
    _write_outputs(summary, Path(cfg.openclaw_eval.out_dir))


if __name__ == "__main__":
    main()
