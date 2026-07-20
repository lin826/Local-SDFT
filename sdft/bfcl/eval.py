"""Run local BFCL-v3 subset evaluation with transformers (MPS/CUDA/CPU).

Example::

    uv run python -m sdft.bfcl.eval --config configs/bfcl_eval.yaml
    uv run python -m sdft.bfcl.eval --config configs/bfcl_eval_1_2b.yaml \\
        --categories simple,irrelevance --num-examples 32
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from sdft.bfcl.ast_score import score_bfcl_example
from sdft.bfcl.data import (
    SUPPORTED_CATEGORIES,
    extract_user_text,
    functions_to_tools,
    load_bfcl_category,
)
from sdft.bfcl.parse import parse_function_calls
from sdft.config import Config, load_config
from sdft.peft_utils import adapter_ready, load_chat_model
from sdft.utils import load_tokenizer, pick_device


def _json_default(obj: Any) -> Any:
    if isinstance(obj, set):
        return sorted(obj, key=lambda x: str(x))
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _parse_categories(raw: str | list[str] | None, default: list[str]) -> list[str]:
    if raw is None:
        return list(default)
    if isinstance(raw, list):
        cats = raw
    else:
        cats = [c.strip() for c in str(raw).split(",") if c.strip()]
    unknown = [c for c in cats if c not in SUPPORTED_CATEGORIES]
    if unknown:
        raise SystemExit(
            f"Unknown categories {unknown}; supported: {list(SUPPORTED_CATEGORIES)}"
        )
    return cats


@torch.inference_mode()
def _generate_one(
    model,
    tokenizer,
    *,
    user_text: str,
    tools: list[dict[str, Any]],
    device: str,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float, int]:
    messages = [{"role": "user", "content": user_text}]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        # Older templates without tools= kwarg — inject a system tool list.
        sys = "List of tools: " + json.dumps(tools, ensure_ascii=False)
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": sys}, *messages],
            tokenize=False,
            add_generation_prompt=True,
        )

    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    out = model.generate(**enc, **gen_kwargs)
    if device == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    new = out[:, enc["input_ids"].shape[1] :]
    text = tokenizer.decode(new[0], skip_special_tokens=False)
    # Strip trailing chat end tokens for cleaner logs / parsers.
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if stop in text:
            text = text.split(stop)[0]
    return text.strip(), elapsed, int(new.shape[1])


def run_bfcl_eval(
    cfg: Config,
    *,
    categories: list[str] | None = None,
    num_examples: int | None = None,
    adapter_dir: str | Path | None = None,
    arm_name: str = "base",
    example_ids: set[str] | list[str] | None = None,
    exclude_ids: set[str] | list[str] | None = None,
) -> dict[str, Any]:
    bfcl = cfg.bfcl_eval
    cats = _parse_categories(categories, list(bfcl.categories))
    n = bfcl.num_examples if num_examples is None else num_examples
    device = pick_device()
    print(f"device: {device}")
    print(f"model: {cfg.model.name}")
    print(f"arm: {arm_name}")
    print(f"categories: {cats}")
    print(f"num_examples/category: {n}")
    if example_ids is not None:
        print(f"example_ids filter: {len(set(example_ids))} ids")
    if exclude_ids:
        print(f"exclude_ids filter: {len(set(exclude_ids))} ids")

    tokenizer = load_tokenizer(cfg.model)
    model = load_chat_model(cfg, device, adapter_dir=adapter_dir)
    model.eval()

    per_category: dict[str, Any] = {}
    all_details: list[dict[str, Any]] = []
    latencies: list[float] = []
    gen_tokens: list[int] = []

    for cat in cats:
        rows = load_bfcl_category(
            cat,
            cache_dir=bfcl.cache_dir,
            num_examples=n if example_ids is None else None,
            force_download=bfcl.force_download,
            example_ids=example_ids,
            exclude_ids=exclude_ids,
        )
        if example_ids is not None and n is not None:
            # Keep category order but still allow a per-cat cap when filtering by ids.
            rows = rows[:n]
        correct = 0
        cat_details: list[dict[str, Any]] = []
        print(f"\n=== {cat} (n={len(rows)}) ===", flush=True)

        for i, row in enumerate(rows):
            user_text = extract_user_text(row["question"])
            tools = functions_to_tools(row["function"])
            text, elapsed, n_tok = _generate_one(
                model,
                tokenizer,
                user_text=user_text,
                tools=tools,
                device=device,
                max_new_tokens=bfcl.max_new_tokens,
                temperature=bfcl.temperature,
            )
            calls = parse_function_calls(text)
            scored = score_bfcl_example(
                category=cat,
                model_calls=calls,
                ground_truth=row.get("ground_truth"),
                functions=row["function"],
            )
            correct += int(scored["acc"])
            latencies.append(elapsed)
            gen_tokens.append(n_tok)
            detail = {
                "id": row["id"],
                "category": cat,
                "acc": scored["acc"],
                "error": scored["error"],
                "n_calls": scored["n_calls"],
                "latency_s": round(elapsed, 3),
                "gen_tokens": n_tok,
                "parsed": calls,
                "response": text,
            }
            cat_details.append(detail)
            all_details.append(detail)
            if (i + 1) % 10 == 0 or i == 0:
                print(
                    f"  [{i+1}/{len(rows)}] acc_so_far={correct/(i+1):.3f} "
                    f"last={row['id']} ok={scored['acc']}",
                    flush=True,
                )

        acc = correct / max(len(rows), 1)
        per_category[cat] = {
            "n": len(rows),
            "correct": correct,
            "accuracy": round(acc, 4),
        }
        print(f"  -> {cat} accuracy: {acc:.3%} ({correct}/{len(rows)})")

    overall_n = sum(v["n"] for v in per_category.values())
    overall_correct = sum(v["correct"] for v in per_category.values())
    mean_lat = statistics.mean(latencies) if latencies else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    toks_per_s = (
        sum(gen_tokens) / sum(latencies) if latencies and sum(latencies) > 0 else 0.0
    )

    summary = {
        "benchmark": "bfcl_v3_local_subset",
        "subset_note": (
            "Local AST/irrelevance subset of BFCL-v3 (simple, multiple, parallel, "
            "parallel_multiple, irrelevance). Not official leaderboard scores; "
            "no live/multi-turn/executable/web-search."
        ),
        "model": cfg.model.name,
        "dtype": cfg.model.dtype,
        "arm": arm_name,
        "adapter": str(adapter_dir) if adapter_dir else None,
        "device": device,
        "categories": per_category,
        "overall": {
            "n": overall_n,
            "correct": overall_correct,
            "accuracy": round(overall_correct / max(overall_n, 1), 4),
        },
        "latency": {
            "mean_s": round(mean_lat, 3),
            "median_s": round(p50, 3),
            "gen_tok_per_s": round(toks_per_s, 1),
            "mean_gen_tokens": round(statistics.mean(gen_tokens), 1) if gen_tokens else 0.0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "details": all_details,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/bfcl_eval.yaml")
    parser.add_argument(
        "--categories",
        default=None,
        help="Comma-separated subset (default: from config)",
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--adapter", default=None, help="Optional LoRA adapter dir")
    parser.add_argument("--arm", default=None, help="Label for this run (default: base/adapter)")
    parser.add_argument("--out", default=None, help="Override output JSON path")
    parser.add_argument("--model", default=None, help="Override model.name")
    parser.add_argument("--dtype", default=None, help="Override model.dtype")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model.name = args.model
    if args.dtype:
        cfg.model.dtype = args.dtype

    cats = _parse_categories(args.categories, list(cfg.bfcl_eval.categories))
    adapter = Path(args.adapter) if args.adapter else None
    if adapter is not None and not adapter_ready(adapter):
        raise SystemExit(f"adapter not ready: {adapter}")
    arm = args.arm or ("adapter" if adapter else "base")

    summary = run_bfcl_eval(
        cfg,
        categories=cats,
        num_examples=args.num_examples,
        adapter_dir=adapter,
        arm_name=arm,
    )

    out = Path(args.out) if args.out else Path(cfg.bfcl_eval.out_dir) / f"{arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write slim summary + full details
    out.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    slim = {k: v for k, v in summary.items() if k != "details"}
    print("\n=== BFCL local subset ===")
    print(json.dumps(slim, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
