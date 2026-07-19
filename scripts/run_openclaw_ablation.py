#!/usr/bin/env python3
"""Sweep OpenClaw tool-use ablation conditions and write comparison artifacts.

Runs all baseline × SDFT conditions on the same item bank, then selects
examples where only SDFT-family conditions succeed (prefer tool_call_count >= 1).

Usage:
  uv run python scripts/run_openclaw_ablation.py
  uv run python scripts/run_openclaw_ablation.py --skip-train --merged outputs/openclaw-tooluse-merged
  uv run python scripts/run_openclaw_ablation.py --skip-data --skip-train
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdft.config import Config, load_config  # noqa: E402
from sdft.toolcall.format import DEFAULT_COT_LINE  # noqa: E402
from sdft.toolcall.openclaw_eval import _load_eval_rows, run_eval_with_model  # noqa: E402
from sdft.toolcall.split_guard import (  # noqa: E402
    assert_no_question_overlap,
    load_questions_from_jsonl,
)
from sdft.utils import load_model, load_tokenizer, pick_device  # noqa: E402

ABLATION_DIR = ROOT / "outputs" / "benchmarks" / "openclaw-rl" / "ablation"
DEFAULT_MERGED = ROOT / "outputs" / "openclaw-tooluse-merged"
DEFAULT_EVAL_CONFIG = ROOT / "configs" / "openclaw_demo_eval.yaml"
DEFAULT_TRAIN_CONFIG = ROOT / "configs" / "openclaw_tooluse_sdft.yaml"
DEFAULT_TRAIN_JSONL = ROOT / "data" / "openclaw_tooluse.jsonl"
FORBIDDEN_FEW_SHOT = ["What is 3 + 5?"]

CONDITIONS: list[dict[str, Any]] = [
    {"id": "ZS", "few_shot_k": 0, "cot_line": None, "sdft": False},
    {"id": "OS", "few_shot_k": 1, "cot_line": None, "sdft": False},
    {"id": "OS+CoT", "few_shot_k": 1, "cot_line": DEFAULT_COT_LINE, "sdft": False},
    {"id": "CoT-only", "few_shot_k": 0, "cot_line": DEFAULT_COT_LINE, "sdft": False},
    {"id": "SDFT-ZS", "few_shot_k": 0, "cot_line": None, "sdft": True},
    {"id": "SDFT+OS", "few_shot_k": 1, "cot_line": None, "sdft": True},
    {"id": "SDFT+OS+CoT", "few_shot_k": 1, "cot_line": DEFAULT_COT_LINE, "sdft": True},
]

SDFT_IDS = {c["id"] for c in CONDITIONS if c["sdft"]}


def _run_cmd(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n==> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def _prepare_data_and_train(*, skip_train: bool) -> None:
    _run_cmd(["uv", "sync", "--extra", "toolcall"])
    _run_cmd(
        ["uv", "run", "python", "scripts/build_openclaw_tooluse_data.py", "--write-sdft"]
    )
    if skip_train:
        return
    _run_cmd(
        [
            "uv",
            "run",
            "python",
            "-m",
            "sdft.train",
            "--config",
            str(DEFAULT_TRAIN_CONFIG),
            "--data",
            "data/openclaw_tooluse_sdft.jsonl",
        ]
    )
    _run_cmd(
        [
            "uv",
            "run",
            "python",
            "-m",
            "sdft.merge",
            "--config",
            str(DEFAULT_TRAIN_CONFIG),
            "--out",
            str(DEFAULT_MERGED),
        ]
    )


def _apply_condition(cfg: Config, condition: dict[str, Any], tool_format: str) -> Config:
    cfg.openclaw_eval.few_shot_k = int(condition["few_shot_k"])
    cfg.toolcall.cot_line = condition["cot_line"]
    cfg.toolcall.format = tool_format
    cfg.openclaw_eval.out_dir = str(ABLATION_DIR / condition["id"])
    return cfg


def _correctness_matrix(all_results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    if not all_results:
        return []
    first = next(iter(all_results.values()))
    n = len(first["results"])
    matrix: list[dict[str, Any]] = []
    for idx in range(n):
        row: dict[str, Any] = {
            "index": idx,
            "prompt": first["results"][idx]["prompt"],
            "label": first["results"][idx]["label"],
            "conditions": {},
        }
        for cid, summary in all_results.items():
            r = summary["results"][idx]
            sample = r["samples"][0]
            row["conditions"][cid] = {
                "pass": r["pass_at_k"],
                "pred": sample["pred"],
                "tool_call_count": sample["tool_call_count"],
                "finish_reason": sample["finish_reason"],
            }
        matrix.append(row)
    return matrix


def _find_demo_wins(matrix: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wins: list[dict[str, Any]] = []
    for row in matrix:
        cond = row["conditions"]
        sdft_ok = any(cond[c]["pass"] for c in SDFT_IDS if c in cond)
        base_ok = any(cond[c]["pass"] for c in cond if c not in SDFT_IDS)
        if not sdft_ok or base_ok:
            continue
        sdft_tools = max(
            cond[c]["tool_call_count"] for c in SDFT_IDS if c in cond and cond[c]["pass"]
        )
        if sdft_tools < 1:
            continue
        wins.append(row)
    wins.sort(
        key=lambda r: (
            -max(r["conditions"][c]["tool_call_count"] for c in SDFT_IDS if c in r["conditions"]),
            r["index"],
        )
    )
    return wins


def _find_near_misses(matrix: list[dict[str, Any]], limit: int = 5) -> list[dict[str, Any]]:
    near: list[dict[str, Any]] = []
    for row in matrix:
        cond = row["conditions"]
        sdft_ok = any(cond[c]["pass"] for c in SDFT_IDS if c in cond)
        base_ok = any(cond[c]["pass"] for c in cond if c not in SDFT_IDS)
        if sdft_ok and base_ok:
            near.append(row)
    return near[:limit]


def _run_condition_group(
    *,
    model_name: str,
    conditions: list[dict[str, Any]],
    eval_cfg: Config,
    rows: list[dict[str, Any]],
    device: str,
    tool_format: str,
) -> dict[str, dict[str, Any]]:
    print(f"\n======== loading {model_name} ({len(conditions)} conditions) ========", flush=True)
    model_cfg = eval_cfg.model
    model_cfg.name = model_name
    tokenizer = load_tokenizer(model_cfg)
    model = load_model(model_cfg, device)
    model.eval()

    out: dict[str, dict[str, Any]] = {}
    for condition in conditions:
        cid = condition["id"]
        print(f"\n--- condition {cid} ---", flush=True)
        cfg = load_config(DEFAULT_EVAL_CONFIG)
        cfg.model.name = model_name
        _apply_condition(cfg, condition, tool_format)
        summary = run_eval_with_model(cfg, model, tokenizer, device, rows=rows)
        summary["condition_id"] = cid
        summary["sdft"] = condition["sdft"]
        out[cid] = summary
        out_path = ABLATION_DIR / f"{cid}.json"
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {out_path}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_EVAL_CONFIG))
    parser.add_argument("--format", default="openclaw", choices=["auto", "openclaw", "lfm"])
    parser.add_argument("--merged", default=str(DEFAULT_MERGED), help="post-SDFT checkpoint dir")
    parser.add_argument("--skip-train", action="store_true", help="reuse existing merged checkpoint")
    parser.add_argument("--skip-data", action="store_true", help="skip data rebuild (implies --skip-train)")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="override toolcall.max_new_tokens for all conditions",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=None,
        help="override toolcall.max_rounds for all conditions",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=None,
        help="override toolcall.max_context_chars for all conditions",
    )
    args = parser.parse_args()

    if args.skip_data:
        args.skip_train = True

    ABLATION_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = Path(args.merged)

    if not args.skip_data:
        _prepare_data_and_train(skip_train=args.skip_train)
    elif not args.skip_train:
        _prepare_data_and_train(skip_train=False)

    if not merged_path.exists() and any(c["sdft"] for c in CONDITIONS):
        print(f"warning: merged checkpoint missing at {merged_path}; SDFT conditions may fail")

    eval_cfg = load_config(args.config)
    if args.max_new_tokens is not None:
        eval_cfg.toolcall.max_new_tokens = args.max_new_tokens
    if args.max_rounds is not None:
        eval_cfg.toolcall.max_rounds = args.max_rounds
    if args.max_context_chars is not None:
        eval_cfg.toolcall.max_context_chars = args.max_context_chars
    # Fail closed if held-out eval leaks into SDFT train prompts.
    eval_file = Path(eval_cfg.openclaw_eval.data_file or "")
    if not eval_file.is_absolute():
        eval_file = ROOT / eval_file
    train_file = DEFAULT_TRAIN_JSONL
    eval_qs = load_questions_from_jsonl(eval_file, field_candidates=("question", "input", "prompt"))
    train_qs = load_questions_from_jsonl(train_file, field_candidates=("input", "question"))
    assert_no_question_overlap(
        eval_questions=eval_qs,
        train_questions=train_qs,
        forbidden=FORBIDDEN_FEW_SHOT,
        label=str(eval_file),
    )
    print(f"overlap guard ok: {len(eval_qs)} eval vs {len(train_qs)} train questions")

    device = pick_device()
    print(f"device: {device}")
    rows = _load_eval_rows(eval_cfg)
    print(f"shared eval bank: {len(rows)} examples from {eval_cfg.openclaw_eval.data_file}")

    base_name = eval_cfg.model.name
    base_conds = [c for c in CONDITIONS if not c["sdft"]]
    sdft_conds = [c for c in CONDITIONS if c["sdft"]]

    all_results: dict[str, dict[str, Any]] = {}
    all_results.update(
        _run_condition_group(
            model_name=base_name,
            conditions=base_conds,
            eval_cfg=eval_cfg,
            rows=rows,
            device=device,
            tool_format=args.format,
        )
    )
    if sdft_conds:
        all_results.update(
            _run_condition_group(
                model_name=str(merged_path),
                conditions=sdft_conds,
                eval_cfg=eval_cfg,
                rows=rows,
                device=device,
                tool_format=args.format,
            )
        )

    table_rows = []
    for condition in CONDITIONS:
        cid = condition["id"]
        s = all_results[cid]
        table_rows.append(
            {
                "id": cid,
                "pass_at_1": s["pass_at_k"],
                "mean_tool_calls": s["mean_tool_calls"],
                "mean_score": s["mean_score"],
                "few_shot_k": condition["few_shot_k"],
                "cot_line": condition["cot_line"],
                "model": s["model"],
            }
        )

    matrix = _correctness_matrix(all_results)
    demo_wins = _find_demo_wins(matrix)
    near_misses = _find_near_misses(matrix)

    comparison = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "format": args.format,
        "base_model": base_name,
        "merged_model": str(merged_path),
        "num_examples": len(matrix),
        "table": table_rows,
        "matrix": matrix,
        "demo_only_sdft_count": len(demo_wins),
        "demo_only_sdft_indices": [w["index"] for w in demo_wins],
    }
    comparison_path = ABLATION_DIR / "comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False) + "\n")
    print(f"\nwrote {comparison_path}")

    demo_payload: dict[str, Any] = {
        "found": bool(demo_wins),
        "count": len(demo_wins),
        "examples": [],
        "near_misses": near_misses,
    }
    for win in demo_wins[:3]:
        idx = win["index"]
        example: dict[str, Any] = {
            "index": idx,
            "prompt": win["prompt"],
            "gold": win["label"],
            "conditions": {},
        }
        for cid in all_results:
            r = all_results[cid]["results"][idx]
            sample = r["samples"][0]
            example["conditions"][cid] = {
                "pass": r["pass_at_k"],
                "pred": sample["pred"],
                "tool_call_count": sample["tool_call_count"],
                "response_snippet": (sample.get("response_text") or "")[:800],
            }
        demo_payload["examples"].append(example)

    demo_path = ABLATION_DIR / "demo_only_sdft.json"
    demo_path.write_text(json.dumps(demo_payload, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {demo_path}")

    print("\n=== Ablation table (pass@1, mean_tool_calls) ===")
    for row in table_rows:
        print(
            f"  {row['id']:12s}  pass@1={row['pass_at_1']:.3f}  "
            f"mean_tools={row['mean_tool_calls']:.2f}"
        )
    if demo_wins:
        print(f"\nFound {len(demo_wins)} demo-only-SDFT win(s); see {demo_path}")
    else:
        print(f"\nNo unique SDFT-only wins; see near_misses in {demo_path}")


if __name__ == "__main__":
    main()
