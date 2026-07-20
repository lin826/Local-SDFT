"""Build BFCL SFT / SDFT / GRPO training rows with a held-out eval split.

Split convention (no train/eval leakage):
  For each category, rows ``[0:num_eval]`` are held out for BFCL eval
  (matches ``load_bfcl_category(..., num_examples=num_eval)`` first-N).
  Training uses ``[num_eval : num_eval + num_train]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sdft.bfcl.data import (
    SUPPORTED_CATEGORIES,
    extract_user_text,
    functions_to_tools,
    load_bfcl_category,
)
from sdft.toolcall.format import LFM_TOOL_CALL_END, LFM_TOOL_CALL_START
from sdft.toolcall.split_guard import normalize_question

# Categories used for both train and the local BFCL eval subset.
TRAIN_CATEGORIES = ("simple", "multiple", "parallel", "irrelevance")

IRRELEVANCE_GOLD = (
    "None of the available tools are relevant to this request, so I will not call a function."
)

# Marker so GRPO loader keeps pre-rendered chat prefixes as plain strings.
RENDERED_PROMPT_MARKER = "<|im_start|>"


def pick_arg_value(possibles: Any) -> Any | None:
    """Pick the first concrete BFCL possible-answer value (skip optional ``\"\"``)."""
    if not isinstance(possibles, list):
        return possibles
    for value in possibles:
        if value != "":
            return value
    return None  # optional-only — omit the argument


def ground_truth_to_model_calls(ground_truth: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Convert BFCL possible_answer entries into concrete ``{name: args}`` calls."""
    if not ground_truth:
        return []
    calls: list[dict[str, Any]] = []
    for entry in ground_truth:
        if not isinstance(entry, dict) or len(entry) != 1:
            continue
        name = next(iter(entry))
        raw_args = entry[name] or {}
        args: dict[str, Any] = {}
        if isinstance(raw_args, dict):
            for key, possibles in raw_args.items():
                value = pick_arg_value(possibles)
                if value is not None:
                    args[key] = value
        calls.append({name: args})
    return calls


def model_calls_to_lfm_tool_text(calls: list[dict[str, Any]]) -> str:
    """Serialize BFCL-shaped calls as an LFM JSON tool-call block."""
    if not calls:
        return ""
    payload = []
    for call in calls:
        name = next(iter(call))
        payload.append({"name": name, "arguments": call[name] or {}})
    return f"{LFM_TOOL_CALL_START}{json.dumps(payload, ensure_ascii=False)}{LFM_TOOL_CALL_END}"


def gold_response_for_row(row: dict[str, Any]) -> str:
    """Gold completion string for SFT / GRPO (tool call or irrelevance text)."""
    if row["category"] == "irrelevance":
        return IRRELEVANCE_GOLD
    calls = ground_truth_to_model_calls(row.get("ground_truth"))
    return model_calls_to_lfm_tool_text(calls)


def render_tool_prompt(tokenizer, *, user_text: str, functions: list[dict[str, Any]]) -> str:
    """Pre-render chat prefix with tools (matches BFCL eval generation)."""
    tools = functions_to_tools(functions)
    messages = [{"role": "user", "content": user_text}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )
    except TypeError:
        sys = "List of tools: " + json.dumps(tools, ensure_ascii=False)
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": sys}, *messages],
            tokenize=False,
            add_generation_prompt=True,
        )


def split_category_rows(
    rows: list[dict[str, Any]],
    *,
    num_eval: int,
    num_train: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Hold out first ``num_eval`` rows; take the next ``num_train`` for training."""
    if num_eval < 0 or num_train < 0:
        raise ValueError("num_eval and num_train must be non-negative")
    eval_rows = rows[:num_eval]
    train_rows = rows[num_eval : num_eval + num_train]
    return train_rows, eval_rows


def load_bfcl_train_eval_split(
    *,
    categories: tuple[str, ...] | list[str] = TRAIN_CATEGORIES,
    num_train_per_cat: int = 16,
    num_eval_per_cat: int = 32,
    cache_dir: str | Path = "data/bfcl",
) -> dict[str, Any]:
    """Load per-category train/eval splits and assert no question overlap."""
    unknown = [c for c in categories if c not in SUPPORTED_CATEGORIES]
    if unknown:
        raise ValueError(f"Unsupported categories: {unknown}")

    train_all: list[dict[str, Any]] = []
    eval_all: list[dict[str, Any]] = []
    per_category: dict[str, Any] = {}

    for cat in categories:
        # Need enough rows for eval holdout + train.
        needed = num_eval_per_cat + num_train_per_cat
        rows = load_bfcl_category(cat, cache_dir=cache_dir, num_examples=needed)
        if len(rows) < needed:
            raise ValueError(
                f"Category {cat!r} has only {len(rows)} rows; "
                f"need {needed} ({num_eval_per_cat} eval + {num_train_per_cat} train)"
            )
        train_rows, eval_rows = split_category_rows(
            rows, num_eval=num_eval_per_cat, num_train=num_train_per_cat
        )
        train_all.extend(train_rows)
        eval_all.extend(eval_rows)
        per_category[cat] = {
            "train_ids": [r["id"] for r in train_rows],
            "eval_ids": [r["id"] for r in eval_rows],
            "train_n": len(train_rows),
            "eval_n": len(eval_rows),
        }

    train_qs = [extract_user_text(r["question"]) for r in train_all]
    eval_qs = [extract_user_text(r["question"]) for r in eval_all]
    # BFCL reuses similar wording across categories; guard by id + train∩eval
    # question text, but allow duplicate questions within the eval slice.
    train_norms = {normalize_question(q) for q in train_qs}
    overlap_qs = [q for q in eval_qs if normalize_question(q) in train_norms]
    if overlap_qs:
        raise ValueError(
            f"BFCL train/eval question overlap ({len(overlap_qs)}): {overlap_qs[:3]!r}"
        )

    train_ids = {r["id"] for r in train_all}
    eval_ids = {r["id"] for r in eval_all}
    overlap_ids = train_ids & eval_ids
    if overlap_ids:
        raise ValueError(f"BFCL train/eval id overlap: {sorted(overlap_ids)[:5]}")

    return {
        "train": train_all,
        "eval": eval_all,
        "per_category": per_category,
        "train_ids": sorted(train_ids),
        "eval_ids": sorted(eval_ids),
        "num_train_per_cat": num_train_per_cat,
        "num_eval_per_cat": num_eval_per_cat,
        "categories": list(categories),
    }


def build_sft_row(row: dict[str, Any], tokenizer) -> dict[str, Any] | None:
    """Build one gold SFT jsonl row with a pre-rendered tool prompt."""
    user_text = extract_user_text(row["question"])
    response = gold_response_for_row(row)
    if not response.strip():
        return None
    prompt = render_tool_prompt(tokenizer, user_text=user_text, functions=row["function"])
    return {
        "id": row["id"],
        "category": row["category"],
        "prompt": prompt,
        "response": response,
        "sdft_response": response,
        "user_text": user_text,
    }


def build_grpo_row(row: dict[str, Any], tokenizer) -> dict[str, Any] | None:
    """Build one GRPO jsonl row with BFCL scoring metadata."""
    sft = build_sft_row(row, tokenizer)
    if sft is None:
        return None
    gt = row.get("ground_truth")
    return {
        "id": sft["id"],
        "category": sft["category"],
        "prompt": sft["prompt"],
        "response": sft["response"],
        "bfcl_category": sft["category"],
        "bfcl_functions": json.dumps(row["function"], ensure_ascii=False),
        "bfcl_ground_truth": json.dumps(gt, ensure_ascii=False) if gt is not None else "",
    }


def write_jsonl(path: Path | str, rows: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def write_split_manifest(path: Path | str, split: dict[str, Any], *, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "categories": split["categories"],
        "num_train_per_cat": split["num_train_per_cat"],
        "num_eval_per_cat": split["num_eval_per_cat"],
        "train_n": len(split["train"]),
        "eval_n": len(split["eval"]),
        "train_ids": split["train_ids"],
        "eval_ids": split["eval_ids"],
        "per_category": split["per_category"],
        "overlap": 0,
        "notes": (
            "Eval = first num_eval_per_cat rows per category (same slice as "
            "default BFCL eval). Train = the following num_train_per_cat rows. "
            "Overlap guarded by normalized question text and example id."
        ),
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def is_rendered_prompt(prompt: str) -> bool:
    """True when ``prompt`` is already a chat-templated string (do not re-wrap)."""
    return RENDERED_PROMPT_MARKER in prompt
