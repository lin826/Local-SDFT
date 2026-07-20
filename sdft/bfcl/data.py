"""Load BFCL-v3 single-turn categories from the public Hugging Face dataset."""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Any

HF_DATASET = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
HF_BASE = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main"

# Local subset categories (AST + irrelevance). Live/multi-turn/exec skipped.
SUPPORTED_CATEGORIES = (
    "simple",
    "multiple",
    "parallel",
    "parallel_multiple",
    "irrelevance",
)

# HF filenames: simple -> BFCL_v3_simple.json (historically also simple_python).
_CATEGORY_FILES = {
    "simple": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as resp:
        dest.write_bytes(resp.read())


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_category_files(
    category: str,
    cache_dir: str | Path,
    *,
    force: bool = False,
) -> tuple[Path, Path | None]:
    """Download question (+ possible_answer when applicable) files into cache_dir."""
    if category not in _CATEGORY_FILES:
        raise ValueError(
            f"Unsupported BFCL category {category!r}; "
            f"supported: {sorted(SUPPORTED_CATEGORIES)}"
        )
    cache = Path(cache_dir)
    fname = _CATEGORY_FILES[category]
    questions = cache / fname
    if force or not questions.is_file():
        _download(f"{HF_BASE}/{fname}", questions)

    answers: Path | None = None
    if category != "irrelevance":
        answers = cache / "possible_answer" / fname
        if force or not answers.is_file():
            _download(f"{HF_BASE}/possible_answer/{fname}", answers)
    return questions, answers


def load_bfcl_category(
    category: str,
    *,
    cache_dir: str | Path = "data/bfcl",
    num_examples: int | None = None,
    force_download: bool = False,
) -> list[dict[str, Any]]:
    """Return joined question + ground-truth rows for one category."""
    q_path, a_path = ensure_category_files(category, cache_dir, force=force_download)
    questions = _read_jsonl(q_path)
    answers_by_id: dict[str, Any] = {}
    if a_path is not None:
        for row in _read_jsonl(a_path):
            answers_by_id[str(row["id"])] = row.get("ground_truth")

    rows: list[dict[str, Any]] = []
    for q in questions:
        qid = str(q["id"])
        rows.append(
            {
                "id": qid,
                "category": category,
                "question": q["question"],
                "function": q["function"],
                "ground_truth": answers_by_id.get(qid),
            }
        )
    if num_examples is not None:
        rows = rows[:num_examples]
    return rows


def extract_user_text(question: Any) -> str:
    """Flatten BFCL nested message lists into a single user string."""
    # question is typically [[{role, content}, ...]] (one turn) or multi-turn.
    if isinstance(question, str):
        return question.strip()
    if not isinstance(question, list) or not question:
        return str(question)

    turns = question
    # Single-turn wrapped as [[msgs]]
    if turns and isinstance(turns[0], list):
        msgs = turns[0]
    else:
        msgs = turns

    parts: list[str] = []
    for msg in msgs:
        if isinstance(msg, dict) and msg.get("role") == "user":
            parts.append(str(msg.get("content", "")).strip())
        elif isinstance(msg, str):
            parts.append(msg.strip())
    return "\n".join(p for p in parts if p)


def functions_to_tools(functions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize BFCL function docs for ``apply_chat_template(..., tools=...)``.

    BFCL uses ``parameters.type: "dict"``; chat templates expect ``"object"``.
    """
    tools: list[dict[str, Any]] = []
    for fn in functions:
        params = dict(fn.get("parameters") or {})
        if params.get("type") == "dict":
            params = {**params, "type": "object"}
        tools.append(
            {
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": params,
            }
        )
    return tools
