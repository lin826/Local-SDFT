"""OpenClaw-RL compatible math scoring (strict boxed answer verification)."""

from __future__ import annotations

import re
from typing import Any


def _last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed{")
    if idx < 0:
        return None
    depth = 0
    i = idx + len("\\boxed{")
    while i < len(string):
        if string[i] == "{":
            depth += 1
        elif string[i] == "}":
            if depth == 0:
                return string[idx : i + 1]
            depth -= 1
        i += 1
    return None


def _remove_boxed(s: str) -> str:
    assert s.startswith("\\boxed{") and s.endswith("}")
    return s[len("\\boxed{") : -1]


def _normalize_answer(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\\!", "", text)
    text = re.sub(r"\\,", "", text)
    text = re.sub(r"\\text\{([^}]*)\}", r"\1", text)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def extract_boxed_answer(solution: str) -> str | None:
    """Extract the last \\boxed{...} content, preferring 'Answer:' prefix when present."""
    if "Answer:" in solution and "\\boxed{" in solution:
        tail = solution[solution.rfind("Answer:") :]
        boxed = _last_boxed_only_string(tail)
        if boxed:
            return _remove_boxed(boxed).strip()

    # Also handle $\boxed{...}$ variants common in small-model outputs.
    for pattern in (r"\\boxed\{", r"\$\s*\\boxed\{"):
        idx = -1
        search = solution
        while True:
            found = re.search(pattern, search)
            if not found:
                break
            idx = (idx + 1 if idx >= 0 else 0) + found.start()
            search = search[found.end() - 1 :]
        if idx >= 0:
            sub = solution[idx:]
            boxed = _last_boxed_only_string(sub)
            if boxed:
                return _remove_boxed(boxed).strip()

    boxed = _last_boxed_only_string(solution)
    if boxed:
        return _remove_boxed(boxed).strip()
    return None


def score_openclaw_solution(
    solution_str: str,
    ground_truth: str,
    *,
    strict_box_verify: bool = True,
) -> dict[str, Any]:
    """Return OpenClaw-RL style score dict: score in {1.0, -1.0}, acc bool, pred str."""
    solution_tail = solution_str[-300:]
    pred = extract_boxed_answer(solution_tail)
    if pred is None:
        return {"score": -1.0, "acc": False, "pred": ""}

    if strict_box_verify and "Answer:" not in solution_tail and "\\boxed{" not in solution_tail:
        return {"score": -1.0, "acc": False, "pred": pred}

    correct = _normalize_answer(pred) == _normalize_answer(str(ground_truth))
    return {"score": 1.0 if correct else -1.0, "acc": correct, "pred": pred}
