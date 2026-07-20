"""Reward functions for local GRPO baselines."""

from __future__ import annotations

import re
from typing import Any

_REFUSAL_RE = re.compile(
    r"(?i)\b("
    r"i('m| am) sorry[,.]?\s*(but\s+)?i (can('t|not)|am unable to)|"
    r"i can('t|not) (assist|help|provide)|"
    r"as an ai( language model)?"
    r")\b"
)


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", "")).strip()
        return str(last).strip()
    if isinstance(completion, dict):
        return str(completion.get("content", "")).strip()
    return str(completion).strip()


def _token_jaccard(a: str, b: str) -> float:
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def instruction_reward(
    completions: list[Any],
    gold: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """Heuristic reward for open-ended instruction following.

    Encourages non-empty, non-refusal answers of moderate length, and (when
    ``gold`` is present on the dataset) lexical overlap with the reference.
    """
    del kwargs  # TRL passes extra columns / trainer state
    rewards: list[float] = []
    golds = list(gold or [])
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        score = 0.0
        if not text:
            rewards.append(-1.0)
            continue
        if _REFUSAL_RE.search(text):
            score -= 1.0
        else:
            score += 0.5
        n = len(text)
        if 40 <= n <= 1200:
            score += 0.5
        elif n < 20:
            score -= 0.5
        elif n > 2000:
            score -= 0.25
        if i < len(golds) and golds[i]:
            score += 1.5 * _token_jaccard(text, golds[i])
        rewards.append(float(score))
    return rewards


def boxed_reward(
    completions: list[Any],
    ground_truth: list[str] | None = None,
    **kwargs: Any,
) -> list[float]:
    """+1/−1 reward using OpenClaw-style ``\\boxed{}`` verification when available."""
    del kwargs
    from sdft.toolcall.scoring import score_openclaw_solution

    truths = list(ground_truth or [])
    rewards: list[float] = []
    for i, completion in enumerate(completions):
        text = _completion_text(completion)
        truth = truths[i] if i < len(truths) else ""
        if not truth:
            rewards.append(0.0)
            continue
        result = score_openclaw_solution(text, truth, strict_box_verify=True)
        rewards.append(float(result["score"]))
    return rewards


def resolve_reward(name: str):
    if name == "instruction":
        return instruction_reward
    if name == "boxed":
        return boxed_reward
    raise ValueError(f"unknown GRPO reward {name!r}; use instruction|boxed")
