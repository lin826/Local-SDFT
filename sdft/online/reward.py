"""Reward functions for reward-selected on-policy self-distillation (RAFT-style).

A reward function scores a candidate reply in [0, 1]. When `online.reward_fn`
names one of these, the controller samples several rollouts per prompt, keeps
the reward-passing ones as demonstrations, and distills the model onto them —
genuinely online RL (reward-driven selection of the model's own samples),
reusing the SDFT loss. This powers the hands-free "success curve" demo.

Register a new task by decorating with @reward("name").
"""

from __future__ import annotations

import re
from typing import Callable

RewardFn = Callable[[str, str], float]  # (prompt, reply) -> reward in [0, 1]

_REGISTRY: dict[str, RewardFn] = {}


def reward(name: str) -> Callable[[RewardFn], RewardFn]:
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def get_reward_fn(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward_fn {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# ---- flagship demo task: "house style" ------------------------------------
# TL;DR line, then <=3 bullets, then a clarifying question. The base 230M does
# not do this by default; it generalizes to unseen prompts; it is objectively
# checkable, so the success curve climbs hands-free.

_BULLET = re.compile(r"^\s*[-*]\s+\S", re.MULTILINE)


@reward("house_style")
def house_style(prompt: str, reply: str) -> float:
    lines = [ln for ln in reply.strip().splitlines() if ln.strip()]
    if not lines:
        return 0.0
    score = 0.0
    # 1) opens with a TL;DR
    if re.match(r"\s*(tl;?dr)\b", lines[0], re.IGNORECASE):
        score += 0.34
    # 2) has 1..3 bullets
    n_bullets = len(_BULLET.findall(reply))
    if 1 <= n_bullets <= 3:
        score += 0.33
    # 3) ends with a question
    if reply.rstrip().endswith("?"):
        score += 0.33
    return round(score, 4)


@reward("five_words")
def five_words(prompt: str, reply: str) -> float:
    """Answer in exactly five words (a crisp, unmistakable constraint)."""
    words = re.findall(r"\b[\w']+\b", reply.strip())
    return 1.0 if len(words) == 5 else 0.0


@reward("terse")
def terse(prompt: str, reply: str) -> float:
    """Reward short, punchy replies (<= 25 words, single paragraph)."""
    words = re.findall(r"\b[\w']+\b", reply.strip())
    n_para = len([b for b in reply.strip().split("\n\n") if b.strip()])
    ok_len = 1 <= len(words) <= 25
    ok_para = n_para <= 1
    return float(ok_len and ok_para)
