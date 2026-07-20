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
ShaperFn = Callable[[str, str], str]    # (prompt, reply) -> reply reshaped to pass

_REGISTRY: dict[str, RewardFn] = {}
_SHAPERS: dict[str, ShaperFn] = {}


def reward(name: str) -> Callable[[RewardFn], RewardFn]:
    def deco(fn: RewardFn) -> RewardFn:
        _REGISTRY[name] = fn
        return fn
    return deco


def shaper(name: str) -> Callable[[ShaperFn], ShaperFn]:
    """Register a deterministic reshaper that turns any reply into a passing one.

    When set, the controller trains on shape(prompt, best_sample) instead of the
    raw sample — guaranteeing a full-marks target, which is what makes a small
    model learn the behavior reliably (the model's own *content*, the task's
    *format*).
    """
    def deco(fn: ShaperFn) -> ShaperFn:
        _SHAPERS[name] = fn
        return fn
    return deco


def get_reward_fn(name: str) -> RewardFn:
    if name not in _REGISTRY:
        raise KeyError(f"unknown reward_fn {name!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def get_shaper(name: str) -> ShaperFn | None:
    return _SHAPERS.get(name)


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


_SENT = re.compile(r"[^.!?\n]+[.!?]?")


@shaper("house_style")
def shape_house_style(prompt: str, reply: str) -> str:
    """Reshape arbitrary content into TL;DR + <=3 bullets + a clarifying question.

    Uses the model's own sentences as material, so it's the model's content in
    the target format — a guaranteed full-marks demonstration.
    """
    raw = re.sub(r"^\s*(tl;?dr:?)\s*", "", reply.strip(), flags=re.IGNORECASE)
    sents = [s.strip(" -*") for s in _SENT.findall(raw) if len(s.strip()) > 3]
    if not sents:
        sents = ["Here is a concise answer."]
    tldr = sents[0].rstrip(".")
    bullets = sents[1:4] if len(sents) > 1 else [tldr]
    bullets = bullets[:3]
    lines = [f"TL;DR: {tldr}."]
    lines += [f"- {b.rstrip('.')}." for b in bullets]
    lines.append("Does that cover what you needed, or should I go deeper?")
    return "\n".join(lines)


# ---- tool-calling demo task: "use a calculator" --------------------------
# The strongest "it LEARNS, not memorizes" task: reward the model for emitting a
# calc() tool call whose expression evaluates to the correct answer. Held-out
# problems use numbers never seen in coaching, so success can't be memorized.


@reward("calc_tool")
def calc_tool(prompt: str, reply: str) -> float:
    """1.0 = valid tool call with the correct value; 0.4 = called but wrong; 0 = no call.

    Rewards tool *use*, not freehand arithmetic — a base model that guesses the
    number without calling the tool scores 0, so what's learned is the tool-call
    policy, which is exactly what generalizes.
    """
    from .tools import extract_arithmetic, parse_calc_call, safe_eval

    truth = safe_eval(extract_arithmetic(prompt))
    if truth is None:
        return 0.0
    call = parse_calc_call(reply)
    if call is None:
        return 0.0  # no tool call at all
    got = safe_eval(call)
    if got is None:
        return 0.2  # emitted something callable-ish but not evaluable
    return 1.0 if abs(got - truth) < 1e-6 else 0.4


@shaper("calc_tool")
def shape_calc_tool(prompt: str, reply: str) -> str:
    """Guaranteed-correct demonstration: the exact tool call for this problem."""
    from .tools import extract_arithmetic

    expr = extract_arithmetic(prompt) or "0"
    return f'<tool>calc("{expr}")</tool>'


@reward("five_words")
def five_words(prompt: str, reply: str) -> float:
    """Answer in exactly five words (a crisp, unmistakable constraint)."""
    words = re.findall(r"\b[\w']+\b", reply.strip())
    return 1.0 if len(words) == 5 else 0.0


_PAD = ["here", "is", "a", "short", "answer", "for", "you", "now"]


@shaper("five_words")
def shape_five_words(prompt: str, reply: str) -> str:
    """Exactly five words from the model's own content (padded if too short)."""
    words = re.findall(r"[\w']+", reply.strip()) or []
    words = (words + _PAD)[:5]
    return " ".join(words)


@reward("direct")
def direct(prompt: str, reply: str) -> float:
    """A single-line 'Answer: …' — the opposite of the multi-line briefing style."""
    s = reply.strip()
    score = 0.0
    if re.match(r"answer\s*:", s, re.IGNORECASE):
        score += 0.5
    lines = [ln for ln in s.splitlines() if ln.strip()]
    if len(lines) == 1 and not re.search(r"^\s*[-*]", s, re.MULTILINE):
        score += 0.5
    return score


@shaper("direct")
def shape_direct(prompt: str, reply: str) -> str:
    first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0] if reply.strip() else "Here is the answer."
    first = re.sub(r"^\s*answer\s*:\s*", "", first, flags=re.IGNORECASE)
    words = first.split() or ["Here", "is", "the", "answer."]
    return "Answer: " + " ".join(words[:25])


@reward("terse")
def terse(prompt: str, reply: str) -> float:
    """Reward short, punchy replies (<= 25 words, single paragraph)."""
    words = re.findall(r"\b[\w']+\b", reply.strip())
    n_para = len([b for b in reply.strip().split("\n\n") if b.strip()])
    ok_len = 1 <= len(words) <= 25
    ok_para = n_para <= 1
    return float(ok_len and ok_para)


@shaper("terse")
def shape_terse(prompt: str, reply: str) -> str:
    """First sentence, single line, capped at 25 words."""
    first = re.split(r"(?<=[.!?])\s", reply.strip(), maxsplit=1)[0] if reply.strip() else ""
    words = re.findall(r"[\w']+", first) or ["A", "brief", "answer"]
    return " ".join(words[:25])
