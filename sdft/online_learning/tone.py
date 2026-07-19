"""Lightweight tone classifier for implicit feedback on the previous assistant reply."""

from __future__ import annotations

import re

POSITIVE_TERMS = (
    "thanks",
    "thank you",
    "thx",
    "great",
    "perfect",
    "good job",
    "nice",
    "love it",
    "awesome",
    "helpful",
    "correct",
    "exactly",
    "well done",
    "nice one",
    "spot on",
    "works",
    "yes!",
    "👍",
    "🙂",
    "😊",
)

NEGATIVE_TERMS = (
    "wrong",
    "incorrect",
    "bad",
    "terrible",
    "useless",
    "not what",
    "doesn't work",
    "don't like",
    "do not like",
    "no good",
    "awful",
    "hate",
    "fail",
    "failed",
    "nonsense",
    "garbage",
    "stop",
    "try again",
    "not helpful",
    "👎",
    "😠",
    "😡",
)

VALID_TONES = ("positive", "neutral", "negative")
TONE_TO_REWARD = {"positive": 1, "neutral": 0, "negative": -1}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def classify_tone(message: str) -> tuple[str, int]:
    """Classify user message tone as implicit feedback on the prior assistant reply."""
    text = _normalize(message)
    if not text:
        return "neutral", 0

    pos_hits = sum(1 for term in POSITIVE_TERMS if term in text)
    neg_hits = sum(1 for term in NEGATIVE_TERMS if term in text)

    if neg_hits > pos_hits and neg_hits > 0:
        return "negative", -1
    if pos_hits > neg_hits and pos_hits > 0:
        return "positive", 1
    if text in {"yes", "y", "ok", "okay", "sure", "yep", "yeah"}:
        return "positive", 1
    if text in {"no", "n", "nope", "nah"}:
        return "negative", -1
    return "neutral", 0


def resolve_tone(
    message: str,
    *,
    override: str | None = None,
) -> tuple[str, int, str]:
    """Return (tone, reward, source) honoring an optional manual override."""
    if override and override in VALID_TONES:
        return override, TONE_TO_REWARD[override], "manual"
    tone, reward = classify_tone(message)
    return tone, reward, "auto"
