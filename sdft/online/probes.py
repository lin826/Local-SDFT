"""Probe-based evaluation: general-capability guardrails + personalization score.

General probes catch catastrophic forgetting (the model should keep answering
simple questions correctly after many online updates). Personalization score
measures whether replies to previously-taught questions now overlap with the
demonstrations — i.e., whether online learning is sticking.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .events import Demonstration

GENERAL_PROBES: list[tuple[str, str]] = [
    ("What is the capital of France?", "paris"),
    ("What is 2 + 2?", "4"),
    ("Which planet is known as the Red Planet?", "mars"),
    ("How many days are there in a week?", "7"),
    ("What is the largest ocean on Earth?", "pacific"),
    ("What is the chemical symbol for water?", "h2o"),
    ("How many sides does a triangle have?", "3"),
    ("What language is primarily spoken in Brazil?", "portuguese"),
]

_PROBE_MAX_TOKENS = 32


def run_general_probes(backend, probes: list[tuple[str, str]] = GENERAL_PROBES) -> dict:
    """Greedy-answer each probe; return accuracy + per-probe detail."""
    hits = 0
    detail = []
    for question, keyword in probes:
        answer = backend.generate(
            [{"role": "user", "content": question}],
            temperature=0.0,
            max_new_tokens=_PROBE_MAX_TOKENS,
        )
        hit = keyword.lower() in answer.lower()
        hits += hit
        detail.append({"question": question, "keyword": keyword, "answer": answer, "hit": hit})
    return {"accuracy": hits / len(probes), "hits": hits, "n": len(probes), "detail": detail}


def personalization_score(backend, demos: list[Demonstration], max_demos: int = 10) -> dict:
    """Word-overlap between greedy replies and the demonstrations they taught."""
    if not demos:
        return {"overlap": 0.0, "n": 0, "detail": []}
    total = 0.0
    detail = []
    for demo in demos[-max_demos:]:
        question = next(
            (m["content"] for m in reversed(demo.messages) if m["role"] == "user"), None
        )
        if question is None:
            continue
        answer = backend.generate(
            [{"role": "user", "content": question}],
            temperature=0.0,
            max_new_tokens=_PROBE_MAX_TOKENS * 2,
        )
        score = _word_overlap(answer, demo.demonstration)
        total += score
        detail.append({"question": question, "answer": answer, "overlap": score})
    n = len(detail)
    return {"overlap": total / n if n else 0.0, "n": n, "detail": detail}


def _word_overlap(a: str, b: str) -> float:
    """F1 over content-word sets (stopwords removed)."""
    wa = _content_words(a)
    wb = _content_words(b)
    if not wa or not wb:
        return 0.0
    inter = len(wa & wb)
    precision = inter / len(wa)
    recall = inter / len(wb)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "it", "its", "of", "to", "in",
    "and", "or", "for", "on", "at", "by", "with", "that", "this", "you", "your",
    "we", "our", "i", "my", "do", "does", "what", "when", "where", "which", "how",
}


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1}


@dataclass
class ProbeEvaluator:
    """Controller eval_hook: flags degradation vs. a captured baseline."""

    threshold_drop: float = 0.2
    baseline: float | None = field(default=None)

    def capture_baseline(self, backend) -> float:
        self.baseline = run_general_probes(backend)["accuracy"]
        return self.baseline

    def __call__(self, controller) -> dict:
        result = run_general_probes(controller.backend)
        baseline = self.baseline if self.baseline is not None else result["accuracy"]
        return {
            "probe_accuracy": result["accuracy"],
            "probe_baseline": baseline,
            "degraded": result["accuracy"] < baseline - self.threshold_drop,
        }
