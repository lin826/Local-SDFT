"""Demo support: held-out prompts + success@held-out for the "Airplane-Mode Coach".

The demo teaches a behavior (a reward_fn task, e.g. house_style) from live
feedback and shows a success-rate curve that climbs. Crucially, success is
measured on HELD-OUT prompts the model was never coached on — that's the beat
that proves it learned a skill, not memorized answers (the RAG-killer).
"""

from __future__ import annotations

# Prompts used to COACH (train) — the model gets feedback on these.
COACH_PROMPTS = [
    "How do I center a div in CSS?",
    "What's a good way to learn a new language?",
    "Explain what a hash map is.",
    "How should I structure a REST API?",
    "What are the benefits of unit testing?",
    "How do I speed up a slow SQL query?",
    "What is dependency injection?",
    "How do I handle errors in async code?",
]

# HELD-OUT prompts — never coached on; used only to measure generalization.
HELDOUT_PROMPTS = [
    "What is the difference between TCP and UDP?",
    "How do I debug a memory leak?",
    "Explain the concept of recursion.",
    "What makes a good code review?",
    "How do I choose a database index?",
    "What is a race condition?",
]


def success_on(backend, reward_fn, prompts: list[str], threshold: float = 0.99) -> dict:
    """Greedy-answer each prompt, score with reward_fn, return success rate.

    A prompt "succeeds" when its reward meets `threshold` (default: full marks).
    """
    scores = []
    detail = []
    for p in prompts:
        reply = backend.generate([{"role": "user", "content": p}],
                                 temperature=0.0, max_new_tokens=96)
        r = reward_fn(p, reply)
        scores.append(r)
        detail.append({"prompt": p, "reply": reply, "reward": r})
    n = len(prompts)
    success = sum(1 for s in scores if s >= threshold) / n if n else 0.0
    mean_reward = sum(scores) / n if n else 0.0
    return {"success": success, "mean_reward": round(mean_reward, 4),
            "n": n, "detail": detail}
