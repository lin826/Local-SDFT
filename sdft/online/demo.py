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

# --- tool-calling task: "use a calculator" --------------------------------
# COACH uses SMALL numbers; HELD-OUT uses LARGE numbers that never appear in
# coaching — so a correct held-out answer can only come from learning the skill
# (translate the question into a calc() call), not from memorizing answers.
COACH_CALC = [
    "What is 3 + 4?", "What is 7 * 6?", "What is 9 - 2?", "What is 12 + 8?",
    "What is 5 * 5?", "What is 15 - 7?", "What is 6 + 9?", "What is 8 * 3?",
    "What is 14 - 6?", "What is 11 + 4?", "What is 7 * 7?", "What is 18 - 9?",
]
HELDOUT_CALC = [
    "What is 347 + 288?", "What is 913 - 476?", "What is 128 * 47?",
    "What is 654 + 279?", "What is 802 - 355?", "What is 236 * 19?",
]

_TASK_PROMPTS = {
    "house_style": (COACH_PROMPTS, HELDOUT_PROMPTS),
    "five_words": (COACH_PROMPTS, HELDOUT_PROMPTS),
    "terse": (COACH_PROMPTS, HELDOUT_PROMPTS),
    "calc_tool": (COACH_CALC, HELDOUT_CALC),
}


def prompts_for(reward_fn: str | None) -> tuple[list[str], list[str]]:
    """(coach, held-out) prompt sets for a task; defaults to the style set."""
    return _TASK_PROMPTS.get(reward_fn or "", (COACH_PROMPTS, HELDOUT_PROMPTS))


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
