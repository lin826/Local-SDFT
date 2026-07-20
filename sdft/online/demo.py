"""Demo support: held-out prompts + success@held-out for the "Airplane-Mode Coach".

The demo teaches a behavior (a reward_fn task, e.g. house_style) from live
feedback and shows a success-rate curve that climbs. Crucially, success is
measured on HELD-OUT prompts the model was never coached on — that's the beat
that proves it learned a skill, not memorized answers (the RAG-killer).
"""

from __future__ import annotations

import re

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


# --- lifelong skill-accumulation demo -------------------------------------
# A curriculum of trigger-keyed skills introduced one at a time. Each skill has
# its own coach/held-out prompts (disjoint content, same lexical trigger). We
# re-evaluate ALL introduced skills every round, so we can watch a repertoire
# accumulate (with replay) or catastrophically overwrite itself (without).

COACH_SUMMARY = [
    "Summarize: The meeting covered the Q3 budget, hiring plans, and the office move to the new building downtown.",
    "Summarize: Our app crashed because a background thread wrote to the database while the main thread was reading it.",
    "Summarize: The recipe calls for marinating the chicken overnight, then grilling it over medium heat for twelve minutes.",
    "Summarize: The novel follows a young cartographer who discovers that the maps she draws quietly reshape the land.",
    "Summarize: Sales rose fifteen percent after we cut prices, expanded to two new cities, and launched the referral program.",
    "Summarize: The bridge was closed for repairs after inspectors found corrosion in three of the main support cables.",
    "Summarize: The study found that students who slept eight hours scored higher than those who crammed through the night.",
    "Summarize: The hurricane weakened to a tropical storm overnight but still brought heavy flooding to coastal towns.",
]
HELDOUT_SUMMARY = [
    "Summarize: The startup ran out of funding after its main investor pulled out weeks before the product launch.",
    "Summarize: Researchers trained a small model on phone-sized hardware and it kept improving as people used it daily.",
    "Summarize: The hikers got lost when fog rolled in, but a park ranger tracked their phone signal and led them out.",
    "Summarize: The museum's new wing doubled visitor numbers but strained a staff that had not grown in a decade.",
    "Summarize: A power outage downtown stopped the trains for an hour during the busiest part of the evening commute.",
]

COACH_LIST = [
    "List ideas for a child's birthday party.",
    "List ways to save money on groceries.",
    "List things to pack for a weekend camping trip.",
    "List reasons a website might load slowly.",
    "List habits that improve focus while working.",
    "List questions to ask in a job interview.",
    "List steps to brew a good cup of coffee.",
    "List ways to make a small room feel bigger.",
]
HELDOUT_LIST = [
    "List ideas for a team offsite.",
    "List ways to reduce plastic at home.",
    "List things to check before a long road trip.",
    "List reasons a plant might be wilting.",
    "List tips for sleeping better at night.",
]

COACH_SIGNOFF = [
    "Reply to: Can we move our meeting to Thursday afternoon?",
    "Reply to: Thanks for sending the report, it looks great.",
    "Reply to: Are you free for a quick call tomorrow morning?",
    "Reply to: The package arrived damaged, what should I do?",
    "Reply to: Congratulations on the launch, well deserved!",
    "Reply to: Could you share the slides from today's talk?",
    "Reply to: I'll be out of office next week for vacation.",
    "Reply to: Do you have time to review my draft this week?",
]
HELDOUT_SIGNOFF = [
    "Reply to: Can you confirm the venue for Friday's event?",
    "Reply to: Just checking in on the status of the invoice.",
    "Reply to: Would next Monday work for the kickoff call?",
    "Reply to: Thanks for the quick fix, it works now.",
    "Reply to: Are we still on for lunch this week?",
]

# name, reward_fn, (coach, held-out), one-line description, cold-start hint
SKILLS = [
    ("summarize", "skill_summary", (COACH_SUMMARY, HELDOUT_SUMMARY),
     "one-line summary", "Reply with a single short sentence."),
    ("calc", "calc_tool", (COACH_CALC, HELDOUT_CALC),
     "calculator tool call", 'To do math, emit <tool>calc("EXPR")</tool>.'),
    ("list", "skill_bullets", (COACH_LIST, HELDOUT_LIST),
     "3 bullet points", "Answer as three short bullet points."),
    ("signoff", "skill_signoff", (COACH_SIGNOFF, HELDOUT_SIGNOFF),
     "fixed e-mail sign-off", "End every reply with the sign-off line."),
]


def _ntokens(backend, messages: list[dict]) -> int:
    """Token length of a chat-templated prompt; falls back to a word count."""
    tok = getattr(backend, "tokenizer", None)
    if tok is None:
        return sum(len(m["content"].split()) for m in messages)
    out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    ids = out["input_ids"] if hasattr(out, "keys") else out
    ids = list(ids)
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return len(ids)


def four_way_compare(controller, k: int = 3) -> dict:
    """Fair base / ICL / RAG / finetuned comparison for the configured reward task.

    The honest "why not just prompt or retrieve?" test: the base model is given
    the SAME learned examples in-context (ICL = all, RAG = top-k retrieved) that
    the finetuned adapter absorbed into its weights, and we report both held-out
    accuracy and the per-call token tax. Finetuning wins by paying that tax once,
    at training time, instead of on every single call.
    """
    from .reward import get_reward_fn

    cfg = controller.cfg.online
    reward_fn = get_reward_fn(cfg.reward_fn)
    _, heldout = prompts_for(cfg.reward_fn)
    rule = cfg.coach_instruction or "Follow the demonstrated style."

    # Knowledge the no-train baselines get: (prompt, target) from learned demos.
    examples: list[tuple[str, str]] = []
    for d in controller.store.all_demonstrations():
        q = d.messages[-1]["content"] if d.messages else ""
        if q and d.demonstration:
            examples.append((q, d.demonstration))
    examples = examples[-8:]  # a realistic handful

    def retrieve(query: str, kk: int):
        qs = set(re.findall(r"[a-z]+", query.lower()))
        return sorted(examples,
                      key=lambda e: len(qs & set(re.findall(r"[a-z]+", e[0].lower()))),
                      reverse=True)[:kk]

    def icl_msgs(query, shots):
        msgs = [{"role": "system", "content": rule}]
        for q, a in shots:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        return msgs + [{"role": "user", "content": query}]

    def eval_variant(ctx_fn):
        hits, overhead = 0, 0
        for q in heldout:
            msgs = ctx_fn(q)
            reply = controller.backend.generate(msgs, temperature=0.0, max_new_tokens=96)
            hits += reward_fn(q, reply) >= 0.99
            overhead += _ntokens(controller.backend, msgs) - \
                _ntokens(controller.backend, [{"role": "user", "content": q}])
        n = len(heldout) or 1
        return {"success": hits / n, "extra_tokens": round(overhead / n, 1)}

    versions = controller.stats()["adapter_versions"]
    latest = versions - 1

    controller.rollback(0)  # base weights for the no-train baselines
    base = eval_variant(lambda q: [{"role": "user", "content": q}])
    icl = eval_variant(lambda q: icl_msgs(q, examples))
    rag = eval_variant(lambda q: icl_msgs(q, retrieve(q, k)))
    controller.rollback(latest)  # restore the finetuned adapter
    finetuned = eval_variant(lambda q: [{"role": "user", "content": q}])

    return {"base": base, "icl": icl, "rag": rag, "finetuned": finetuned,
            "n_examples": len(examples), "n_heldout": len(heldout)}


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
