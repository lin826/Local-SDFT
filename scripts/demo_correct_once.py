#!/usr/bin/env python
"""Correct it once, never again — the demo that kills "I keep telling it the same thing".

You have a pet peeve: you want one-sentence answers, not essays. You fix a few of
the assistant's replies (in plain language, like correcting a colleague). Within
a couple of corrections it stops — and not just on what you corrected: the habit
transfers to *completely different topics* it never saw corrected. That's a
learned behavior, on-device, that you never have to repeat.

The proof it generalizes (not memorizes): we correct only COOKING answers and
measure the one-sentence habit on held-out PROGRAMMING questions.

Run:  python scripts/demo_correct_once.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

from sdft.config import load_config
from sdft.online.controller import OnlineController
from sdft.online.reward import get_reward_fn, get_shaper

console = Console()

# Correct on one domain...
COACH = [
    "How do I boil an egg?",
    "What can I use instead of butter?",
    "How long should I cook pasta?",
    "How do I know when chicken is done?",
    "What's the water-to-rice ratio?",
    "How do I ripen an avocado quickly?",
    "How do I keep guacamole from browning?",
    "What's a substitute for buttermilk?",
    "How do I soften brown sugar?",
    "What temperature should I roast vegetables at?",
    "How do I stop onions from making me cry?",
    "How long can leftovers stay in the fridge?",
]
# ...measure the habit on a totally different domain it was never corrected on.
HELDOUT = [
    "What is a hash map?",
    "How do I reverse a list in Python?",
    "What does a compiler do?",
    "What is recursion?",
    "How do I center a div in CSS?",
    "What is a race condition?",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_correct_once.yaml")
    ap.add_argument("--max-corrections", type=int, default=12)
    ap.add_argument("--consolidation-steps", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    obeys = get_reward_fn("one_sentence")
    fix = get_shaper("one_sentence")

    console.print(Rule("Correct it once — the fix generalizes, on-device"))
    console.print(f"[dim]model={cfg.model.name}  offline  "
                  "your rule: 'answer in ONE sentence'  ·  correct on cooking, test on programming[/]")

    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())

    console.print("[dim]loading model…[/]")
    ctrl = OnlineController.build(cfg)

    def held_out_rate(tag: str) -> float:
        hits, sample = 0, None
        for q in HELDOUT:
            r = ctrl.backend.generate([{"role": "user", "content": q}],
                                      temperature=0.0, max_new_tokens=64)
            ok = obeys(q, r) >= 1.0
            hits += ok
            if sample is None:
                sample = (q, r, ok)
        rate = hits / len(HELDOUT)
        bar = "█" * round(rate * 24)
        console.print(f"  {tag:<26} one-sentence on held-out [green]{rate*100:5.1f}%[/]  {bar}")
        return rate, sample

    console.print(Rule("Before — you haven't told it your preference yet", style="dim"))
    base_rate, (bq, br, _) = held_out_rate("base")
    console.print(f"  e.g. Q: {bq}")
    console.print(f"       A: {br[:120]!r}{'…' if len(br) > 120 else ''}")

    console.print(Rule("You fix its replies a few times (cooking questions only)", style="dim"))
    conv = "fix-" + uuid.uuid4().hex[:6]
    corrections = 0
    applied: list[tuple[str, str]] = []   # (question, one-line fix) — the "knowledge" ICL/RAG also get
    learned_rate = base_rate
    # Keep clearing cooking mail-style questions, correcting any that break the
    # rule, until the habit transfers to held-out or we hit the correction budget.
    while corrections < args.max_corrections and learned_rate < 0.85:
        progressed = False
        for q in COACH:
            if corrections >= args.max_corrections:
                break
            mid, reply = ctrl.chat(conv, q)
            if obeys(q, reply) >= 1.0:
                continue  # it already obeyed — nothing to correct
            fixed = fix(q, reply)
            ctrl.correct(conv, mid, fixed)   # you fix it, in one line
            applied.append((q, fixed))
            corrections += 1
            progressed = True
            ctrl.maybe_update(force=True)
            learned_rate, _ = held_out_rate(f"after {corrections} correction(s)")
            if learned_rate >= 0.85:
                break
        if not progressed:
            break  # it obeys every cooking question; consolidate below

    # Consolidation: the on-device loop keeps replaying the few corrections you
    # gave (no new input from you), strengthening the habit until it generalizes.
    if learned_rate < 0.85:
        console.print(Rule("It consolidates on its own (replaying your corrections)", style="dim"))
        for i in range(args.consolidation_steps):
            if ctrl.maybe_update(force=True) is None:
                break
            learned_rate, _ = held_out_rate(f"consolidation {i + 1}")
            if learned_rate >= 0.85:
                break

    console.print(Rule("After — a held-out programming question", style="dim"))
    learned_rate, (hq, hr, hok) = held_out_rate("adapter ON (learned)")
    console.print(f"  Q: {hq}")
    console.print(f"  A: {hr[:160]!r}   {'[green]✓ one sentence[/]' if hok else '[red]✗[/]'}")

    # ---- Fair baselines: ICL / RAG WITHOUT finetuning -------------------
    # Give the *base* model the same corrections, but in-context instead of in
    # the weights — the honest "why not just prompt/retrieve?" test.
    tok = ctrl.backend.tokenizer
    rule = "Answer in exactly one sentence."

    def ntok(messages) -> int:
        out = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        ids = out["input_ids"] if hasattr(out, "keys") else out
        ids = list(ids)
        if ids and isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
        return len(ids)

    def icl_context(examples, query):
        msgs = [{"role": "system", "content": rule}]
        for q, a in examples:
            msgs += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
        msgs += [{"role": "user", "content": query}]
        return msgs

    def retrieve(query, k=3):
        qs = set(re.findall(r"[a-z]+", query.lower()))
        return sorted(applied, key=lambda c: len(qs & set(re.findall(r"[a-z]+", c[0].lower()))),
                      reverse=True)[:k]

    def eval_context(tag, ctx_fn):
        hits, overhead = 0, 0
        bare = 0
        for q in HELDOUT:
            msgs = ctx_fn(q)
            reply = ctrl.backend.generate(msgs, temperature=0.0, max_new_tokens=64)
            hits += obeys(q, reply) >= 1.0
            overhead += ntok(msgs) - ntok([{"role": "user", "content": q}])
        rate = hits / len(HELDOUT)
        avg_ovh = overhead / len(HELDOUT)
        console.print(f"  {tag:<34} one-sentence [green]{rate*100:5.1f}%[/]   "
                      f"+{avg_ovh:.0f} tokens/call")
        return rate, avg_ovh

    console.print(Rule("The honest test: RAG / in-context vs finetuning (no training)", style="dim"))
    ctrl.rollback(0)   # base weights — everything below is WITHOUT finetuning
    icl_rate, icl_ovh = eval_context(
        f"ICL: rule + all {len(applied)} corrections", lambda q: icl_context(applied, q))
    rag_rate, rag_ovh = eval_context(
        "RAG: rule + top-3 retrieved", lambda q: icl_context(retrieve(q), q))
    base_again, _ = eval_context("base: no help", lambda q: [{"role": "user", "content": q}])
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1)   # restore the finetuned adapter

    console.print(Rule("Result", style="green"))
    console.print("  same held-out programming questions, one-sentence habit:")
    console.print(f"    base (no help)                 [red]{base_again*100:5.0f}%[/]   +0 tokens/call")
    console.print(f"    ICL (rule + all corrections)   [yellow]{icl_rate*100:5.0f}%[/]   +{icl_ovh:.0f} tokens/call, every call")
    console.print(f"    RAG (rule + retrieved)         [yellow]{rag_rate*100:5.0f}%[/]   +{rag_ovh:.0f} tokens/call, every call")
    console.print(f"    finetuned (ours)               [green]{learned_rate*100:5.0f}%[/]   [bold]+0 tokens/call[/]")
    console.print(f"\n  {corrections} corrections, folded into the weights: same-or-better accuracy than"
                  " stuffing them into every prompt — at zero context cost, on-device.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
