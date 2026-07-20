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
            ctrl.correct(conv, mid, fix(q, reply))   # you fix it, in one line
            corrections += 1
            progressed = True
            ctrl.maybe_update(force=True)
            learned_rate, _ = held_out_rate(f"after {corrections} correction(s)")
            if learned_rate >= 0.85:
                break
        if not progressed:
            break  # it obeys every cooking question and generalized as far as it will

    console.print(Rule("After — a held-out programming question", style="dim"))
    learned_rate, (hq, hr, hok) = held_out_rate("adapter ON (learned)")
    console.print(f"  Q: {hq}")
    console.print(f"  A: {hr[:160]!r}   {'[green]✓ one sentence[/]' if hok else '[red]✗[/]'}")

    console.print(Rule("A/B: toggle the fix off", style="dim"))
    ctrl.rollback(0)
    off_rate, _ = held_out_rate("adapter OFF (base)")
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1)

    console.print(Rule("Result", style="green"))
    console.print(f"  {corrections} plain corrections on cooking → one-sentence habit on held-out "
                  f"programming: base [red]{base_rate*100:.0f}%[/] → learned [green]{learned_rate*100:.0f}%[/] "
                  f"(off again: {off_rate*100:.0f}%).")
    console.print("  You told it once, by fixing a few replies. It generalized to a different topic,"
                  " on-device — you never have to say it again.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
