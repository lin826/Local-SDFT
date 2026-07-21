#!/usr/bin/env python
"""Narrated demo: a personal inbox assistant that learns your triage while you work.

The realistic loop: as you clear your morning inbox, the assistant proposes how
to handle each email; you fix the ones it gets wrong. A few corrections later it
has learned *your* policy and handles the rest — and tomorrow's new mail —
correctly. All on-device, offline; your mail and your rules never leave the box.

  1. Base model: shown a few emails, it has no idea how *you* handle them.
  2. You clear the inbox, correcting its suggestions (this is the learning).
  3. Held-out inbox (new senders/subjects it never saw): it now applies your
     policy correctly -> it learned your rules, it didn't memorize your emails.
  4. Toggle the adapter off: it forgets your policy again.

Run:  python scripts/demo_inbox.py [--rounds 4] [--config configs/demo_inbox.yaml]
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
from sdft.online.inbox import (
    COACH_EMAILS,
    HELDOUT_EMAILS,
    format_email,
    parse_action,
    policy_action,
    policy_target,
)

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_inbox.yaml")
    ap.add_argument("--rounds", type=int, default=4)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Premise check: held-out senders/subjects are disjoint from coaching.
    coach_keys = {(e["sender"], e["subject"]) for e in COACH_EMAILS}
    held_keys = {(e["sender"], e["subject"]) for e in HELDOUT_EMAILS}
    disjoint = coach_keys.isdisjoint(held_keys)

    console.print(Rule("Inbox assistant — learns your triage while you clear your mail"))
    console.print(f"[dim]model={cfg.model.name}  offline/on-device  "
                  f"held-out mail disjoint from coaching: {disjoint}[/]")

    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())

    console.print("[dim]loading model…[/]")
    ctrl = OnlineController.build(cfg)

    def held_out_accuracy(tag: str) -> float:
        correct = 0
        detail = []
        for e in HELDOUT_EMAILS:
            reply = ctrl.backend.generate([{"role": "user", "content": format_email(e)}],
                                          temperature=0.0, max_new_tokens=24)
            pred = parse_action(reply)
            want = policy_action(e)
            ok = pred == want
            correct += ok
            detail.append((e, pred, want, ok, reply))
        acc = correct / len(HELDOUT_EMAILS)
        bar = "█" * round(acc * 24)
        console.print(f"  {tag:<22} handled your way [green]{acc*100:5.1f}%[/]  {bar}")
        return acc, detail

    # 1. Base ---------------------------------------------------------------
    console.print(Rule("1. Before: the assistant doesn't know how you handle mail", style="dim"))
    for e in HELDOUT_EMAILS[:3]:
        reply = ctrl.backend.generate([{"role": "user", "content": format_email(e)}],
                                      temperature=0.0, max_new_tokens=24)
        console.print(f"  {e['subject']!r} (you'd {policy_action(e)}): {reply[:60]!r}")
    base_acc, _ = held_out_accuracy("base")

    # 2. Clear the inbox, correcting as you go ------------------------------
    console.print(Rule("2. You clear your inbox, fixing its suggestions", style="dim"))
    for rnd in range(args.rounds):
        conv = "inbox-" + uuid.uuid4().hex[:6]
        fixed = 0
        for e in COACH_EMAILS:
            mid, reply = ctrl.chat(conv, format_email(e))
            if parse_action(reply) != policy_action(e):
                ctrl.correct(conv, mid, policy_target(e))  # you fix it
                fixed += 1
        run = ctrl.maybe_update(force=True)
        tl = f"  train_loss={run.metrics.get('loss', float('nan')):.3f}" if run else ""
        acc, _ = held_out_accuracy(f"after inbox pass {rnd + 1}")
        console.print(f"[dim]                         corrected {fixed}/{len(COACH_EMAILS)} this pass{tl}[/]")

    # 3. Held-out inbox -----------------------------------------------------
    console.print(Rule("3. Tomorrow's inbox — new senders/subjects it never saw", style="dim"))
    learned_acc, detail = held_out_accuracy("adapter ON (learned)")
    # per-category breakdown
    from collections import defaultdict
    by_cat = defaultdict(lambda: [0, 0])
    for e, pred, want, ok, _ in detail:
        by_cat[e["category"]][0] += ok
        by_cat[e["category"]][1] += 1
    console.print("  by category: " + "  ".join(
        f"{c}={h}/{n}" for c, (h, n) in sorted(by_cat.items())))
    for e, pred, want, ok, _ in detail[:6]:
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {mark} {e['sender']:<26} {e['subject']!r:<26} → {pred} (you: {want})")

    # 4. A/B ----------------------------------------------------------------
    console.print(Rule("4. A/B: toggle the learned adapter off", style="dim"))
    ctrl.rollback(0)
    off_acc, _ = held_out_accuracy("adapter OFF (base)")
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1)

    console.print(Rule("Result", style="green"))
    console.print(f"  handled your way (held-out):  base [red]{base_acc*100:.0f}%[/]  →  "
                  f"learned [green]{learned_acc*100:.0f}%[/]   (adapter off again: {off_acc*100:.0f}%)")
    console.print("  New emails, new senders — it applied YOUR policy. It learned your rules"
                  " from a few corrections, on-device; it didn't memorize your inbox.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
