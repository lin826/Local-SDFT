#!/usr/bin/env python
"""Continual-learning demo: adapt to the current mode, recover old modes fast.

Realistic framing: your assistant's response *mode* changes with what you're
doing. In the morning you want structured briefings; midday you want terse
one-liners. It adapts to whichever you're using now. When you switch, it may
forget the other mode — that's fine — but when you switch BACK, it re-learns in
far fewer steps than the first time. Plasticity + fast recovery, all on-device.

  Phase 1 — learn mode A (briefing):   A ↑, count passes to competence  = P1
  Phase 2 — switch to mode B (terse):  B ↑, A ↓ (forgetting is fine)
  Phase 3 — switch back to mode A:     A ↑ again, passes to competence  = P3
  The point: P3 << P1  (savings — it recovers the old skill quickly).

Run:  python scripts/demo_continual.py [--config configs/demo_continual.yaml]
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
from sdft.online.demo import HELDOUT_PROMPTS, prompts_for, success_on
from sdft.online.reward import get_reward_fn

console = Console()

MODE_A = ("house_style", "briefing (TL;DR + bullets + question)")
MODE_B = ("five_words", "five-word summary")
THRESH = 0.66


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_continual.yaml")
    ap.add_argument("--max-passes", type=int, default=6)
    ap.add_argument("--coach-per-pass", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    coach_prompts, heldout = prompts_for("house_style")  # generic questions; both modes apply
    rA, rB = get_reward_fn(MODE_A[0]), get_reward_fn(MODE_B[0])

    console.print(Rule("Continual learning — adapt now, recover old skills fast"))
    console.print(f"[dim]model={cfg.model.name}  offline/on-device  "
                  f"mode A = {MODE_A[1]}  ·  mode B = {MODE_B[1]}[/]")

    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())

    console.print("[dim]loading model…[/]")
    ctrl = OnlineController.build(cfg)

    def scores():
        a = success_on(ctrl.backend, rA, heldout, threshold=THRESH)["success"]
        b = success_on(ctrl.backend, rB, heldout, threshold=THRESH)["success"]
        return a, b

    def row(tag, a, b):
        ba = "█" * round(a * 16); bb = "█" * round(b * 16)
        console.print(f"  {tag:<16} A [green]{a*100:5.1f}%[/] {ba:<16}   B [cyan]{b*100:5.1f}%[/] {bb}")

    def learn(task_name: str, target_reward, label: str) -> int:
        """Coach the given mode until held-out success >= THRESH; return #passes."""
        ctrl.set_task(task_name)
        passes = 0
        for _ in range(args.max_passes):
            conv = "c-" + uuid.uuid4().hex[:6]
            for i in range(args.coach_per_pass):
                ctrl.chat(conv, coach_prompts[i % len(coach_prompts)])
            ctrl.maybe_update(force=True)
            passes += 1
            a, b = scores()
            row(f"{label} pass {passes}", a, b)
            reached = success_on(ctrl.backend, target_reward, heldout, threshold=THRESH)["success"]
            if reached >= THRESH:
                break
        return passes

    a0, b0 = scores(); row("start", a0, b0)

    console.print(Rule("Phase 1 — learn mode A (briefing)", style="dim"))
    p1 = learn(MODE_A[0], rA, "learn A")

    console.print(Rule("Phase 2 — switch to mode B (terse); A may fade", style="dim"))
    learn(MODE_B[0], rB, "learn B")

    console.print(Rule("Phase 3 — switch BACK to mode A", style="dim"))
    p3 = learn(MODE_A[0], rA, "relearn A")

    console.print(Rule("Result — fast recovery (savings)", style="green"))
    console.print(f"  mode A reached competence in [yellow]{p1}[/] pass(es) the first time, "
                  f"[green]{p3}[/] on return.")
    if p3 < p1:
        console.print(f"  → recovered the old skill [bold]{p1 - p3} pass(es) faster[/] — "
                      "forgetting is cheap because re-adaptation is fast.")
    else:
        console.print("  → recovery was not faster this run (small model / few passes); "
                      "the loop still adapts to the current mode.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
