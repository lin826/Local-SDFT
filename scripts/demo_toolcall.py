#!/usr/bin/env python
"""Narrated tool-calling demo: the model LEARNS to use a calculator.

Story (all on-device, offline):
  1. The base LFM2.5-230M answers arithmetic freehand — and gets it wrong.
  2. We coach it: each turn it samples answers, we keep a correct calc() tool
     call as the training target, and take a few LoRA steps. A success curve on
     HELD-OUT problems climbs.
  3. On held-out problems whose numbers NEVER appeared in coaching, it now emits
     correct tool calls -> exact answers. It can't have memorized them, so it
     learned the skill (question -> tool call).
  4. Toggle the adapter off: it's confidently wrong again.

Run:  python scripts/demo_toolcall.py [--rounds 6] [--config configs/demo_toolcall.yaml]
"""

from __future__ import annotations

import argparse
import sys
import uuid

from rich.console import Console
from rich.rule import Rule

from sdft.config import load_config
from sdft.online.controller import OnlineController
from sdft.online.demo import prompts_for
from sdft.online.tools import extract_arithmetic, parse_calc_call, run_calc_call, safe_eval

console = Console()


def answer_with_tools(backend, question: str) -> tuple[str, float | None]:
    """Serve one query; if the reply calls calc(), execute it for the result."""
    reply = backend.generate([{"role": "user", "content": question}],
                             temperature=0.0, max_new_tokens=32)
    result = run_calc_call(reply)  # None if the model didn't emit a usable call
    return reply, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_toolcall.yaml")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--coach-per-round", type=int, default=6)
    args = ap.parse_args()

    cfg = load_config(args.config)
    assert cfg.online.reward_fn == "calc_tool", "use configs/demo_toolcall.yaml"
    coach_prompts, heldout = prompts_for("calc_tool")

    # Prove the premise: held-out numbers are disjoint from coaching numbers.
    coach_nums = {n for p in coach_prompts for n in _nums(p)}
    held_nums = {n for p in heldout for n in _nums(p)}
    disjoint = coach_nums.isdisjoint(held_nums)

    console.print(Rule("Tool-calling demo — the model learns to use a calculator"))
    console.print(f"[dim]model={cfg.model.name}  offline/on-device  "
                  f"held-out numbers disjoint from coaching: {disjoint}[/]")

    # Fresh state so the curve reflects this run only.
    import shutil
    from pathlib import Path
    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())

    console.print("[dim]loading model…[/]")
    ctrl = OnlineController.build(cfg)

    def success(tag: str) -> float:
        det = []
        hits = 0
        for q in heldout:
            reply, result = answer_with_tools(ctrl.backend, q)
            truth = safe_eval(extract_arithmetic(q))
            ok = result is not None and truth is not None and abs(result - truth) < 1e-6
            hits += ok
            det.append((q, reply, result, truth, ok))
        rate = hits / len(heldout)
        bar = "█" * round(rate * 24)
        console.print(f"  {tag:<20} held-out correct [green]{rate*100:5.1f}%[/]  {bar}")
        return rate, det

    # 1. Base model, freehand ---------------------------------------------
    console.print(Rule("1. Base model answers arithmetic freehand", style="dim"))
    for q in heldout[:3]:
        reply, result = answer_with_tools(ctrl.backend, q)
        truth = safe_eval(extract_arithmetic(q))
        called = parse_calc_call(reply) is not None
        console.print(f"  Q: {q}")
        console.print(f"     reply: {reply[:70]!r}  (tool call: {called})  truth={truth}")
    base_rate, _ = success("base")

    # 2. Coach -------------------------------------------------------------
    console.print(Rule("2. Coach it (reward = correct calc() call)", style="dim"))
    for rnd in range(args.rounds):
        conv = "coach-" + uuid.uuid4().hex[:6]
        for i in range(args.coach_per_round):
            ctrl.chat(conv, coach_prompts[(rnd * args.coach_per_round + i) % len(coach_prompts)])
        run = ctrl.maybe_update(force=True)
        tl = f"  train_loss={run.metrics.get('loss', float('nan')):.3f}" if run else ""
        rate, _ = success(f"after round {rnd + 1}")
        if run:
            console.print(f"[dim]                       {tl.strip()}[/]")

    # 3. Held-out generalization ------------------------------------------
    console.print(Rule("3. Held-out problems (numbers never coached on)", style="dim"))
    learned_rate, det = success("adapter ON (learned)")
    for q, reply, result, truth, ok in det[:4]:
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        console.print(f"  {mark} {q}  ->  {parse_calc_call(reply)!r} = {result}  (truth {truth})")

    # 4. A/B toggle --------------------------------------------------------
    console.print(Rule("4. A/B: toggle the learned adapter off", style="dim"))
    ctrl.rollback(0)
    off_rate, _ = success("adapter OFF (base)")
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1)

    console.print(Rule("Result", style="green"))
    console.print(f"  held-out correct:  base [red]{base_rate*100:.0f}%[/]  ->  "
                  f"learned [green]{learned_rate*100:.0f}%[/]   (adapter off again: {off_rate*100:.0f}%)")
    console.print("  The held-out numbers never appeared in coaching, so this is a learned"
                  " skill (question -> tool call), not memorized answers.")
    return 0


def _nums(text: str) -> set[str]:
    import re
    return set(re.findall(r"\d+", text))


if __name__ == "__main__":
    sys.exit(main())
