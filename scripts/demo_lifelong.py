#!/usr/bin/env python
"""Lifelong learning: your assistant keeps getting more capable, on-device.

Most demos teach one trick. This one runs a *curriculum*: four distinct,
trigger-keyed skills arrive one at a time, the way a real assistant would pick
up your habits over a week of use —

    Summarize: <text>   -> a one-line summary
    What is A op B?      -> a calculator tool call
    List <topic>         -> three bullet points
    Reply to: <message>  -> a reply ending with your fixed sign-off

After each skill is taught we re-measure ALL skills learned so far on held-out
prompts. That's the whole point: does the assistant *accumulate* a repertoire,
or does learning the new thing overwrite the old ones?

We run the identical curriculum twice:
  * WITHOUT replay  — the classic failure: each new skill clobbers the last
                      (catastrophic forgetting; capability never grows past 1).
  * WITH replay     — the on-device loop rehearses a few past examples of every
                      skill each update, so the repertoire GROWS and stays up.

This is the case for experience replay in online SDFT, and it needs a long
horizon to see — which is exactly what this demo gives you.

Run:  python scripts/demo_lifelong.py [--rounds 3] [--coach 8]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import uuid
from pathlib import Path

from rich.console import Console
from rich.rule import Rule
from rich.table import Table

from sdft.config import load_config
from sdft.online.controller import OnlineController
from sdft.online.demo import SKILLS, success_on
from sdft.online.reward import get_reward_fn

console = Console()
THRESH = 0.6  # held-out success at which a skill counts as "retained"


def _clear(cfg) -> None:
    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())


def run_curriculum(cfg, replay_ratio: float, rounds: int, coach_n: int) -> dict:
    """Teach each skill in turn; snapshot every skill's held-out success after
    each skill's training block. Returns {finished_skill: {skill: success}}."""
    cfg.online.replay_ratio = replay_ratio
    _clear(cfg)
    ctrl = OnlineController.build(cfg)

    reward_fns = {name: get_reward_fn(rf) for name, rf, _, _, _ in SKILLS}
    heldouts = {name: ho for name, _, (_, ho), _, _ in SKILLS}

    def eval_all(names):
        return {s: success_on(ctrl.backend, reward_fns[s], heldouts[s],
                              threshold=THRESH)["success"] for s in names}

    introduced: list[str] = []
    snapshots: dict[str, dict] = {}
    for name, rf, (coach, _), desc, hint in SKILLS:
        introduced.append(name)
        ctrl.set_task(rf)
        ctrl.cfg.online.coach_instruction = hint  # cold-start teacher hint for this skill
        console.print(Rule(f"teach '{name}' ({desc})  ·  now juggling {len(introduced)} skill(s)",
                           style="dim"))
        for r in range(rounds):
            conv = "L-" + uuid.uuid4().hex[:6]
            for i in range(coach_n):
                ctrl.chat(conv, coach[i % len(coach)])
            ctrl.maybe_update(force=True)
        row = eval_all(introduced)
        snapshots[name] = row
        cells = "  ".join(f"{s}:[{'green' if row[s] >= THRESH else 'red'}]{row[s]*100:3.0f}%[/]"
                          for s in introduced)
        console.print(f"  after '{name}':  {cells}")

    ctrl.store.close()
    return snapshots


def render_stats(snapshots: dict) -> tuple[int, float]:
    """(skills retained ≥ THRESH, mean held-out success) after the full curriculum.

    Measured on the final column (every skill taught) — mean is the less-noisy
    headline for how firmly the whole repertoire is held.
    """
    final = snapshots.get([n for n, *_ in SKILLS][-1], {})
    retained = sum(1 for v in final.values() if v >= THRESH)
    mean = sum(final.values()) / len(final) if final else 0.0
    return retained, mean


def render(tag: str, snapshots: dict) -> tuple[int, float]:
    """Print the accumulation matrix (rows = skills, cols = 'after teaching X')."""
    names = [n for n, *_ in SKILLS]
    table = Table(title=f"{tag}: held-out success as skills accumulate", title_style="bold")
    table.add_column("skill \\ after", style="cyan")
    for n in names:
        table.add_column(n, justify="right")
    for skill in names:
        cells = []
        for col in names:
            snap = snapshots.get(col, {})
            if skill not in snap:
                cells.append("[dim]·[/]")  # not taught yet
            else:
                v = snap[skill]
                cells.append(f"[{'green' if v >= THRESH else 'red'}]{v*100:.0f}%[/]")
        table.add_row(skill, *cells)
    console.print(table)
    return render_stats(snapshots)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_lifelong.yaml")
    ap.add_argument("--rounds", type=int, default=6, help="update rounds per skill")
    ap.add_argument("--coach", type=int, default=8, help="coach prompts per round")
    ap.add_argument("--replay", type=float, default=0.5, help="replay_ratio for the WITH-replay run")
    ap.add_argument("--ranks", default="32,2",
                    help="comma-separated LoRA ranks to sweep. Ample capacity (e.g. 32) vs the "
                         "tight always-on on-device budget (e.g. 2) — replay only matters when "
                         "capacity is scarce enough that skills compete. Pass one value to skip the sweep.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    n = len(SKILLS)

    console.print(Rule("Lifelong learning on-device — accumulate a repertoire, don't overwrite it"))
    console.print(f"[dim]model={cfg.model.name}  offline  ·  {n} skills, {args.rounds} rounds each  ·  "
                  f"LoRA ranks {ranks}  ·  retained = held-out ≥ {THRESH*100:.0f}%[/]")

    # capacity -> {"off": (retained, mean, snaps), "on": (...)}
    results: dict[int, dict] = {}
    for r in ranks:
        cfg.lora.r = r
        cfg.lora.alpha = 2 * r
        label = "ample" if r >= 16 else "tight"
        console.print(Rule(f"Capacity: LoRA r={r} ({label})  ·  no-replay vs replay", style="bold"))
        snaps_off = run_curriculum(cfg, 0.0, args.rounds, args.coach)
        console.print(f"  [red]↑ without replay[/]   [green]↓ with replay[/]")
        snaps_on = run_curriculum(cfg, args.replay, args.rounds, args.coach)
        results[r] = {"off": (*render_stats(snaps_off),), "on": (*render_stats(snaps_on),),
                      "snaps_off": snaps_off, "snaps_on": snaps_on}

    # Detail matrices for the tightest rank — where the dynamics are visible.
    tight = min(ranks)
    console.print(Rule(f"Held-out matrices at the tight budget (r={tight})", style="bold"))
    render(f"r={tight} WITHOUT replay", results[tight]["snaps_off"])
    console.print()
    render(f"r={tight} WITH replay", results[tight]["snaps_on"])

    # Capacity x replay summary.
    console.print(Rule("Capacity × replay — mean held-out across all skills", style="bold"))
    table = Table()
    table.add_column("LoRA rank", style="cyan")
    table.add_column("no replay", justify="right")
    table.add_column("replay", justify="right")
    table.add_column("replay gain", justify="right")
    for r in ranks:
        (_, m_off), (_, m_on) = results[r]["off"], results[r]["on"]
        gain = (m_on - m_off) * 100
        gcol = "green" if gain > 1 else "dim"
        table.add_row(f"r={r} ({'ample' if r >= 16 else 'tight'})",
                      f"{m_off*100:.0f}%", f"{m_on*100:.0f}%",
                      f"[{gcol}]{gain:+.0f} pts[/]")
    console.print(table)

    console.print(Rule("The point", style="green"))
    (_, m_off_t), (_, m_on_t) = results[tight]["off"], results[tight]["on"]
    console.print(
        f"  Well-separated skills accumulate for free when the adapter has room. But at the "
        f"tight always-on on-device budget (r={tight}), skills compete for capacity: without "
        f"replay the repertoire degrades ([red]{m_off_t*100:.0f}%[/]), while rehearsing a few "
        f"past examples of every skill each update holds it together ([green]{m_on_t*100:.0f}%[/]).")
    console.print("  → experience replay is what makes online SDFT stable exactly where edge "
                  "devices live — scarce capacity, learning continually, no second chance at old data.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
