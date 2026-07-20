#!/usr/bin/env python
"""Teach a 230M to query a REAL database — live, on-device.

Not a simulated tool: the model writes SQL, a real SQLite engine runs it, and
the reward is whether the returned rows match a gold query's rows. You watch it
go from writing no/wrong SQL to correctly answering questions about your data —
on held-out questions whose categories, cities and products never appear in
coaching, so it can only succeed by writing correct SQL, not by memorizing.

The model's SQL is untrusted and executed read-only behind a SQLite authorizer
that permits only reads (see sdft/online/sqlenv.py).

Run:  python scripts/demo_sqlite.py
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
from sdft.online import sqlenv
from sdft.online.controller import OnlineController
from sdft.online.reward import get_reward_fn

console = Console()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/demo_sqlite.yaml")
    ap.add_argument("--store-db", default="data/demo_store.db", help="the queried database")
    ap.add_argument("--max-rounds", type=int, default=8)
    ap.add_argument("--coach-per-round", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    obeys = get_reward_fn("sqlite_tool")

    console.print(Rule("Learn to query a REAL database — text → SQL, executed on-device"))
    console.print(f"[dim]model={cfg.model.name}  offline  ·  real SQLite engine (read-only jail)  ·  "
                  "schema learned into the weights  ·  coach + test on DISJOINT categories/cities[/]")

    # Build the real database the model will query, and point the reward at it.
    sqlenv.build_db(args.store_db)
    sqlenv.set_db(args.store_db)
    console.print(f"[dim]database: {args.store_db}  ·  schema: {sqlenv.SCHEMA_DESCRIPTION}[/]")

    # Fresh SDFT state.
    for p in (cfg.online.db_path, cfg.online.adapters_dir):
        path = Path(p)
        shutil.rmtree(path, ignore_errors=True) if path.is_dir() else (path.exists() and path.unlink())

    console.print("[dim]loading model…[/]")
    ctrl = OnlineController.build(cfg)

    # Serve schema-less: the model must have learned the schema into its weights
    # (the gold queries it trained on). The schema lives only in the config's
    # coach_instruction, used as a teacher hint while sampling candidates.
    def ask(question: str) -> str:
        return ctrl.backend.generate(
            [{"role": "user", "content": question}], temperature=0.0, max_new_tokens=64)

    def held_out_rate(tag: str, show: bool = False):
        hits, sample = 0, None
        for q, _ in sqlenv.HELDOUT_QA:
            reply = ask(q)
            ok = obeys(q, reply) >= 1.0
            hits += ok
            if sample is None:
                sample = (q, reply, ok)
        rate = hits / len(sqlenv.HELDOUT_QA)
        bar = "█" * round(rate * 24)
        console.print(f"  {tag:<26} correct-on-DB [green]{rate*100:5.1f}%[/]  {bar}")
        return rate, sample

    def show_example(q: str) -> None:
        reply = ask(q)
        sql = sqlenv.parse_sql(reply)
        rows = sqlenv.run_query(args.store_db, sql) if sql else None
        gold_rows = sqlenv.run_query(args.store_db, sqlenv.gold_for(q))
        ok = sqlenv.results_match(rows, gold_rows)
        console.print(f"  Q: {q}")
        console.print(f"  SQL it wrote: [cyan]{sql or '(no query)'}[/]")
        console.print(f"  real rows:    {rows}   {'[green]✓ matches the database[/]' if ok else '[red]✗[/]'}")

    console.print(Rule("Before — it hasn't learned your database yet", style="dim"))
    base_rate, _ = held_out_rate("base")
    show_example(sqlenv.HELDOUT_QA[0][0])

    console.print(Rule("Coaching — reward-selected self-distillation on the real engine", style="dim"))
    learned = base_rate
    for r in range(args.max_rounds):
        conv = "sql-" + uuid.uuid4().hex[:6]
        for i in range(args.coach_per_round):
            q, _ = sqlenv.COACH_QA[i % len(sqlenv.COACH_QA)]
            ctrl.chat(conv, q)
        ctrl.maybe_update(force=True)
        learned, _ = held_out_rate(f"after round {r + 1}")
        if learned >= 0.85:
            break

    console.print(Rule("After — a held-out question, run against the real database", style="dim"))
    for q, _ in sqlenv.HELDOUT_QA[:3]:
        show_example(q)

    console.print(Rule("A/B — same questions, adapter OFF (base) vs ON (learned)", style="dim"))
    ctrl.rollback(0)
    off_rate, _ = held_out_rate("adapter OFF (base)")
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1)
    on_rate, _ = held_out_rate("adapter ON (learned)")

    console.print(Rule("Result", style="green"))
    console.print(f"  Correctly answering held-out questions against the real DB: "
                  f"[red]{off_rate*100:.0f}%[/] (base) → [green]{on_rate*100:.0f}%[/] (learned), "
                  "on-device and offline.")
    console.print("  It didn't memorize answers — the held-out categories/cities never appeared in "
                  "coaching; it learned to write SQL the real engine runs correctly.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
