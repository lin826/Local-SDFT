"""CLI for the online SDFT loop.

    python -m sdft.online.cli chat      --config configs/online.yaml
    python -m sdft.online.cli serve     --config configs/online.yaml
    python -m sdft.online.cli demo      --config configs/demo_house_style.yaml
    python -m sdft.online.cli stats     --config configs/online.yaml

`demo` runs the "Airplane-Mode Coach" headlessly: measure success on held-out
prompts, coach in rounds, and watch the number climb — the CLI twin of the web
demo, for cluster/debug runs.
"""

from __future__ import annotations

import argparse
import sys
import uuid

from rich.console import Console

from ..config import Config, load_config
from .controller import OnlineController

console = Console()


def _cfg(args) -> Config:
    cfg = load_config(args.config) if args.config else Config()
    if args.model:
        cfg.model.name = args.model
    if args.backend:
        cfg.online.backend = args.backend
    return cfg


def cmd_chat(args) -> int:
    cfg = _cfg(args)
    console.print(f"[dim]loading {cfg.model.name} ({cfg.online.backend})…[/]")
    ctrl = OnlineController.build(cfg)
    conv = uuid.uuid4().hex[:8]
    last_id = None
    console.print("[dim]/correct <text> · /train · /stats · /rollback [v] · /new · /quit[/]")
    while True:
        try:
            line = console.input("[bold cyan]you>[/] ").strip()
        except (EOFError, KeyboardInterrupt):
            line = "/quit"
        if not line:
            continue
        if line.startswith("/"):
            cmd, _, rest = line.partition(" ")
            if cmd == "/quit":
                ctrl.close_conversation(conv); ctrl.maybe_update(); break
            if cmd == "/new":
                ctrl.close_conversation(conv); ctrl.maybe_update()
                conv = uuid.uuid4().hex[:8]; last_id = None; continue
            if cmd == "/correct":
                if last_id and rest.strip():
                    if ctrl.correct(conv, last_id, rest.strip()):
                        console.print("[green]recorded[/]")
                        r = ctrl.maybe_update()
                        if r: console.print(f"[magenta]adapter v{r.adapter_version} loss {r.metrics.get('loss', float('nan')):.4f}[/]")
                continue
            if cmd == "/train":
                r = ctrl.maybe_update(force=True)
                console.print(f"[magenta]adapter v{r.adapter_version}[/]" if r else "[dim]nothing to train[/]"); continue
            if cmd == "/stats":
                for k, v in ctrl.stats().items(): console.print(f"  {k}: {v}")
                continue
            if cmd == "/rollback":
                av = ctrl.rollback(int(rest) if rest.strip().isdigit() else None)
                console.print(f"[green]-> v{av.version}[/]" if av else "[yellow]nothing[/]"); continue
            console.print("[yellow]unknown command[/]"); continue
        last_id, reply = ctrl.chat(conv, line)
        console.print(f"[blue]model>[/] {reply}")
        r = ctrl.maybe_update()
        if r: console.print(f"[magenta]learned → adapter v{r.adapter_version}[/]")
    return 0


def cmd_serve(args) -> int:
    import uvicorn

    from .serve import create_app

    cfg = _cfg(args)
    ctrl = OnlineController.build(cfg)
    _attach_probe_hook(ctrl, cfg)
    app = create_app(ctrl)
    console.print(f"[green]http://{cfg.online.host}:{cfg.online.port}[/]")
    uvicorn.run(app, host=cfg.online.host, port=cfg.online.port, log_level="info")
    return 0


def cmd_demo(args) -> int:
    from .demo import HELDOUT_PROMPTS, success_on

    cfg = _cfg(args)
    if not cfg.online.reward_fn:
        console.print("[red]demo needs online.reward_fn (e.g. house_style)[/]"); return 2
    console.print(f"[dim]loading {cfg.model.name} · task={cfg.online.reward_fn}[/]")
    ctrl = OnlineController.build(cfg)
    from .reward import get_reward_fn

    rfn = get_reward_fn(cfg.online.reward_fn)

    def report(tag):
        res = success_on(ctrl.backend, rfn, HELDOUT_PROMPTS)
        bar = "█" * round(res["success"] * 20)
        console.print(f"  {tag:<18} success@held-out [green]{res['success']*100:5.1f}%[/] {bar}")
        return res["success"]

    console.print("[bold]Airplane-Mode Coach (offline, on-device)[/]")
    report("before coaching")
    for rnd in range(args.rounds):
        conv = "coach-" + uuid.uuid4().hex[:6]
        from .demo import COACH_PROMPTS
        for i in range(args.coach_per_round):
            ctrl.chat(conv, COACH_PROMPTS[(rnd * args.coach_per_round + i) % len(COACH_PROMPTS)])
        ctrl.maybe_update(force=True)
        report(f"after round {rnd + 1}")
    # A/B: base vs learned on the SAME held-out set
    console.print("[dim]toggling adapter OFF (base) for A/B…[/]")
    ctrl.rollback(0); report("adapter OFF (base)")
    ctrl.rollback(ctrl.stats()["adapter_versions"] - 1); report("adapter ON (learned)")
    return 0


def cmd_stats(args) -> int:
    ctrl = OnlineController.build(_cfg(args))
    for k, v in ctrl.stats().items():
        console.print(f"{k}: {v}")
    return 0


def _attach_probe_hook(ctrl, cfg) -> None:
    if cfg.online.eval_every_n_updates <= 0 or cfg.online.backend == "echo":
        return
    from .probes import ProbeEvaluator

    ev = ProbeEvaluator()
    ev.capture_baseline(ctrl.backend)
    ctrl.eval_hook = ev


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="sdft-online", description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--backend", default=None, choices=["torch", "echo"])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("chat")
    sub.add_parser("serve")
    d = sub.add_parser("demo")
    d.add_argument("--rounds", type=int, default=5)
    d.add_argument("--coach-per-round", type=int, default=4)
    sub.add_parser("stats")

    args = ap.parse_args(argv)
    import logging
    logging.basicConfig(level=logging.WARNING, format="%(name)s: %(message)s")
    return {"chat": cmd_chat, "serve": cmd_serve, "demo": cmd_demo, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
