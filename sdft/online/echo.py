"""Model-free backend for testing the online loop and demo plumbing."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config
from .events import Demonstration


class EchoTrainer:
    name = "echo"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = {"steps": 0}

    def load(self) -> None:
        pass

    def generate(self, messages: list[dict[str, str]], **overrides) -> str:
        last = messages[-1]["content"] if messages else ""
        return f"echo: {last[:64]}"

    def sample(self, messages: list[dict[str, str]], n: int = 1) -> list[str]:
        last = messages[-1]["content"] if messages else ""
        # Vary samples deterministically so reward selection has something to pick.
        return [f"echo[{i}]: {last[:48]}" for i in range(n)]

    def train_on_demos(self, demos: list[Demonstration]) -> dict[str, float]:
        self.state["steps"] += 1
        return {
            "loss": 1.0 / (1 + self.state["steps"]),
            "kl_to_base": 0.0,
            "completion_tokens": 8.0,
            "trained": float(len(demos)),
        }

    def save_adapter(self, path: str) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        (p / "echo_state.json").write_text(json.dumps(self.state))

    def load_adapter(self, path: str) -> None:
        f = Path(path) / "echo_state.json"
        if f.exists():
            self.state = json.loads(f.read_text())


def create_backend(cfg: Config):
    if cfg.online.backend == "torch":
        from .trainer import TorchTrainer

        return TorchTrainer(cfg)
    if cfg.online.backend == "echo":
        return EchoTrainer(cfg)
    raise ValueError(f"unknown online backend: {cfg.online.backend!r}")
