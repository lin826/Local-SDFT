"""Inspect the model: device, parameter count, and LoRA-targetable modules."""

from __future__ import annotations

import argparse
from collections import Counter

import torch.nn as nn

from .config import load_config
from .utils import load_model, model_device, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = pick_device()
    model = load_model(cfg.model, device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: {cfg.model.name} on {device} (params @ {model_device(model)})")
    print(f"parameters: {n_params / 1e6:.1f}M")
    print(f"model class: {type(model).__name__}")

    leaf_counts: Counter[str] = Counter()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            leaf_counts[name.rsplit(".", 1)[-1]] += 1
    print("LoRA-targetable leaf module names (Linear/Conv1d):")
    for leaf, count in sorted(leaf_counts.items()):
        print(f"  {leaf:<16} x{count}")

    # One example full path per leaf name to disambiguate where leaves live.
    seen: dict[str, str] = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            seen.setdefault(name.rsplit(".", 1)[-1], name)
    print("example paths:")
    for leaf, path in sorted(seen.items()):
        print(f"  {leaf:<16} <- {path}")


if __name__ == "__main__":
    main()
