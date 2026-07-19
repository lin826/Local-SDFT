"""SDFT step 3 — merge a trained LoRA adapter back into the base weights."""

from __future__ import annotations

import argparse

from peft import PeftModel

from .config import load_config
from .utils import load_model, load_tokenizer, pick_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--adapter", default=None, help="adapter dir (default: training.output_dir)")
    parser.add_argument("--out", default=None, help="merged output dir (default: <adapter>-merged)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    adapter = args.adapter or cfg.training.output_dir
    out = args.out or f"{adapter}-merged"

    base = load_model(cfg.model, pick_device())
    model = PeftModel.from_pretrained(base, adapter)
    model = model.merge_and_unload()
    model.save_pretrained(out)
    load_tokenizer(cfg.model).save_pretrained(out)
    print(f"merged model saved to {out}")


if __name__ == "__main__":
    main()
