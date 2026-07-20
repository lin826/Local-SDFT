"""SDFT step 2 — LoRA fine-tuning on self-generated (or gold) responses.

Trains only the LoRA adapters (attention projections by default); the base
weights stay frozen. Loss is computed on the completion tokens only.

Use ``--target sdft`` (default) for SDFT, or ``--target gold`` for a supervised
baseline on the original ``response`` field.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from trl import SFTConfig, SFTTrainer

from .config import load_config
from .utils import load_model, load_tokenizer, pick_device


def load_sft_dataset(path: str, *, target_field: str = "sdft_response") -> Dataset:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows:
        return Dataset.from_list([])

    def _completion(row: dict) -> str:
        text = str(row.get(target_field) or "").strip()
        if target_field == "sdft_response" and not text:
            text = str(row.get("response") or "").strip()
        return text

    # Pre-rendered OpenClaw prefixes (string prompt) — matches tool-loop eval.
    if isinstance(rows[0].get("prompt"), str):
        return Dataset.from_list(
            [
                {"prompt": row["prompt"], "completion": completion}
                for row in rows
                if (completion := _completion(row))
            ]
        )

    return Dataset.from_list(
        [
            {
                "prompt": [{"role": "user", "content": row["prompt"]}],
                "completion": [{"role": "assistant", "content": completion}],
            }
            for row in rows
            if (completion := _completion(row))
        ]
    )


# Back-compat alias
load_sdft_dataset = load_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data", default=None, help="training jsonl (default: generation.out_path)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument(
        "--target",
        choices=("sdft", "gold"),
        default=None,
        help="sdft -> sdft_response field; gold -> response field (SFT baseline)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    t = cfg.training
    data_path = args.data or cfg.generation.out_path
    output_dir = args.output_dir or t.output_dir
    epochs = args.epochs if args.epochs is not None else t.epochs

    if args.target == "gold":
        target_field = "response"
    elif args.target == "sdft":
        target_field = "sdft_response"
    else:
        target_field = t.target_field

    device = pick_device()
    print(f"device: {device}")
    ds = load_sft_dataset(data_path, target_field=target_field)
    print(f"training on {len(ds)} pairs from {data_path} (target={target_field})")

    tokenizer = load_tokenizer(cfg.model)
    model = load_model(cfg.model, device)
    model.config.use_cache = False

    peft_config = PeftLoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.alpha,
        lora_dropout=cfg.lora.dropout,
        target_modules=cfg.lora.target_modules,
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        learning_rate=t.lr,
        per_device_train_batch_size=t.batch_size,
        gradient_accumulation_steps=t.grad_accum,
        max_length=t.max_length,
        warmup_steps=t.warmup_steps,
        logging_steps=t.logging_steps,
        save_strategy=t.save_strategy,
        seed=t.seed,
        report_to=[],
        completion_only_loss=True,
        use_cpu=(device == "cpu"),
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    trainer.save_model(output_dir)  # saves the adapter (base weights untouched)
    tokenizer.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
