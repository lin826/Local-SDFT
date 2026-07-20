"""GRPO baseline — LoRA Group Relative Policy Optimization via TRL.

Trains adapters with on-policy rollouts and a local reward function. Designed
to run at small batch sizes on Apple Silicon (no vLLM): set
``grpo.batch_size == grpo.num_generations`` (e.g. both 2).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from trl import GRPOConfig, GRPOTrainer

from .config import load_config
from .rewards import resolve_reward
from .utils import load_model, load_tokenizer, pick_device


def load_grpo_dataset(path: str) -> Dataset:
    """Build a GRPO dataset with ``prompt`` (+ optional ``gold``) columns."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    items: list[dict] = []
    for row in rows:
        prompt = row.get("prompt")
        if isinstance(prompt, str):
            prompt_messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, list):
            prompt_messages = prompt
        else:
            continue
        item: dict = {"prompt": prompt_messages}
        gold = row.get("response") or row.get("gold") or row.get("ground_truth")
        if gold:
            item["gold"] = str(gold)
            item["ground_truth"] = str(gold)
        items.append(item)
    if not items:
        raise SystemExit(f"no GRPO rows in {path}")
    return Dataset.from_list(items)


def examples_to_grpo_jsonl(examples: list[dict], out_path: str | Path) -> Path:
    """Write gold ``{prompt, response}`` examples as GRPO training jsonl."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps({"prompt": ex["prompt"], "response": ex["response"]}, ensure_ascii=False) + "\n")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/compare/batch1_grpo.yaml")
    parser.add_argument("--data", default=None, help="jsonl with prompt (+ optional response/gold)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="optional step cap for smoke runs")
    args = parser.parse_args()

    cfg = load_config(args.config)
    g = cfg.grpo
    data_path = args.data or cfg.generation.out_path.replace("sdft_", "grpo_")
    if args.data is None and not Path(data_path).is_file():
        # Fall back to SDFT jsonl (uses gold ``response`` column when present).
        data_path = cfg.generation.out_path
    output_dir = args.output_dir or g.output_dir
    epochs = args.epochs if args.epochs is not None else g.epochs

    if g.batch_size % g.num_generations != 0:
        raise SystemExit(
            f"grpo.batch_size ({g.batch_size}) must be divisible by "
            f"grpo.num_generations ({g.num_generations})"
        )

    device = pick_device()
    print(f"device: {device}")
    ds = load_grpo_dataset(data_path)
    print(f"GRPO on {len(ds)} prompts from {data_path} (reward={g.reward})")

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
    grpo_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        learning_rate=g.lr,
        per_device_train_batch_size=g.batch_size,
        gradient_accumulation_steps=g.grad_accum,
        num_generations=g.num_generations,
        generation_batch_size=g.batch_size,
        max_completion_length=g.max_completion_length,
        temperature=g.temperature,
        warmup_steps=g.warmup_steps,
        logging_steps=g.logging_steps,
        save_strategy=g.save_strategy,
        seed=g.seed,
        report_to=[],
        use_vllm=False,
        use_cpu=(device == "cpu"),
    )
    # max_prompt_length may or may not exist depending on TRL version
    if hasattr(grpo_args, "max_prompt_length"):
        grpo_args.max_prompt_length = g.max_prompt_length

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=resolve_reward(g.reward),
        args=grpo_args,
        train_dataset=ds,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"saved LoRA adapter to {output_dir}")


if __name__ == "__main__":
    main()
