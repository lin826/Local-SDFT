"""Tiny LoRA SDFT step — train on model-generated ``sdft_response`` targets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig
from peft import PeftModel
from trl import SFTConfig, SFTTrainer

from sdft.config import Config
from sdft.utils import load_model, load_tokenizer, pick_device

from .generate_step import row_to_prompt


def _adapter_ready(adapter_dir: Path) -> bool:
    return (adapter_dir / "adapter_config.json").is_file()


def _rows_to_dataset(rows: list[dict[str, str]]) -> Dataset:
    items: list[dict[str, Any]] = []
    for row in rows:
        user_content = row_to_prompt(row)
        completion = row.get("sdft_response", "").strip()
        if not user_content or not completion:
            continue
        items.append(
            {
                "prompt": [{"role": "user", "content": user_content}],
                "completion": [{"role": "assistant", "content": completion}],
            }
        )
    if not items:
        raise ValueError("no trainable rows with sdft_response")
    return Dataset.from_list(items)


def run_train_step(
    cfg: Config,
    adapter_dir: Path,
    examples: list[dict[str, str]],
    *,
    max_steps: int | None = None,
) -> None:
    """Run a short LoRA update on SDFT targets; creates or resumes ``adapter_dir``."""
    if not examples:
        raise ValueError("examples must not be empty")

    adapter_dir.mkdir(parents=True, exist_ok=True)
    ol = cfg.online_learning
    steps = max_steps if max_steps is not None else ol.train_steps
    device = pick_device()
    tokenizer = load_tokenizer(cfg.model)
    ds = _rows_to_dataset(examples)

    base = load_model(cfg.model, device)
    base.config.use_cache = False

    if _adapter_ready(adapter_dir):
        model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=True)
    else:
        peft_config = PeftLoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.alpha,
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )
        sft_config = SFTConfig(
            output_dir=str(adapter_dir),
            max_steps=steps,
            num_train_epochs=1,
            learning_rate=cfg.training.lr,
            per_device_train_batch_size=cfg.training.batch_size,
            gradient_accumulation_steps=cfg.training.grad_accum,
            max_length=cfg.training.max_length,
            warmup_steps=cfg.training.warmup_steps,
            logging_steps=cfg.training.logging_steps,
            save_strategy="no",
            seed=cfg.training.seed,
            report_to=[],
            completion_only_loss=True,
            use_cpu=(device == "cpu"),
            dataset_num_proc=1,
        )
        trainer = SFTTrainer(
            model=base,
            args=sft_config,
            train_dataset=ds,
            peft_config=peft_config,
            processing_class=tokenizer,
        )
        trainer.train()
        trainer.save_model(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        return

    sft_config = SFTConfig(
        output_dir=str(adapter_dir),
        max_steps=steps,
        num_train_epochs=1,
        learning_rate=cfg.training.lr,
        per_device_train_batch_size=cfg.training.batch_size,
        gradient_accumulation_steps=cfg.training.grad_accum,
        max_length=cfg.training.max_length,
        warmup_steps=cfg.training.warmup_steps,
        logging_steps=cfg.training.logging_steps,
        save_strategy="no",
        seed=cfg.training.seed,
        report_to=[],
        completion_only_loss=True,
        use_cpu=(device == "cpu"),
        dataset_num_proc=1,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=ds,
        processing_class=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
