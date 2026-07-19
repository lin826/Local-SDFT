"""Dataset loading and prompt/response formatting."""

from __future__ import annotations

import random

from datasets import load_dataset

from .config import DataConfig


def _join_prompt(fields: list[str], row: dict) -> str:
    parts = [str(row[f]).strip() for f in fields if row.get(f) and str(row[f]).strip()]
    return "\n\n".join(parts)


def load_examples(cfg: DataConfig) -> list[dict]:
    """Return [{"prompt", "response"}] pairs sampled from the configured dataset."""
    ds = load_dataset(cfg.dataset, data_files=cfg.data_files, split=cfg.split)
    ds = ds.shuffle(seed=cfg.seed)
    examples = []
    for row in ds.select(range(min(cfg.num_examples, len(ds)))):
        prompt = _join_prompt(cfg.prompt_fields, row)
        response = str(row[cfg.response_field]).strip()
        if prompt and response:
            examples.append({"prompt": prompt, "response": response})
    return examples


def sample_fewshots(
    examples: list[dict], exclude_idx: int, num_shots: int, rng: random.Random
) -> list[dict]:
    """In-context demonstrations: other examples' gold (prompt, response) pairs."""
    pool = list(range(len(examples)))
    pool.remove(exclude_idx)
    idxs = rng.sample(pool, min(num_shots, len(pool)))
    return [examples[i] for i in idxs]


def build_teacher_messages(fewshots: list[dict], prompt: str) -> list[dict]:
    """Multi-turn conversation: k demonstration pairs, then the target prompt."""
    messages = []
    for shot in fewshots:
        messages.append({"role": "user", "content": shot["prompt"]})
        messages.append({"role": "assistant", "content": shot["response"]})
    messages.append({"role": "user", "content": prompt})
    return messages
