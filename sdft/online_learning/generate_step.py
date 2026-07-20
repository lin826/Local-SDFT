"""SDFT teacher generation for one online-learning turn."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import torch

from sdft.config import Config
from sdft.data import build_teacher_messages
from sdft.utils import load_model, load_tokenizer, pick_device, to_model_device

if TYPE_CHECKING:
    from .schema import OnlineTurn


def row_to_prompt(row: dict[str, str]) -> str:
    parts = [p for p in (row.get("instruction", ""), row.get("input", "")) if p.strip()]
    return "\n\n".join(parts)


def turns_to_fewshot_examples(turns: list[OnlineTurn]) -> list[dict[str, str]]:
    """Prior turns as in-context demos; responses are model-generated SDFT targets."""
    examples: list[dict[str, str]] = []
    for turn in turns:
        response = (turn.sdft_response or "").strip()
        if not response:
            continue
        examples.append(
            {
                "prompt": row_to_prompt({"instruction": turn.instruction, "input": turn.input}),
                "response": response,
            }
        )
    return examples


@torch.inference_mode()
def generate_sdft_response(
    cfg: Config,
    *,
    instruction: str,
    user_input: str = "",
    fewshot_examples: list[dict[str, str]],
    device: str | None = None,
) -> tuple[str, int, int]:
    """Run one SDFT teacher rewrite; returns (sdft_response, input_tokens, output_tokens)."""
    device = device or pick_device()
    gen = cfg.generation
    do_sample = gen.temperature > 0
    rng = random.Random(cfg.data.seed)

    target_prompt = row_to_prompt({"instruction": instruction, "input": user_input})
    if fewshot_examples:
        pool = list(range(len(fewshot_examples)))
        idxs = rng.sample(pool, min(gen.num_shots, len(pool)))
        fewshots = [fewshot_examples[i] for i in idxs]
    else:
        fewshots = []

    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = load_model(cfg.model, device)
    model.eval()

    prompt_text = tokenizer.apply_chat_template(
        build_teacher_messages(fewshots, target_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc = to_model_device(enc, model)
    input_tokens = int(enc["input_ids"].numel())

    out = model.generate(
        **enc,
        max_new_tokens=gen.max_new_tokens,
        do_sample=do_sample,
        temperature=gen.temperature if do_sample else None,
        top_p=gen.top_p if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    output_tokens = int(new_tokens.numel())
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    if len(text) < gen.min_response_chars:
        raise ValueError("SDFT generation produced an empty or degenerate response")
    return text, input_tokens, output_tokens
