"""Chat inference with an optional LoRA adapter for online-learning previews."""

from __future__ import annotations

from pathlib import Path

import torch
from peft import PeftModel

from sdft.config import Config
from sdft.utils import load_model, load_tokenizer, pick_device


def _adapter_ready(adapter_dir: Path) -> bool:
    return adapter_dir.is_dir() and (adapter_dir / "adapter_config.json").is_file()


@torch.inference_mode()
def generate_preview(
    cfg: Config,
    adapter_dir: Path,
    instruction: str,
    user_input: str = "",
    *,
    max_new_tokens: int | None = None,
    device: str | None = None,
) -> tuple[str, int, int]:
    """Generate one assistant reply; returns (text, input_tokens, output_tokens)."""
    device = device or pick_device()
    max_new_tokens = max_new_tokens or cfg.online_learning.preview_max_new_tokens

    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    base = load_model(cfg.model, device)
    if _adapter_ready(adapter_dir):
        model = PeftModel.from_pretrained(base, str(adapter_dir))
    else:
        model = base
    model.eval()

    parts = [p for p in (instruction.strip(), user_input.strip()) if p]
    user_content = "\n\n".join(parts)
    messages = [{"role": "user", "content": user_content}]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
    enc = enc.to(device)
    input_tokens = int(enc["input_ids"].numel())

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    new_tokens = out[:, enc["input_ids"].shape[1] :]
    output_tokens = int(new_tokens.numel())
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()
    return text, input_tokens, output_tokens
