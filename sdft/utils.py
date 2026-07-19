"""Device and model-loading helpers shared by the pipeline stages."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def pick_device() -> str:
    """Prefer MPS (Apple Silicon), then CUDA, then CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_tokenizer(cfg: ModelConfig) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(cfg: ModelConfig, device: str) -> PreTrainedModel:
    kwargs: dict = {"dtype": _DTYPES[cfg.dtype]}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation
    model = AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
    return model.to(device)
