"""Device and model-loading helpers shared by the pipeline stages."""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from .config import ModelConfig

_T = TypeVar("_T")

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


def model_device(model: PreTrainedModel | torch.nn.Module) -> torch.device:
    """Device where model inputs should land (first parameter).

    With HuggingFace ``device_map=\"auto\"``, embeddings usually sit on the first
    device; send tokenized tensors there rather than assuming a single ``.to``.
    """
    return next(model.parameters()).device


def to_model_device(batch: _T, model: PreTrainedModel | torch.nn.Module) -> _T:
    """Move tokenized tensors onto the model's device.

    Tokenizers are not ``nn.Module``s — there is no reliable ``tokenizer.to(\"cuda\")``.
    Callers that want \"put the encoding on CUDA\" should use this helper (or
    ``batch.to(model_device(model))``) instead.
    """
    device = model_device(model)
    to_fn = getattr(batch, "to", None)
    if callable(to_fn):
        return to_fn(device)  # type: ignore[no-any-return]
    if isinstance(batch, Mapping):
        return {  # type: ignore[return-value]
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()
        }
    raise TypeError(f"cannot move batch of type {type(batch).__name__} to model device")


def load_model(cfg: ModelConfig, device: str) -> PreTrainedModel:
    """Load a causal LM.

    Uses HuggingFace ``device_map=\"auto\"`` on CUDA/CPU (no extra ``.to(device)``,
    which would fight Accelerate placement). On MPS, ``device_map=\"auto\"`` is
    unreliable, so we fall back to a plain load + ``.to(\"mps\")``.
    """
    kwargs: dict[str, Any] = {"dtype": _DTYPES[cfg.dtype]}
    if cfg.attn_implementation:
        kwargs["attn_implementation"] = cfg.attn_implementation

    if device == "mps":
        model = AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
        return model.to(device)

    kwargs["device_map"] = "auto"
    return AutoModelForCausalLM.from_pretrained(cfg.name, **kwargs)
