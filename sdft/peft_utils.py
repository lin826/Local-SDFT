"""Shared PEFT / LoRA helper utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def adapter_ready(adapter_dir: Path | str) -> bool:
    """True when a PEFT LoRA adapter has been saved under ``adapter_dir``."""
    path = Path(adapter_dir)
    return path.is_dir() and (path / "adapter_config.json").is_file()


def peft_adapter_metadata(model: Any) -> dict[str, Any]:
    """Summarize whether a chat model has an active PEFT LoRA adapter."""
    if not hasattr(model, "peft_config"):
        return {"adapter_loaded": False, "peft_active_adapters": []}
    active = getattr(model, "active_adapters", None)
    if callable(active):
        active = active()
    adapters = list(active) if active else list(model.peft_config.keys())
    return {"adapter_loaded": True, "peft_active_adapters": adapters}


def load_chat_model(cfg: Any, device: str, *, adapter_dir: Path | str | None = None):
    """Load base causal LM, optionally wrapped with a PEFT LoRA adapter."""
    from .utils import load_model

    base = load_model(cfg.model, device)
    if adapter_dir is not None and adapter_ready(adapter_dir):
        from peft import PeftModel

        return PeftModel.from_pretrained(base, str(adapter_dir))
    return base
