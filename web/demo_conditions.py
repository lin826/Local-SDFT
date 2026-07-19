"""Plain-chat demo conditions for the /perf chat UI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_DEMO_CONDITION = "plain"
SDFT_ALPACA_CONFIG = "configs/lfm25_alpacaeval2_trained.yaml"

# /perf dropdown configs: AlpacaEval-faithful local generation uses no system message.
IGNORE_USER_INSTRUCTION_MESSAGE = (
    "(No system instruction — matches AlpacaEval local generation)"
)
CONFIGS_IGNORE_USER_INSTRUCTION: frozenset[str] = frozenset(
    {DEFAULT_CONFIG, SDFT_ALPACA_CONFIG}
)


@dataclass(frozen=True)
class DemoCondition:
    id: str
    label: str
    description: str
    config_path: str


# Web /perf exposes plain multi-turn chat only (OpenClaw ablations stay CLI-side).
DEMO_CONDITIONS: tuple[DemoCondition, ...] = (
    DemoCondition(
        id="plain",
        label="Plain chat",
        description="Multi-turn Alpaca-style chat; pick base (default.yaml) or SDFT merge via Config.",
        config_path=DEFAULT_CONFIG,
    ),
)

CONDITION_BY_ID: dict[str, DemoCondition] = {c.id: c for c in DEMO_CONDITIONS}


def get_condition(condition_id: str) -> DemoCondition:
    cond = CONDITION_BY_ID.get(condition_id)
    if cond is None:
        raise ValueError(f"unknown demo condition {condition_id!r}")
    return cond


def config_ignores_user_instruction(config_path: str) -> bool:
    """True when /perf chat should omit a custom system message (AE-faithful)."""
    return config_path in CONFIGS_IGNORE_USER_INSTRUCTION


def condition_options() -> list[dict[str, Any]]:
    """Serialize conditions for the template."""
    return [
        {
            "id": c.id,
            "label": c.label,
            "description": c.description,
        }
        for c in DEMO_CONDITIONS
    ]


def build_design_summary(
    *,
    demo_condition: str,
    config_path: str,
    model_path: str,
) -> dict[str, str]:
    """Human-readable run context persisted in benchmark JSON metadata."""
    is_sdft = config_path == SDFT_ALPACA_CONFIG or "sdft-merged" in model_path
    variant = "LFM2.5-230M SDFT merge (AlpacaEval2 recipe)" if is_sdft else "base LFM2.5-230M"
    ignores_instruction = config_ignores_user_instruction(config_path)
    return {
        "purpose": (
            "AlpacaEval-style local plain chat: compare base vs SDFT-merged LFM2.5-230M "
            "on open-ended instructions (e.g. sewing, apple juice)."
        ),
        "demo_condition": demo_condition,
        "config_path": config_path,
        "model_path": model_path,
        "variant": variant,
        "eval_surface": "Local generation in /perf chat; no GPT-4 judge required.",
        "system_instruction": (
            "none (AlpacaEval-faithful; custom system ignored in /perf chat)"
            if ignores_instruction
            else "user-provided in /perf chat"
        ),
    }
