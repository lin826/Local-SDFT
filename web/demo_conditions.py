"""Plain-chat demo conditions for the /perf chat UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sdft.alpacaeval_ablation import (
    DEFAULT_ABLATION,
    AblationSettings,
    get_ablation_arm,
    list_ablation_arm_names,
    prompt_strategy_display_text,
    prompt_strategy_field_hint,
    prompt_strategy_field_locked,
)
from sdft.config import load_config

DEFAULT_CONFIG = "configs/default.yaml"
DEFAULT_DEMO_CONDITION = "plain"
DEFAULT_PROMPT_STRATEGY = DEFAULT_ABLATION
SDFT_ALPACA_CONFIG = "configs/lfm25_alpacaeval2_trained.yaml"

# Legacy: plain editable system text only when not using AlpacaEval ablation arms.
CONFIGS_IGNORE_USER_INSTRUCTION: frozenset[str] = frozenset(
    {DEFAULT_CONFIG, SDFT_ALPACA_CONFIG}
)

NO_SYSTEM_INSTRUCTION_HINT = (
    "No system instruction (AlpacaEval-faithful); custom text ignored."
)
FIXED_SYSTEM_INSTRUCTION_HINT = (
    "Fixed system instruction from config; custom text ignored."
)

FIXED_SYSTEM_INSTRUCTIONS: dict[str, str] = {}


@dataclass(frozen=True)
class DemoCondition:
    id: str
    label: str
    description: str
    config_path: str


@dataclass(frozen=True)
class PromptStrategyOption:
    id: str
    label: str
    description: str


DEMO_CONDITIONS: tuple[DemoCondition, ...] = (
    DemoCondition(
        id="plain",
        label="Plain chat",
        description="Multi-turn Alpaca-style chat; pick base or SDFT merge via Config.",
        config_path=DEFAULT_CONFIG,
    ),
)

CONDITION_BY_ID: dict[str, DemoCondition] = {c.id: c for c in DEMO_CONDITIONS}


def _hint_for_arm(arm_id: str) -> str:
    return prompt_strategy_field_hint(get_ablation_arm(arm_id))


PROMPT_STRATEGY_OPTIONS: tuple[PromptStrategyOption, ...] = tuple(
    PromptStrategyOption(
        id=arm_id,
        label=arm_id,
        description=_hint_for_arm(arm_id),
    )
    for arm_id in list_ablation_arm_names()
    if arm_id != "SysHelpful"
) + (
    PromptStrategyOption(
        id="SysHelpful",
        label="SysHelpful",
        description="Fixed helpful system prompt (non-AE-faithful ablation).",
    ),
)


def get_condition(condition_id: str) -> DemoCondition:
    cond = CONDITION_BY_ID.get(condition_id)
    if cond is None:
        raise ValueError(f"unknown demo condition {condition_id!r}")
    return cond


def get_prompt_strategy(strategy_id: str) -> AblationSettings:
    return get_ablation_arm(strategy_id)


def config_ignores_user_instruction(config_path: str) -> bool:
    """True when /perf chat should omit a custom system message (AE-faithful configs)."""
    return config_path in CONFIGS_IGNORE_USER_INSTRUCTION


def fixed_system_instruction(config_path: str) -> str:
    """Resolved fixed system prompt from YAML overrides (empty when none)."""
    override = FIXED_SYSTEM_INSTRUCTIONS.get(config_path, "").strip()
    if override:
        return override
    path = Path(config_path)
    if not path.is_file():
        return ""
    cfg = load_config(path)
    if cfg.toolcall.system_prompt:
        return cfg.toolcall.system_prompt.strip()
    raw = yaml.safe_load(path.read_text()) or {}
    top_level = raw.get("system_prompt")
    if isinstance(top_level, str) and top_level.strip():
        return top_level.strip()
    return ""


def instruction_field_locked(config_path: str, prompt_strategy: str = DEFAULT_PROMPT_STRATEGY) -> bool:
    if fixed_system_instruction(config_path):
        return True
    if config_path in CONFIGS_IGNORE_USER_INSTRUCTION:
        return prompt_strategy_field_locked(get_prompt_strategy(prompt_strategy))
    return False


def instruction_field_hint(config_path: str, prompt_strategy: str = DEFAULT_PROMPT_STRATEGY) -> str:
    if fixed_system_instruction(config_path):
        return FIXED_SYSTEM_INSTRUCTION_HINT
    if config_path in CONFIGS_IGNORE_USER_INSTRUCTION:
        return prompt_strategy_field_hint(get_prompt_strategy(prompt_strategy))
    return ""


def instruction_display_text(
    config_path: str,
    *,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    stored_instruction: str = "",
) -> str:
    fixed = fixed_system_instruction(config_path)
    if fixed:
        return fixed
    if config_path in CONFIGS_IGNORE_USER_INSTRUCTION:
        return prompt_strategy_display_text(get_prompt_strategy(prompt_strategy))
    return stored_instruction.strip()


def condition_options() -> list[dict[str, Any]]:
    return [
        {
            "id": c.id,
            "label": c.label,
            "description": c.description,
        }
        for c in DEMO_CONDITIONS
    ]


def prompt_strategy_options() -> list[dict[str, Any]]:
    return [
        {
            "id": o.id,
            "label": o.label,
            "description": o.description,
        }
        for o in PROMPT_STRATEGY_OPTIONS
    ]


def build_design_summary(
    *,
    demo_condition: str,
    config_path: str,
    model_path: str,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
) -> dict[str, str]:
    """Human-readable run context persisted in benchmark JSON metadata."""
    is_sdft = config_path == SDFT_ALPACA_CONFIG or "sdft-merged" in model_path
    variant = "LFM2.5-230M SDFT merge (AlpacaEval2 recipe)" if is_sdft else "base LFM2.5-230M"
    settings = get_prompt_strategy(prompt_strategy)
    if config_path in CONFIGS_IGNORE_USER_INSTRUCTION:
        system_instruction = prompt_strategy_field_hint(settings)
    elif fixed_system_instruction(config_path):
        system_instruction = f"fixed in config: {fixed_system_instruction(config_path)}"
    else:
        system_instruction = "user-provided in /perf chat"
    return {
        "purpose": (
            "AlpacaEval-style local plain chat: compare base vs SDFT-merged LFM2.5-230M "
            "and eval-time prompt ablations (ZS / FS / CoT) on open-ended instructions."
        ),
        "demo_condition": demo_condition,
        "config_path": config_path,
        "model_path": model_path,
        "variant": variant,
        "prompt_strategy": settings.ablation_name,
        "eval_surface": "Local generation in /perf chat; no GPT-4 judge required.",
        "system_instruction": system_instruction,
    }
