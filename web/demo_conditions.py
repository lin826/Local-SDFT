"""OpenClaw ablation-style demo conditions for the /perf chat UI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sdft.records.paths import project_root
from sdft.toolcall.format import DEFAULT_COT_LINE

DEFAULT_OPENCLAW_CONFIG = "configs/openclaw_demo_eval.yaml"
DEFAULT_MERGED_REL = "outputs/openclaw-tooluse-merged"


@dataclass(frozen=True)
class DemoCondition:
    id: str
    label: str
    description: str
    toolcall: bool
    few_shot_k: int
    cot_line: str | None
    sdft: bool
    config_path: str

    @property
    def requires_merged_checkpoint(self) -> bool:
        return self.sdft


def merged_checkpoint_path(root: Path | None = None) -> Path:
    return project_root(root) / DEFAULT_MERGED_REL


def merged_checkpoint_available(root: Path | None = None) -> bool:
    path = merged_checkpoint_path(root)
    return path.is_dir() and any(path.iterdir())


# Aligned with scripts/run_openclaw_ablation.py CONDITIONS (+ plain chat for geek jokes).
DEMO_CONDITIONS: tuple[DemoCondition, ...] = (
    DemoCondition(
        id="plain",
        label="Plain chat",
        description="Multi-turn chat without tool loop (geek jokes / Alpaca-style).",
        toolcall=False,
        few_shot_k=0,
        cot_line=None,
        sdft=False,
        config_path="configs/default.yaml",
    ),
    DemoCondition(
        id="ZS",
        label="ZS (zero-shot tools)",
        description="Tool-use system prompt only; no one-shot demo.",
        toolcall=True,
        few_shot_k=0,
        cot_line=None,
        sdft=False,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="OS",
        label="OS (one-shot demo)",
        description="Prepend one canned tool-use demonstration.",
        toolcall=True,
        few_shot_k=1,
        cot_line=None,
        sdft=False,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="CoT",
        label="CoT (cue only)",
        description="One-line chain-of-thought cue in the system prompt.",
        toolcall=True,
        few_shot_k=0,
        cot_line=DEFAULT_COT_LINE,
        sdft=False,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="OS+CoT",
        label="OS + CoT",
        description="One-shot demo plus CoT cue.",
        toolcall=True,
        few_shot_k=1,
        cot_line=DEFAULT_COT_LINE,
        sdft=False,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="SDFT-ZS",
        label="SDFT-ZS",
        description="Merged SDFT checkpoint; zero-shot tools.",
        toolcall=True,
        few_shot_k=0,
        cot_line=None,
        sdft=True,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="SDFT+OS",
        label="SDFT + OS",
        description="Merged SDFT checkpoint with one-shot demo.",
        toolcall=True,
        few_shot_k=1,
        cot_line=None,
        sdft=True,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
    DemoCondition(
        id="SDFT+OS+CoT",
        label="SDFT + OS + CoT",
        description="Merged SDFT checkpoint with one-shot demo and CoT cue.",
        toolcall=True,
        few_shot_k=1,
        cot_line=DEFAULT_COT_LINE,
        sdft=True,
        config_path=DEFAULT_OPENCLAW_CONFIG,
    ),
)

CONDITION_BY_ID: dict[str, DemoCondition] = {c.id: c for c in DEMO_CONDITIONS}


def get_condition(condition_id: str) -> DemoCondition:
    cond = CONDITION_BY_ID.get(condition_id)
    if cond is None:
        raise ValueError(f"unknown demo condition {condition_id!r}")
    return cond


def condition_options(root: Path | None = None) -> list[dict[str, Any]]:
    """Serialize conditions for the template (includes SDFT availability)."""
    merged_ok = merged_checkpoint_available(root)
    merged_path = str(merged_checkpoint_path(root))
    out: list[dict[str, Any]] = []
    for c in DEMO_CONDITIONS:
        disabled = c.requires_merged_checkpoint and not merged_ok
        out.append(
            {
                "id": c.id,
                "label": c.label,
                "description": c.description,
                "toolcall": c.toolcall,
                "sdft": c.sdft,
                "disabled": disabled,
                "disabled_reason": (
                    f"Merged checkpoint not found at {merged_path}. "
                    "Run train + merge (see docs/openclaw-tooluse-sdft.md)."
                    if disabled
                    else ""
                ),
            }
        )
    return out


def resolve_model_name(condition: DemoCondition, root: Path | None = None) -> str:
    if condition.sdft:
        return str(merged_checkpoint_path(root))
    return "LiquidAI/LFM2.5-230M"
