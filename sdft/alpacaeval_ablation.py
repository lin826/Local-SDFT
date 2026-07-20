"""AlpacaEval 2 eval-time prompt ablations (ZS, few-shot ICL, CoT).

Few-shot demonstrations are drawn from train-side ``yahma/alpaca-cleaned`` only.
Eval instructions from ``tatsu-lab/alpaca_eval`` are never used as demos (leakage guard).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

from .config import DataConfig
from .data import load_examples

DEFAULT_COT_LINE = "Let's think step by step."
DEFAULT_SYSTEM_HELPFUL = "You are a helpful assistant."

# Full AlpacaEval 2.0 instruction set (``alpaca_eval.json`` on the HF dataset).
ALPACA_EVAL_HF_REPO = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_JSON = "alpaca_eval.json"
ALPACA_EVAL_FULL_N = 805

_WS_RE = re.compile(r"\s+")


def load_alpaca_eval_examples(
    *,
    num_examples: int | None = None,
    json_path: str | Path | None = None,
) -> list[dict[str, str]]:
    """Load AlpacaEval 2.0 instructions as ``{"prompt", "response"}`` pairs.

    Never train on these prompts (leakage). ``response`` is the dataset
    reference (text-davinci-003) for local heuristic scoring only — not for SFT.

    Loads the hub JSON file directly: ``datasets>=4`` no longer runs the legacy
    ``alpaca_eval.py`` script on ``tatsu-lab/alpaca_eval``.
    """
    if json_path is None:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                ALPACA_EVAL_HF_REPO,
                ALPACA_EVAL_JSON,
                repo_type="dataset",
            )
        )
    else:
        path = Path(json_path)

    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"expected a JSON list in {path}, got {type(rows).__name__}")

    examples: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        prompt = str(row.get("instruction", "")).strip()
        response = str(row.get("output", "")).strip()
        if prompt:
            examples.append({"prompt": prompt, "response": response})

    if num_examples is not None:
        if num_examples < 0:
            raise ValueError("num_examples must be non-negative")
        examples = examples[:num_examples]
    return examples


def normalize_instruction(text: str) -> str:
    """Collapse whitespace for leakage comparisons."""
    return _WS_RE.sub(" ", (text or "").strip().lower())


def instruction_hash(text: str) -> str:
    return hashlib.sha256(normalize_instruction(text).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class AblationSettings:
    """Resolved eval-time prompt strategy."""

    ablation_name: str = "ZS"
    few_shot_k: int = 0
    cot: bool = False
    cot_line: str = DEFAULT_COT_LINE
    cot_as_system: bool = False
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ablation_name": self.ablation_name,
            "few_shot_k": self.few_shot_k,
            "cot": self.cot,
            "cot_line": self.cot_line,
            "cot_as_system": self.cot_as_system,
            "system_prompt": self.system_prompt,
        }


ABLATION_ARMS: dict[str, AblationSettings] = {
    "ZS": AblationSettings(ablation_name="ZS"),
    "FS1": AblationSettings(ablation_name="FS1", few_shot_k=1),
    "FS3": AblationSettings(ablation_name="FS3", few_shot_k=3),
    "CoT": AblationSettings(ablation_name="CoT", cot=True),
    "FS1+CoT": AblationSettings(ablation_name="FS1+CoT", few_shot_k=1, cot=True),
    "FS3+CoT": AblationSettings(ablation_name="FS3+CoT", few_shot_k=3, cot=True),
    "SysHelpful": AblationSettings(
        ablation_name="SysHelpful",
        system_prompt=DEFAULT_SYSTEM_HELPFUL,
    ),
}

DEFAULT_ABLATION = "ZS"


def list_ablation_arm_names() -> list[str]:
    return list(ABLATION_ARMS.keys())


def get_ablation_arm(name: str) -> AblationSettings:
    key = (name or DEFAULT_ABLATION).strip()
    arm = ABLATION_ARMS.get(key)
    if arm is None:
        raise ValueError(f"unknown ablation arm {name!r} (valid: {list_ablation_arm_names()})")
    return arm


def resolve_ablation_settings(
    *,
    ablation_name: str | None = None,
    few_shot_k: int | None = None,
    cot: bool | None = None,
    cot_line: str | None = None,
    cot_as_system: bool | None = None,
    system_prompt: str | None = None,
) -> AblationSettings:
    """Merge CLI/YAML overrides onto a named arm (or build a custom arm)."""
    if ablation_name and ablation_name in ABLATION_ARMS and all(
        v is None
        for v in (few_shot_k, cot, cot_line, cot_as_system, system_prompt)
    ):
        return get_ablation_arm(ablation_name)

    base = get_ablation_arm(ablation_name) if ablation_name in ABLATION_ARMS else AblationSettings(
        ablation_name=ablation_name or DEFAULT_ABLATION
    )
    return AblationSettings(
        ablation_name=base.ablation_name if ablation_name is None else ablation_name,
        few_shot_k=base.few_shot_k if few_shot_k is None else few_shot_k,
        cot=base.cot if cot is None else cot,
        cot_line=base.cot_line if cot_line is None else cot_line,
        cot_as_system=base.cot_as_system if cot_as_system is None else cot_as_system,
        system_prompt=base.system_prompt if system_prompt is None else system_prompt,
    )


def apply_cot_to_user(instruction: str, cot_line: str) -> str:
    line = (cot_line or DEFAULT_COT_LINE).strip()
    text = instruction.strip()
    if not line:
        return text
    if text.endswith(line):
        return text
    return f"{text}\n\n{line}" if text else line


def build_eval_messages(
    instruction: str,
    settings: AblationSettings,
    *,
    few_shots: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Build chat messages for one AlpacaEval instruction."""
    messages: list[dict[str, str]] = []

    system_parts: list[str] = []
    if settings.system_prompt:
        system_parts.append(settings.system_prompt.strip())
    if settings.cot and settings.cot_as_system:
        system_parts.append(settings.cot_line.strip())
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for shot in few_shots or []:
        messages.append({"role": "user", "content": shot["prompt"]})
        messages.append({"role": "assistant", "content": shot["response"]})

    user_text = instruction.strip()
    if settings.cot and not settings.cot_as_system:
        user_text = apply_cot_to_user(user_text, settings.cot_line)
    messages.append({"role": "user", "content": user_text})
    return messages


def build_perf_chat_messages(
    settings: AblationSettings,
    history: list[dict[str, str]],
    user_message: str,
    *,
    few_shot_seed: int = 0,
) -> list[dict[str, str]]:
    """Multi-turn /perf chat with optional ICL prefix and CoT on user turns."""
    few_shots = web_few_shots(settings.few_shot_k, few_shot_seed) if settings.few_shot_k else []
    messages: list[dict[str, str]] = []

    system_parts: list[str] = []
    if settings.system_prompt:
        system_parts.append(settings.system_prompt.strip())
    if settings.cot and settings.cot_as_system:
        system_parts.append(settings.cot_line.strip())
    if system_parts:
        messages.append({"role": "system", "content": "\n\n".join(system_parts)})

    for shot in few_shots:
        messages.append({"role": "user", "content": shot["prompt"]})
        messages.append({"role": "assistant", "content": shot["response"]})

    for turn in history:
        role = turn["role"]
        if role == "system":
            continue
        content = turn["content"]
        if role == "user" and settings.cot and not settings.cot_as_system:
            content = apply_cot_to_user(content, settings.cot_line)
        messages.append({"role": role, "content": content})

    user_text = user_message.strip()
    if settings.cot and not settings.cot_as_system:
        user_text = apply_cot_to_user(user_text, settings.cot_line)
    messages.append({"role": "user", "content": user_text})
    return messages


def verify_no_eval_leakage(
    few_shots: list[dict[str, str]],
    eval_instructions: list[str],
) -> None:
    """Raise if any demo prompt matches a held-out eval instruction."""
    eval_norm = {normalize_instruction(x) for x in eval_instructions if x.strip()}
    eval_hashes = {instruction_hash(x) for x in eval_instructions if x.strip()}
    for shot in few_shots:
        prompt = shot.get("prompt", "")
        norm = normalize_instruction(prompt)
        if norm in eval_norm:
            raise ValueError(
                "few-shot leakage: demo prompt matches an AlpacaEval instruction "
                f"(hash={instruction_hash(prompt)})"
            )
        if instruction_hash(prompt) in eval_hashes:
            raise ValueError(
                "few-shot leakage: demo prompt hash matches an AlpacaEval instruction"
            )


def filter_train_pool_for_eval(
    train_examples: list[dict[str, str]],
    eval_instructions: list[str],
) -> list[dict[str, str]]:
    eval_norm = {normalize_instruction(x) for x in eval_instructions if x.strip()}
    kept: list[dict[str, str]] = []
    for ex in train_examples:
        norm = normalize_instruction(ex.get("prompt", ""))
        if norm and norm not in eval_norm:
            kept.append(ex)
    return kept


def select_fixed_few_shots(
    pool: list[dict[str, str]],
    k: int,
    seed: int,
) -> list[dict[str, str]]:
    if k <= 0 or not pool:
        return []
    rng = random.Random(seed)
    idxs = rng.sample(range(len(pool)), min(k, len(pool)))
    return [pool[i] for i in sorted(idxs)]


@dataclass
class FewShotContext:
    pool_size: int
    demos: list[dict[str, str]] = field(default_factory=list)


def prepare_few_shot_context(
    data_cfg: DataConfig,
    eval_instructions: list[str],
    *,
    k: int,
    seed: int = 0,
) -> FewShotContext:
    if k <= 0:
        return FewShotContext(pool_size=0, demos=[])

    train_examples = load_examples(data_cfg)
    pool = filter_train_pool_for_eval(train_examples, eval_instructions)
    demos = select_fixed_few_shots(pool, k, seed)
    verify_no_eval_leakage(demos, eval_instructions)
    return FewShotContext(pool_size=len(pool), demos=demos)


@lru_cache(maxsize=16)
def cached_web_few_shots(k: int, seed: int = 0) -> tuple[dict[str, str], ...]:
    """Fixed ICL demos for /perf (no eval set loaded — hash guard uses empty eval list)."""
    data_cfg = DataConfig(num_examples=512, seed=seed)
    ctx = prepare_few_shot_context(data_cfg, eval_instructions=[], k=k, seed=seed)
    return tuple(ctx.demos)


def web_few_shots(k: int, seed: int = 0) -> list[dict[str, str]]:
    return list(cached_web_few_shots(k, seed))


def prompt_strategy_display_text(settings: AblationSettings) -> str:
    """Read-only /perf instruction field content for an ablation arm."""
    if settings.system_prompt:
        return settings.system_prompt.strip()
    if settings.cot and not settings.cot_as_system:
        return settings.cot_line.strip()
    return ""


def prompt_strategy_field_hint(settings: AblationSettings) -> str:
    if settings.system_prompt:
        return "Fixed system instruction for this ablation arm; custom text ignored."
    if settings.few_shot_k > 0 and settings.cot:
        return (
            f"AlpacaEval ablation {settings.ablation_name}: "
            f"{settings.few_shot_k} fixed ICL demo(s) from alpaca-cleaned; "
            f"CoT cue appended to each user turn (not a system message)."
        )
    if settings.few_shot_k > 0:
        return (
            f"AlpacaEval ablation {settings.ablation_name}: "
            f"{settings.few_shot_k} fixed ICL demo(s) from alpaca-cleaned; no system message."
        )
    if settings.cot:
        return (
            f"AlpacaEval ablation {settings.ablation_name}: "
            "CoT cue appended to each user message (AE-faithful — no system message)."
        )
    return "No system instruction (AlpacaEval-faithful ZS); custom text ignored."


def prompt_strategy_field_locked(settings: AblationSettings) -> bool:
    return True
