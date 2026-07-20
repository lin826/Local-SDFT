"""Chat history / instruction UI helpers for the /perf and /data surfaces."""

from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException

from sdft.alpacaeval_ablation import build_perf_chat_messages
from sdft.records import load_performance_result, performance_result_path
from web.demo_conditions import (
    DEFAULT_CONFIG,
    DEFAULT_DEMO_CONDITION,
    DEFAULT_PROMPT_STRATEGY,
    fixed_system_instruction,
    get_prompt_strategy,
    instruction_display_text,
    instruction_field_hint,
    instruction_field_locked,
)
from web.perf_models import resolve_perf_config

ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}
DEFAULT_INSTRUCTION = "Answer helpfully and directly in plain text."
CONFIG_OPTIONS = [
    DEFAULT_CONFIG,
    "configs/lfm25_alpacaeval2_trained.yaml",
]


def include_message_for_display(m: dict) -> bool:
    if not isinstance(m, dict):
        return False
    role = m.get("role")
    if role not in ALLOWED_CHAT_ROLES:
        return False
    if role == "assistant":
        return True
    return bool(str(m.get("content", "")).strip())


def parse_messages_json(raw: str) -> list[dict[str, str]]:
    """Parse request-carried chat history (OpenAI-style role/content list)."""
    text = (raw or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"invalid messages_json: {exc}") from exc
    if not isinstance(data, list):
        raise HTTPException(status_code=400, detail="messages_json must be a JSON array")
    cleaned: list[dict[str, str]] = []
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise HTTPException(status_code=400, detail=f"messages_json[{i}] must be an object")
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role not in ALLOWED_CHAT_ROLES:
            raise HTTPException(
                status_code=400,
                detail=f"messages_json[{i}].role must be system|user|assistant",
            )
        if not content:
            continue
        cleaned.append({"role": role, "content": content})
    return cleaned


def history_without_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [m for m in messages if m["role"] != "system"]


def system_from_messages(messages: list[dict[str, str]]) -> str:
    return next((m["content"] for m in messages if m["role"] == "system"), "")


def instruction_ui_context(
    config_path: str,
    *,
    prompt_strategy: str = DEFAULT_PROMPT_STRATEGY,
    stored_instruction: str = "",
) -> dict[str, Any]:
    """Instruction textarea state for /perf (display text + whether user input is ignored)."""
    locked = instruction_field_locked(config_path, prompt_strategy)
    if locked:
        return {
            "instruction": instruction_display_text(
                config_path,
                prompt_strategy=prompt_strategy,
                stored_instruction=stored_instruction,
            ),
            "instruction_ignored": True,
            "instruction_hint": instruction_field_hint(config_path, prompt_strategy),
        }
    text = stored_instruction.strip() or DEFAULT_INSTRUCTION
    return {
        "instruction": text,
        "instruction_ignored": False,
        "instruction_hint": "",
    }


def build_chat_messages(
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
    *,
    config_path: str,
    prompt_strategy: str,
) -> list[dict[str, str]]:
    """Assemble model input for /perf chat."""
    ablation_config = config_path
    try:
        ablation_config = resolve_perf_config(config_path).ablation_config_path
    except ValueError:
        ablation_config = DEFAULT_CONFIG
    fixed = fixed_system_instruction(ablation_config)
    if fixed or ablation_config not in CONFIG_OPTIONS:
        messages: list[dict[str, str]] = []
        instr = (fixed or instruction).strip()
        if instr:
            messages.append({"role": "system", "content": instr})
        for m in history:
            role = m["role"]
            if role == "system":
                continue
            messages.append({"role": role, "content": m["content"]})
        messages.append({"role": "user", "content": user_message.strip()})
        return messages
    return build_perf_chat_messages(
        get_prompt_strategy(prompt_strategy),
        history,
        user_message,
    )


def chat_context_from_result(
    result: Any,
    *,
    instruction_fallback: str = "",
) -> dict[str, Any]:
    """Build chat UI context from a just-finished PerformanceResult."""
    meta = result.metadata or {}
    messages = meta.get("messages")
    typed: list[dict[str, str]] = []
    if isinstance(messages, list):
        typed = [
            {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
            for m in messages
            if include_message_for_display(m)
        ]
    config_path = str(result.config_path or meta.get("config_path") or DEFAULT_CONFIG)
    stored_instruction = system_from_messages(typed) or instruction_fallback
    instruction_ctx = instruction_ui_context(
        config_path,
        prompt_strategy=str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY),
        stored_instruction=stored_instruction,
    )
    history = history_without_system(typed)
    return {
        **instruction_ctx,
        "messages": history,
        "messages_json": json.dumps(history, ensure_ascii=False),
        "last_run_id": result.id,
        "composer_prefill": "",
        "demo_condition": str(meta.get("demo_condition") or DEFAULT_DEMO_CONDITION),
        "prompt_strategy": str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY),
        "config_path": config_path,
        "metrics": result.metrics,
        "max_new_tokens": meta.get("max_new_tokens"),
        "output_tokens_total": getattr(result.metrics, "output_tokens_total", None),
        "latency_phases": meta.get("latency_phases") or [],
    }


def chat_context_from_continue(run_id: str | None) -> dict[str, Any]:
    """Load sticky instruction + history from a prior chat run for ?continue=."""
    empty_instruction = instruction_ui_context(DEFAULT_CONFIG)
    empty = {
        **empty_instruction,
        "messages": [],
        "messages_json": "[]",
        "last_run_id": None,
        "demo_condition": DEFAULT_DEMO_CONDITION,
        "prompt_strategy": DEFAULT_PROMPT_STRATEGY,
        "config_path": DEFAULT_CONFIG,
        "metrics": None,
        "max_new_tokens": None,
        "output_tokens_total": None,
        "latency_phases": [],
    }
    if not run_id:
        return empty
    path = performance_result_path(run_id)
    if not path.is_file():
        return empty
    result = load_performance_result(path)
    meta = result.metadata or {}
    demo_condition = str(meta.get("demo_condition") or DEFAULT_DEMO_CONDITION)
    config_path = str(result.config_path or meta.get("config_path") or DEFAULT_CONFIG)
    messages = meta.get("messages")
    if not isinstance(messages, list) or not messages:
        examples = meta.get("examples") or []
        if examples:
            ex = examples[0]
            stored_instruction = str(ex.get("instruction") or "")
            user_text = str(ex.get("input") or "").strip() or stored_instruction
            history = [{"role": "user", "content": user_text}]
            if ex.get("output"):
                history.append({"role": "assistant", "content": str(ex["output"])})
            instruction_ctx = instruction_ui_context(
                config_path,
                prompt_strategy=str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY),
                stored_instruction=stored_instruction if ex.get("input") else "",
            )
            return {
                **instruction_ctx,
                "messages": history,
                "messages_json": json.dumps(history, ensure_ascii=False),
                "last_run_id": run_id,
                "demo_condition": demo_condition,
                "prompt_strategy": str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY),
                "config_path": config_path,
                "metrics": result.metrics,
                "max_new_tokens": meta.get("max_new_tokens"),
                "output_tokens_total": result.metrics.output_tokens_total,
                "latency_phases": meta.get("latency_phases") or [],
            }
        return {**empty, "last_run_id": run_id}

    typed = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if include_message_for_display(m)
    ]
    stored_instruction = system_from_messages(typed)
    prompt_strategy = str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY)
    instruction_ctx = instruction_ui_context(
        config_path,
        prompt_strategy=prompt_strategy,
        stored_instruction=stored_instruction,
    )
    history = history_without_system(typed)
    return {
        **instruction_ctx,
        "messages": history,
        "messages_json": json.dumps(history, ensure_ascii=False),
        "last_run_id": run_id,
        "demo_condition": demo_condition,
        "prompt_strategy": prompt_strategy,
        "config_path": config_path,
        "metrics": result.metrics,
        "max_new_tokens": meta.get("max_new_tokens"),
        "output_tokens_total": result.metrics.output_tokens_total,
        "latency_phases": meta.get("latency_phases") or [],
    }
