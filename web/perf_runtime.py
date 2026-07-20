"""Perf chat inference helpers (sync + SSE event stream)."""

from __future__ import annotations

import json
from typing import Any, Iterator
from urllib.parse import quote

from sdft.config import load_config
from sdft.records import iter_measure_chat, measure_chat, persist_performance_result, performance_result_path
from sdft.records.latency import LatencyPhases
from sdft.records.store import save_performance_result
from web.chat_context import build_chat_messages
from web.demo_conditions import build_design_summary, get_condition, get_prompt_strategy
from web.perf_models import resolve_perf_config


def load_perf_chat_cfg(selection: Any) -> Any:
    cfg = load_config(selection.yaml_config_path)
    if selection.base_model:
        cfg.model.name = selection.base_model
    return cfg


def attach_run_metadata(
    result: Any,
    *,
    demo_condition: str,
    config_path: str,
    model_path: str,
    prompt_strategy: str,
    online_session_id: str | None = None,
    adapter_dir: str | None = None,
) -> None:
    result.config_path = config_path
    result.metadata = result.metadata or {}
    result.metadata["config_path"] = config_path
    result.metadata["demo_condition"] = demo_condition
    result.metadata["prompt_strategy"] = prompt_strategy
    result.metadata["model_path"] = model_path
    if online_session_id:
        result.metadata["online_session_id"] = online_session_id
    if adapter_dir:
        result.metadata["adapter_dir"] = adapter_dir
    result.metadata["design_summary"] = build_design_summary(
        demo_condition=demo_condition,
        config_path=config_path,
        model_path=model_path,
        prompt_strategy=prompt_strategy,
        online_session_id=online_session_id,
        adapter_dir=adapter_dir,
    )


def format_sse(event: str, data: dict[str, Any]) -> str:
    """Encode one Server-Sent Event (``event`` + JSON ``data`` lines)."""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def prepare_chat_inputs(
    *,
    config_path: str,
    condition_id: str,
    prompt_strategy: str,
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
) -> tuple[Any, str, list[dict[str, str]]]:
    """Validate inputs and build model messages (no model load)."""
    condition = get_condition(condition_id)
    selection = resolve_perf_config(config_path)
    get_prompt_strategy(prompt_strategy)
    messages = build_chat_messages(
        instruction,
        history,
        user_message,
        config_path=selection.config_path,
        prompt_strategy=prompt_strategy,
    )
    return selection, condition.id, messages


def finalize_chat_result(
    result: Any,
    *,
    phases: LatencyPhases,
    demo_condition: str,
    selection: Any,
    prompt_strategy: str,
) -> Any:
    """Attach design metadata, persist once, rewrite latency_phases with persist span."""
    model_path = selection.model_path
    adapter_dir = str(selection.adapter_dir) if selection.adapter_dir else None
    attach_run_metadata(
        result,
        demo_condition=demo_condition,
        config_path=selection.config_path,
        model_path=model_path,
        prompt_strategy=prompt_strategy,
        online_session_id=selection.online_session_id,
        adapter_dir=adapter_dir,
    )
    with phases.span("persist"):
        result.metadata["latency_phases"] = phases.to_list()
        persist_performance_result(result)
    result.metadata["latency_phases"] = phases.to_list()
    save_performance_result(performance_result_path(result.id), result)
    return result


def run_chat_inference(
    *,
    config_path: str,
    condition_id: str,
    prompt_strategy: str,
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
) -> Any:
    """Sync plain multi-turn chat inference."""
    phases = LatencyPhases()
    selection, demo_condition, messages = prepare_chat_inputs(
        config_path=config_path,
        condition_id=condition_id,
        prompt_strategy=prompt_strategy,
        instruction=instruction,
        history=history,
        user_message=user_message,
    )
    with phases.span("config_load"):
        cfg = load_perf_chat_cfg(selection)
    result = measure_chat(
        cfg,
        messages,
        latency_phases=phases,
        adapter_dir=selection.adapter_dir,
        model_name=selection.model_path,
    )
    return finalize_chat_result(
        result,
        phases=phases,
        demo_condition=demo_condition,
        selection=selection,
        prompt_strategy=prompt_strategy,
    )


def iter_chat_inference_events(
    *,
    config_path: str,
    condition_id: str,
    prompt_strategy: str,
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
) -> Iterator[tuple[str, Any]]:
    """Yield ``(\"token\", text)`` then ``(\"result\", PerformanceResult)`` for SSE."""
    phases = LatencyPhases()
    selection, demo_condition, messages = prepare_chat_inputs(
        config_path=config_path,
        condition_id=condition_id,
        prompt_strategy=prompt_strategy,
        instruction=instruction,
        history=history,
        user_message=user_message,
    )
    with phases.span("config_load"):
        cfg = load_perf_chat_cfg(selection)
    for kind, payload in iter_measure_chat(
        cfg,
        messages,
        latency_phases=phases,
        adapter_dir=selection.adapter_dir,
        model_name=selection.model_path,
    ):
        if kind == "token":
            yield ("token", payload)
        elif kind == "result":
            result = finalize_chat_result(
                payload,
                phases=phases,
                demo_condition=demo_condition,
                selection=selection,
                prompt_strategy=prompt_strategy,
            )
            yield ("result", result)
        else:
            raise ValueError(f"unexpected stream event {kind!r}")


def continue_url(result_id: str, config_path: str, prompt_strategy: str) -> str:
    cfg_q = quote(config_path, safe="")
    strat_q = quote(prompt_strategy, safe="")
    return f"/perf?continue={result_id}&config_path={cfg_q}&prompt_strategy={strat_q}&sent=1"
