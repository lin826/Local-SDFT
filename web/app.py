"""Local-SDFT web UI for data collection and performance testing.

Uses the shared contract in ``sdft.records`` only — see ``docs/shared-contract.md``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from urllib.parse import quote

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sdft.alpacaeval_ablation import build_perf_chat_messages
from sdft.config import load_config
from sdft.online_learning import create_session, load_session, run_online_turn
from sdft.online_learning.session import list_sessions
from sdft.records import (
    collect_record,
    collected_records_path,
    export_collected_for_training,
    list_performance_results,
    load_collected_records,
    load_performance_index,
    load_performance_result,
    measure_chat,
    performance_dir,
    performance_index_path,
    performance_result_path,
    persist_performance_result,
    run_benchmark,
)
from sdft.records.benchmark import LatencyPhases
from sdft.records.store import save_performance_result

from web.demo_conditions import (
    DEFAULT_CONFIG,
    DEFAULT_DEMO_CONDITION,
    DEFAULT_PROMPT_STRATEGY,
    build_design_summary,
    config_ignores_user_instruction,
    fixed_system_instruction,
    get_condition,
    get_prompt_strategy,
    instruction_display_text,
    instruction_field_hint,
    instruction_field_locked,
    prompt_strategy_options,
)
from web.transcript_parse import (
    display_assistant_content,
    highlight_boxed,
    parse_message_content,
)

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

templates.env.filters["parse_transcript"] = lambda content, role="assistant": [
    s.to_dict() for s in parse_message_content(role, content or "")
]
templates.env.filters["highlight_boxed"] = highlight_boxed
templates.env.filters["display_assistant"] = display_assistant_content

app = FastAPI(title="Local-SDFT", description="Data collection and performance testing")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

CONFIG_OPTIONS = [
    DEFAULT_CONFIG,
    "configs/lfm25_alpacaeval2_trained.yaml",
]
ONLINE_CONFIG_OPTIONS = [
    "configs/online_learning.yaml",
    DEFAULT_CONFIG,
]
DEFAULT_ONLINE_CONFIG = ONLINE_CONFIG_OPTIONS[0]
DEFAULT_INSTRUCTION = "Answer helpfully and directly in plain text."
ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}


def _include_message_for_display(m: dict) -> bool:
    if not isinstance(m, dict):
        return False
    role = m.get("role")
    if role not in ALLOWED_CHAT_ROLES:
        return False
    if role == "assistant":
        return True
    return bool(str(m.get("content", "")).strip())


def _index_entries(limit: int = 50) -> list[dict]:
    rows = load_performance_index(performance_index_path())
    return list(reversed(rows[-limit:]))


def _load_results(limit: int = 20):
    results = list_performance_results(performance_dir())
    return list(reversed(results[-limit:]))


def _parse_messages_json(raw: str) -> list[dict[str, str]]:
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


def _history_without_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [m for m in messages if m["role"] != "system"]


def _system_from_messages(messages: list[dict[str, str]]) -> str:
    return next((m["content"] for m in messages if m["role"] == "system"), "")


def _instruction_ui_context(
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


def _build_chat_messages(
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
    *,
    config_path: str,
    prompt_strategy: str,
) -> list[dict[str, str]]:
    """Assemble model input for /perf chat."""
    fixed = fixed_system_instruction(config_path)
    if fixed or config_path not in CONFIG_OPTIONS:
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


def _attach_run_metadata(
    result: Any,
    *,
    demo_condition: str,
    config_path: str,
    model_path: str,
    prompt_strategy: str,
) -> None:
    result.config_path = config_path
    result.metadata = result.metadata or {}
    result.metadata["config_path"] = config_path
    result.metadata["demo_condition"] = demo_condition
    result.metadata["prompt_strategy"] = prompt_strategy
    result.metadata["model_path"] = model_path
    result.metadata["design_summary"] = build_design_summary(
        demo_condition=demo_condition,
        config_path=config_path,
        model_path=model_path,
        prompt_strategy=prompt_strategy,
    )


def _chat_context_from_result(
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
            if _include_message_for_display(m)
        ]
    config_path = str(result.config_path or meta.get("config_path") or DEFAULT_CONFIG)
    stored_instruction = _system_from_messages(typed) or instruction_fallback
    instruction_ctx = _instruction_ui_context(
        config_path,
        prompt_strategy=str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY),
        stored_instruction=stored_instruction,
    )
    history = _history_without_system(typed)
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


def _chat_context_from_continue(run_id: str | None) -> dict[str, Any]:
    """Load sticky instruction + history from a prior chat run for ?continue=."""
    empty_instruction = _instruction_ui_context(DEFAULT_CONFIG)
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
        # Fall back to single-turn examples for re-run compatibility.
        examples = meta.get("examples") or []
        if examples:
            ex = examples[0]
            stored_instruction = str(ex.get("instruction") or "")
            user_text = str(ex.get("input") or "").strip() or stored_instruction
            history = [{"role": "user", "content": user_text}]
            if ex.get("output"):
                history.append({"role": "assistant", "content": str(ex["output"])})
            instruction_ctx = _instruction_ui_context(
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
        if _include_message_for_display(m)
    ]
    stored_instruction = _system_from_messages(typed)
    prompt_strategy = str(meta.get("prompt_strategy") or DEFAULT_PROMPT_STRATEGY)
    instruction_ctx = _instruction_ui_context(
        config_path,
        prompt_strategy=prompt_strategy,
        stored_instruction=stored_instruction,
    )
    history = _history_without_system(typed)
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


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    records = load_collected_records(collected_records_path())
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "collected_count": len(records),
            "bench_index": _index_entries(5),
        },
    )


@app.get("/data", response_class=HTMLResponse)
async def data_page(request: Request) -> HTMLResponse:
    session_id = request.query_params.get("session")
    if request.query_params.get("new"):
        session = create_session(DEFAULT_ONLINE_CONFIG)
        session_id = session.id
    elif session_id:
        try:
            session = load_session(session_id)
        except FileNotFoundError:
            session = create_session(DEFAULT_ONLINE_CONFIG)
            session_id = session.id
    else:
        recent = list_sessions(limit=1)
        session = recent[0] if recent else create_session(DEFAULT_ONLINE_CONFIG)
        session_id = session.id

    prefill = {
        "message": request.query_params.get("message")
        or request.query_params.get("instruction", ""),
    }
    last_turn = None
    turn_q = request.query_params.get("turn")
    if turn_q and session.turns:
        try:
            idx = int(turn_q)
            last_turn = next((t for t in session.turns if t.turn_index == idx), session.turns[-1])
        except ValueError:
            last_turn = session.turns[-1]

    return templates.TemplateResponse(
        request,
        "data.html",
        {
            "session": session,
            "sessions": list_sessions(limit=10),
            "config_options": ONLINE_CONFIG_OPTIONS,
            "request": request,
            "prefill": prefill,
            "last_turn": last_turn,
        },
    )


@app.get("/data/{session_id}", response_class=HTMLResponse)
async def online_session_detail(request: Request, session_id: str) -> HTMLResponse:
    try:
        session = load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return templates.TemplateResponse(
        request,
        "online_session_detail.html",
        {"session": session},
    )


def _run_online_turn_task(
    session_id: str,
    *,
    instruction: str,
    input_text: str,
    output: str,
    tags: list[str] | None,
    preview: bool,
    tone_override: str | None = None,
) -> Any:
    return run_online_turn(
        session_id,
        instruction=instruction,
        input=input_text,
        output=output,
        tags=tags,
        preview=preview,
        tone_override=tone_override,
    )


@app.post("/data/turn")
async def online_learning_turn(
    request: Request,
    session_id: str = Form(...),
    config_path: str = Form(DEFAULT_ONLINE_CONFIG),
    message: str = Form(""),
    instruction: str = Form(""),
    preview: str = Form(""),
    tone_override: str = Form(""),
):
    instruction = (message or instruction).strip()
    if not instruction:
        raise HTTPException(status_code=400, detail="message is required")
    if config_path not in ONLINE_CONFIG_OPTIONS:
        config_path = DEFAULT_ONLINE_CONFIG

    try:
        session = load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    if session.config_path != config_path:
        session.config_path = config_path
        from sdft.online_learning.session import save_session as save_online_session

        save_online_session(session)

    tone = tone_override.strip().lower() or None
    if tone and tone not in {"positive", "neutral", "negative"}:
        tone = None
    try:
        turn = await asyncio.to_thread(
            _run_online_turn_task,
            session_id,
            instruction=instruction,
            input_text="",
            output="",
            tags=None,
            preview=preview.lower() in {"1", "true", "on", "yes"},
            tone_override=tone,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    redirect_url = f"/data?session={session_id}&turn={turn.turn_index}"
    if request.headers.get("HX-Request", "").lower() == "true":
        response = RedirectResponse(url=redirect_url, status_code=303)
        response.headers["HX-Redirect"] = redirect_url
        return response
    return RedirectResponse(url=redirect_url, status_code=303)


@app.post("/data/entry")
async def add_entry(
    instruction: str = Form(...),
    input_text: str = Form(""),
    output: str = Form(...),
    tags: str = Form(""),
) -> RedirectResponse:
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    try:
        collect_record(
            instruction,
            input=input_text,
            output=output,
            source="web",
            tags=tag_list or None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return RedirectResponse(url="/data", status_code=303)


@app.post("/data/export")
async def export_data(export_name: str = Form("geek-jokes-export")) -> RedirectResponse:
    path, count = export_collected_for_training(export_name)
    return RedirectResponse(
        url=f"/data?exported={count}&path={path.name}",
        status_code=303,
    )


@app.get("/perf", response_class=HTMLResponse)
async def perf_page(request: Request) -> HTMLResponse:
    continue_id = request.query_params.get("continue")
    chat = _chat_context_from_continue(continue_id)
    # Query overrides for legacy Re-run links (instruction + input → seed chat).
    q_instruction = request.query_params.get("instruction")
    q_input = request.query_params.get("input", "")
    q_config = request.query_params.get("config_path")
    q_strategy = request.query_params.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY)
    if q_config:
        chat["config_path"] = q_config
    if q_strategy:
        chat["prompt_strategy"] = q_strategy
    if q_instruction and not continue_id:
        seed: list[dict[str, str]] = []
        selected_for_seed = chat.get("config_path", DEFAULT_CONFIG)
        selected_strategy = chat.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY)
        instruction_ctx = _instruction_ui_context(
            selected_for_seed,
            prompt_strategy=selected_strategy,
            stored_instruction=q_instruction,
        )
        if q_input.strip():
            chat = {
                **instruction_ctx,
                "messages": seed,
                "messages_json": "[]",
                "last_run_id": None,
                "demo_condition": DEFAULT_DEMO_CONDITION,
                "prompt_strategy": selected_strategy,
                "config_path": selected_for_seed,
            }
            # Prefill composer via placeholder template var
            chat["composer_prefill"] = q_input
        else:
            chat = {
                **instruction_ctx,
                "messages": seed,
                "messages_json": "[]",
                "last_run_id": None,
                "composer_prefill": "",
                "demo_condition": DEFAULT_DEMO_CONDITION,
                "prompt_strategy": selected_strategy,
                "config_path": selected_for_seed,
            }
    else:
        chat.setdefault("composer_prefill", "")
        if "instruction_ignored" not in chat:
            chat.update(
                _instruction_ui_context(
                    chat.get("config_path", DEFAULT_CONFIG),
                    prompt_strategy=chat.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY),
                )
            )

    selected_config = chat.get("config_path") or request.query_params.get(
        "config_path", CONFIG_OPTIONS[0]
    )
    if selected_config not in CONFIG_OPTIONS:
        selected_config = CONFIG_OPTIONS[0]

    selected_strategy = chat.get("prompt_strategy") or request.query_params.get(
        "prompt_strategy", DEFAULT_PROMPT_STRATEGY
    )
    try:
        get_prompt_strategy(str(selected_strategy))
    except ValueError:
        selected_strategy = DEFAULT_PROMPT_STRATEGY
    chat["prompt_strategy"] = selected_strategy

    return templates.TemplateResponse(
        request,
        "perf.html",
        {
            "results": _load_results(20),
            "index": _index_entries(20),
            "config_options": CONFIG_OPTIONS,
            "prompt_strategy_options": prompt_strategy_options(),
            "request": request,
            "chat": chat,
            "selected_config": selected_config,
            "selected_prompt_strategy": selected_strategy,
        },
    )


def _run_benchmark_task(
    benchmark: str,
    config_path: str,
    num_examples: int,
    prompts: list[str] | None,
    records: list[dict[str, str]] | None,
    messages: list[dict[str, str]] | None = None,
    toolcall_kwargs: dict[str, Any] | None = None,
):
    kwargs: dict = {"config_path": config_path, "persist": True}
    if benchmark == "generate":
        kwargs["num_examples"] = num_examples
    elif messages is not None:
        kwargs["messages"] = messages
        if toolcall_kwargs:
            kwargs["toolcall_kwargs"] = toolcall_kwargs
    elif prompts:
        kwargs["prompts"] = prompts
        kwargs["records"] = records
        kwargs["warmup_batches"] = 0
    return run_benchmark(benchmark, **kwargs)


def _run_chat_inference(
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
    condition = get_condition(condition_id)
    effective_config = config_path if config_path in CONFIG_OPTIONS else condition.config_path
    try:
        get_prompt_strategy(prompt_strategy)
    except ValueError as exc:
        raise ValueError(str(exc)) from exc
    messages = _build_chat_messages(
        instruction,
        history,
        user_message,
        config_path=effective_config,
        prompt_strategy=prompt_strategy,
    )

    with phases.span("config_load"):
        cfg = load_config(effective_config)
    result = measure_chat(cfg, messages, latency_phases=phases)
    _attach_run_metadata(
        result,
        demo_condition=condition.id,
        config_path=effective_config,
        model_path=cfg.model.name,
        prompt_strategy=prompt_strategy,
    )
    with phases.span("persist"):
        result.metadata["latency_phases"] = phases.to_list()
        persist_performance_result(result)
    # Rewrite so on-disk JSON includes the completed persist span.
    result.metadata["latency_phases"] = phases.to_list()
    save_performance_result(performance_result_path(result.id), result)
    return result


@app.post("/perf/chat")
async def run_perf_chat(
    request: Request,
    config_path: str = Form(DEFAULT_CONFIG),
    demo_condition: str = Form(DEFAULT_DEMO_CONDITION),
    prompt_strategy: str = Form(DEFAULT_PROMPT_STRATEGY),
    instruction: str = Form(""),
    user_message: str = Form(...),
    messages_json: str = Form("[]"),
):
    """Synchronous multi-turn chat: HTMX gets a panel fragment; else redirect."""
    user_message = user_message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message is required")

    history = _parse_messages_json(messages_json)
    try:
        get_condition(demo_condition)
        get_prompt_strategy(prompt_strategy)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await asyncio.to_thread(
        _run_chat_inference,
        config_path=config_path,
        condition_id=demo_condition,
        prompt_strategy=prompt_strategy,
        instruction=instruction,
        history=history,
        user_message=user_message,
    )
    cfg_q = quote(config_path, safe="")
    strat_q = quote(prompt_strategy, safe="")
    continue_url = f"/perf?continue={result.id}&config_path={cfg_q}&prompt_strategy={strat_q}&sent=1"

    if request.headers.get("HX-Request", "").lower() == "true":
        chat = _chat_context_from_result(result, instruction_fallback=instruction)
        response = templates.TemplateResponse(
            request,
            "chat_panel.html",
            {
                "chat": chat,
                "show_sent_notice": True,
            },
        )
        response.headers["HX-Push-Url"] = continue_url
        return response

    return RedirectResponse(url=continue_url, status_code=303)


@app.post("/perf/run")
async def run_perf(
    background_tasks: BackgroundTasks,
    benchmark: str = Form("generate"),
    config_path: str = Form("configs/default.yaml"),
    num_examples: int = Form(4),
) -> RedirectResponse:
    """Background generate benchmark (chat inference uses POST /perf/chat)."""
    if benchmark != "generate":
        raise HTTPException(
            status_code=400,
            detail="use POST /perf/chat for multi-turn inference; only generate is accepted here",
        )
    background_tasks.add_task(
        _run_benchmark_task,
        benchmark,
        config_path,
        num_examples,
        None,
        None,
        None,
    )
    return RedirectResponse(url="/perf?started=1", status_code=303)


@app.get("/perf/{run_id}", response_class=HTMLResponse)
async def perf_detail(request: Request, run_id: str) -> HTMLResponse:
    path = performance_result_path(run_id)
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"benchmark {run_id!r} not found")
    result = load_performance_result(path)
    return templates.TemplateResponse(
        request,
        "run_detail.html",
        {"result": result},
    )


@app.get("/api/collected")
async def api_collected() -> dict:
    records = load_collected_records(collected_records_path())
    return {"count": len(records), "records": [r.to_dict() for r in records]}


@app.get("/api/benchmarks")
async def api_benchmarks() -> dict:
    return {"index": load_performance_index(performance_index_path())}


def main() -> None:
    import uvicorn

    uvicorn.run("web.app:app", host="127.0.0.1", port=8765, reload=False)


if __name__ == "__main__":
    main()
