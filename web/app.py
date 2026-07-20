"""Local-SDFT web UI for data collection and performance testing.

Uses the shared contract in ``sdft.records`` — see ``docs/shared-contract.md``.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from sdft.config import load_config
from sdft.online_learning import build_session, load_session, run_online_turn
from sdft.online_learning.session import adapter_ready, list_sessions, resolve_session, save_session, session_persisted
from sdft.records import (
    collect_record,
    collected_records_path,
    export_collected_for_training,
    iter_measure_chat,  # noqa: F401 — re-exported for tests that patch web.app
    list_performance_results,
    load_collected_records,
    load_performance_index,
    load_performance_result,
    measure_chat,  # noqa: F401
    performance_dir,
    performance_index_path,
    performance_result_path,
    persist_performance_result,  # noqa: F401
    run_benchmark,
)
from sdft.records.benchmark import LatencyPhases  # noqa: F401 — tests may patch
from sdft.records.store import save_performance_result  # noqa: F401

from web.chat_context import (
    CONFIG_OPTIONS,  # noqa: F401
    chat_context_from_continue,
    chat_context_from_result,
    instruction_ui_context,
    parse_messages_json,
)
from web.demo_conditions import (
    DEFAULT_CONFIG,
    DEFAULT_DEMO_CONDITION,
    DEFAULT_PROMPT_STRATEGY,
    get_condition,
    get_prompt_strategy,
    prompt_strategy_options,
)
from web.perf_models import (
    normalize_perf_config_path,
    perf_infer_url_for_session,
    perf_model_options,
    resolve_perf_config,
    resolve_perf_config_from_adapter,
)
from web.perf_runtime import (
    continue_url,
    format_sse,
    iter_chat_inference_events,
    run_chat_inference,
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

ONLINE_CONFIG_OPTIONS = [
    "configs/online_learning.yaml",
    DEFAULT_CONFIG,
]
DEFAULT_ONLINE_CONFIG = ONLINE_CONFIG_OPTIONS[0]


def _perf_config_options() -> list[dict[str, str]]:
    return [
        {"value": opt.value, "label": opt.label, "kind": opt.kind}
        for opt in perf_model_options()
    ]


def _normalize_perf_config(config_path: str) -> str:
    return normalize_perf_config_path(config_path)


def _index_entries(limit: int = 50) -> list[dict]:
    rows = load_performance_index(performance_index_path())
    return list(reversed(rows[-limit:]))


def _load_results(limit: int = 20):
    results = list_performance_results(performance_dir())
    return list(reversed(results[-limit:]))


def _online_data_context(
    session: Any,
    *,
    prefill: dict[str, str] | None = None,
    last_turn: Any | None = None,
) -> dict[str, Any]:
    return {
        "session": session,
        "session_persisted": session_persisted(session.id),
        "config_options": ONLINE_CONFIG_OPTIONS,
        "prefill": prefill or {"message": ""},
        "last_turn": last_turn,
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
    session_param = request.query_params.get("session") or request.query_params.get("session_id")
    if session_param:
        try:
            session = load_session(session_param)
        except FileNotFoundError:
            session = build_session(DEFAULT_ONLINE_CONFIG)
    else:
        session = build_session(DEFAULT_ONLINE_CONFIG)

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
            **_online_data_context(session, prefill=prefill, last_turn=last_turn),
            "sessions": list_sessions(limit=10),
            "request": request,
        },
    )


@app.get("/data/{session_id}", response_class=HTMLResponse)
async def online_session_detail(request: Request, session_id: str) -> HTMLResponse:
    try:
        session = load_session(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    perf_infer_url = perf_infer_url_for_session(session_id) if adapter_ready(session.adapter_dir) else None
    return templates.TemplateResponse(
        request,
        "online_session_detail.html",
        {"session": session, "perf_infer_url": perf_infer_url},
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
    config_path: str | None = None,
) -> Any:
    return run_online_turn(
        session_id,
        instruction=instruction,
        input=input_text,
        output=output,
        tags=tags,
        preview=preview,
        tone_override=tone_override,
        config_path=config_path,
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

    session = resolve_session(session_id, config_path=config_path)
    if session.config_path != config_path:
        session.config_path = config_path
        cfg = load_config(config_path)
        session.model = cfg.model.name
        if session_persisted(session_id):
            save_session(session)

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
            config_path=config_path,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    redirect_url = f"/data?session={session_id}&turn={turn.turn_index}"
    if request.headers.get("HX-Request", "").lower() == "true":
        session = load_session(session_id)
        response = templates.TemplateResponse(
            request,
            "data_panel.html",
            _online_data_context(session, prefill={"message": ""}, last_turn=turn),
        )
        response.headers["HX-Push-Url"] = redirect_url
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
async def export_data(export_name: str = Form("training-export")) -> RedirectResponse:
    path, count = export_collected_for_training(export_name)
    return RedirectResponse(
        url=f"/data?exported={count}&path={path.name}",
        status_code=303,
    )


@app.get("/perf", response_class=HTMLResponse)
async def perf_page(request: Request) -> HTMLResponse:
    continue_id = request.query_params.get("continue")
    chat = chat_context_from_continue(continue_id)
    q_instruction = request.query_params.get("instruction")
    q_input = request.query_params.get("input", "")
    q_config = request.query_params.get("config_path")
    q_adapter = request.query_params.get("adapter")
    q_strategy = request.query_params.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY)
    if q_adapter and not q_config:
        mapped = resolve_perf_config_from_adapter(q_adapter)
        if mapped:
            q_config = mapped
    if q_config:
        chat["config_path"] = _normalize_perf_config(q_config)
    if q_strategy:
        chat["prompt_strategy"] = q_strategy
    if q_instruction and not continue_id:
        seed: list[dict[str, str]] = []
        selected_for_seed = chat.get("config_path", DEFAULT_CONFIG)
        selected_strategy = chat.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY)
        instruction_ctx = instruction_ui_context(
            selected_for_seed,
            prompt_strategy=selected_strategy,
            stored_instruction=q_instruction,
        )
        chat = {
            **instruction_ctx,
            "messages": seed,
            "messages_json": "[]",
            "last_run_id": None,
            "composer_prefill": q_input.strip() if q_input.strip() else "",
            "demo_condition": DEFAULT_DEMO_CONDITION,
            "prompt_strategy": selected_strategy,
            "config_path": selected_for_seed,
        }
    else:
        chat.setdefault("composer_prefill", "")
        if "instruction_ignored" not in chat:
            chat.update(
                instruction_ui_context(
                    chat.get("config_path", DEFAULT_CONFIG),
                    prompt_strategy=chat.get("prompt_strategy", DEFAULT_PROMPT_STRATEGY),
                )
            )

    selected_config = _normalize_perf_config(
        chat.get("config_path") or request.query_params.get("config_path", DEFAULT_CONFIG)
    )
    chat["config_path"] = selected_config

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
            "config_options": _perf_config_options(),
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

    history = parse_messages_json(messages_json)
    config_path = _normalize_perf_config(config_path)
    try:
        get_condition(demo_condition)
        get_prompt_strategy(prompt_strategy)
        resolve_perf_config(config_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await asyncio.to_thread(
        run_chat_inference,
        config_path=config_path,
        condition_id=demo_condition,
        prompt_strategy=prompt_strategy,
        instruction=instruction,
        history=history,
        user_message=user_message,
    )
    url = continue_url(
        result.id,
        str(result.config_path or config_path),
        prompt_strategy,
    )

    if request.headers.get("HX-Request", "").lower() == "true":
        chat = chat_context_from_result(result, instruction_fallback=instruction)
        response = templates.TemplateResponse(
            request,
            "chat_panel.html",
            {"chat": chat, "show_sent_notice": True},
        )
        response.headers["HX-Push-Url"] = url
        return response

    return RedirectResponse(url=url, status_code=303)


@app.post("/perf/chat/stream")
async def run_perf_chat_stream(
    request: Request,
    config_path: str = Form(DEFAULT_CONFIG),
    demo_condition: str = Form(DEFAULT_DEMO_CONDITION),
    prompt_strategy: str = Form(DEFAULT_PROMPT_STRATEGY),
    instruction: str = Form(""),
    user_message: str = Form(...),
    messages_json: str = Form("[]"),
):
    """SSE stream of chat tokens, then a ``done`` event with the panel HTML."""
    user_message = user_message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message is required")

    history = parse_messages_json(messages_json)
    config_path = _normalize_perf_config(config_path)
    try:
        get_condition(demo_condition)
        get_prompt_strategy(prompt_strategy)
        resolve_perf_config(config_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    out_q: queue.Queue[tuple[str, Any] | None] = queue.Queue()

    def _producer() -> None:
        try:
            for kind, payload in iter_chat_inference_events(
                config_path=config_path,
                condition_id=demo_condition,
                prompt_strategy=prompt_strategy,
                instruction=instruction,
                history=history,
                user_message=user_message,
            ):
                out_q.put((kind, payload))
        except Exception as exc:  # noqa: BLE001 — surfaced to client as SSE error
            out_q.put(("error", str(exc)))
        finally:
            out_q.put(None)

    threading.Thread(target=_producer, daemon=True).start()

    async def event_gen():
        while True:
            item = await asyncio.to_thread(out_q.get)
            if item is None:
                break
            kind, payload = item
            if kind == "token":
                yield format_sse("token", {"text": str(payload)})
            elif kind == "result":
                chat = chat_context_from_result(payload, instruction_fallback=instruction)
                html = templates.get_template("chat_panel.html").render(
                    {
                        "request": request,
                        "chat": chat,
                        "show_sent_notice": True,
                    }
                )
                url = continue_url(
                    payload.id,
                    str(payload.config_path or config_path),
                    prompt_strategy,
                )
                yield format_sse(
                    "done",
                    {
                        "run_id": payload.id,
                        "continue_url": url,
                        "html": html,
                        "messages": chat["messages"],
                        "latency_phases": chat.get("latency_phases") or [],
                        "metrics": {
                            "latency_ms_mean": payload.metrics.latency_ms_mean,
                            "tokens_per_second": payload.metrics.tokens_per_second,
                            "output_tokens_total": payload.metrics.output_tokens_total,
                        },
                    },
                )
            elif kind == "error":
                yield format_sse("error", {"detail": str(payload)})
            else:
                yield format_sse("error", {"detail": f"unexpected event {kind!r}"})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
