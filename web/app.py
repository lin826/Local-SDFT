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

from sdft.records import (
    collect_record,
    collected_records_path,
    export_collected_for_training,
    list_performance_results,
    load_collected_records,
    load_performance_index,
    load_performance_result,
    performance_dir,
    performance_index_path,
    performance_result_path,
    run_benchmark,
)

WEB_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(WEB_DIR / "templates"))

app = FastAPI(title="Local-SDFT", description="Data collection and performance testing")
app.mount("/static", StaticFiles(directory=str(WEB_DIR / "static")), name="static")

CONFIG_OPTIONS = ["configs/default.yaml", "configs/geek_jokes.yaml"]
ALLOWED_CHAT_ROLES = {"system", "user", "assistant"}


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


def _build_chat_messages(
    instruction: str,
    history: list[dict[str, str]],
    user_message: str,
) -> list[dict[str, str]]:
    """Assemble model input: optional system + prior turns + new user message."""
    messages: list[dict[str, str]] = []
    instr = instruction.strip()
    if instr:
        messages.append({"role": "system", "content": instr})
    for m in history:
        role = m["role"]
        if role == "system":
            continue
        messages.append({"role": role, "content": m["content"]})
    messages.append({"role": "user", "content": user_message.strip()})
    return messages


def _chat_context_from_continue(run_id: str | None) -> dict[str, Any]:
    """Load sticky instruction + history from a prior chat run for ?continue=."""
    empty = {
        "instruction": "Tell a geek joke about PhD life",
        "messages": [],
        "messages_json": "[]",
        "last_run_id": None,
    }
    if not run_id:
        return empty
    path = performance_result_path(run_id)
    if not path.is_file():
        return empty
    result = load_performance_result(path)
    meta = result.metadata or {}
    messages = meta.get("messages")
    if not isinstance(messages, list) or not messages:
        # Fall back to single-turn examples for re-run compatibility.
        examples = meta.get("examples") or []
        if examples:
            ex = examples[0]
            instruction = str(ex.get("instruction") or empty["instruction"])
            user_text = str(ex.get("input") or "").strip() or instruction
            history = [{"role": "user", "content": user_text}]
            if ex.get("output"):
                history.append({"role": "assistant", "content": str(ex["output"])})
            return {
                "instruction": instruction if ex.get("input") else empty["instruction"],
                "messages": history,
                "messages_json": json.dumps(history, ensure_ascii=False),
                "last_run_id": run_id,
            }
        return empty

    typed = [
        {"role": str(m.get("role", "")), "content": str(m.get("content", ""))}
        for m in messages
        if isinstance(m, dict) and m.get("role") in ALLOWED_CHAT_ROLES and str(m.get("content", "")).strip()
    ]
    instruction = _system_from_messages(typed) or empty["instruction"]
    history = _history_without_system(typed)
    return {
        "instruction": instruction,
        "messages": history,
        "messages_json": json.dumps(history, ensure_ascii=False),
        "last_run_id": run_id,
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
    records = load_collected_records(collected_records_path())
    prefill = {
        "instruction": request.query_params.get("instruction", ""),
        "input_text": request.query_params.get("input_text", ""),
        "output": request.query_params.get("output", ""),
    }
    return templates.TemplateResponse(
        request,
        "data.html",
        {"records": list(reversed(records)), "request": request, "prefill": prefill},
    )


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
    if q_instruction and not continue_id:
        seed: list[dict[str, str]] = []
        if q_input.strip():
            chat = {
                "instruction": q_instruction,
                "messages": seed,
                "messages_json": "[]",
                "last_run_id": None,
            }
            # Prefill composer via placeholder template var
            chat["composer_prefill"] = q_input
        else:
            chat = {
                "instruction": q_instruction,
                "messages": seed,
                "messages_json": "[]",
                "last_run_id": None,
                "composer_prefill": "",
            }
    else:
        chat.setdefault("composer_prefill", "")

    return templates.TemplateResponse(
        request,
        "perf.html",
        {
            "results": _load_results(20),
            "index": _index_entries(20),
            "config_options": CONFIG_OPTIONS,
            "request": request,
            "chat": chat,
            "selected_config": request.query_params.get(
                "config_path", CONFIG_OPTIONS[0]
            ),
        },
    )


def _run_benchmark_task(
    benchmark: str,
    config_path: str,
    num_examples: int,
    prompts: list[str] | None,
    records: list[dict[str, str]] | None,
    messages: list[dict[str, str]] | None = None,
):
    kwargs: dict = {"config_path": config_path, "persist": True}
    if benchmark == "generate":
        kwargs["num_examples"] = num_examples
    elif messages is not None:
        kwargs["messages"] = messages
    elif prompts:
        kwargs["prompts"] = prompts
        kwargs["records"] = records
        # Interactive web runs use a single prompt; skip warmup so I/O is counted.
        kwargs["warmup_batches"] = 0
    return run_benchmark(benchmark, **kwargs)


@app.post("/perf/chat")
async def run_perf_chat(
    config_path: str = Form("configs/default.yaml"),
    instruction: str = Form(""),
    user_message: str = Form(...),
    messages_json: str = Form("[]"),
) -> RedirectResponse:
    """Synchronous multi-turn chat: wait for the model, then continue on /perf."""
    user_message = user_message.strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="user_message is required")

    history = _parse_messages_json(messages_json)
    try:
        messages = _build_chat_messages(instruction, history, user_message)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result = await asyncio.to_thread(
        _run_benchmark_task,
        "inference",
        config_path,
        1,
        None,
        None,
        messages,
    )
    cfg_q = quote(config_path, safe="")
    return RedirectResponse(
        url=f"/perf?continue={result.id}&config_path={cfg_q}&sent=1",
        status_code=303,
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
