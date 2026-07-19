"""Local-SDFT web UI for data collection and performance testing.

Uses the shared contract in ``sdft.records`` only — see ``docs/shared-contract.md``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

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


def _index_entries(limit: int = 50) -> list[dict]:
    rows = load_performance_index(performance_index_path())
    return list(reversed(rows[-limit:]))


def _load_results(limit: int = 20):
    results = list_performance_results(performance_dir())
    return list(reversed(results[-limit:]))


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
    defaults = {
        "instruction": request.query_params.get(
            "instruction", "Tell a geek joke about PhD life"
        ),
        "input_text": request.query_params.get("input", ""),
    }
    return templates.TemplateResponse(
        request,
        "perf.html",
        {
            "results": _load_results(20),
            "index": _index_entries(20),
            "config_options": CONFIG_OPTIONS,
            "request": request,
            "defaults": defaults,
        },
    )


def _join_inference_prompt(instruction: str, input_text: str) -> str:
    parts = [p for p in (instruction.strip(), input_text.strip()) if p]
    return "\n\n".join(parts)


def _run_benchmark_task(
    benchmark: str,
    config_path: str,
    num_examples: int,
    prompts: list[str] | None,
    records: list[dict[str, str]] | None,
) -> None:
    kwargs: dict = {"config_path": config_path, "persist": True}
    if benchmark == "generate":
        kwargs["num_examples"] = num_examples
    elif prompts:
        kwargs["prompts"] = prompts
        kwargs["records"] = records
        # Interactive web runs use a single prompt; skip warmup so I/O is counted.
        kwargs["warmup_batches"] = 0
    run_benchmark(benchmark, **kwargs)


@app.post("/perf/run")
async def run_perf(
    background_tasks: BackgroundTasks,
    benchmark: str = Form("inference"),
    config_path: str = Form("configs/default.yaml"),
    num_examples: int = Form(4),
    instruction: str = Form("Explain self-distillation fine-tuning in one sentence."),
    input_text: str = Form(""),
) -> RedirectResponse:
    if benchmark not in {"generate", "inference"}:
        raise HTTPException(status_code=400, detail="benchmark must be generate or inference")
    prompts: list[str] | None = None
    records: list[dict[str, str]] | None = None
    if benchmark == "inference":
        prompt = _join_inference_prompt(instruction, input_text)
        if not prompt:
            raise HTTPException(status_code=400, detail="instruction is required for inference")
        prompts = [prompt]
        records = [
            {
                "instruction": instruction.strip(),
                "input": input_text.strip(),
            }
        ]
    background_tasks.add_task(
        _run_benchmark_task,
        benchmark,
        config_path,
        num_examples,
        prompts,
        records,
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
