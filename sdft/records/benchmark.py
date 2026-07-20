"""Performance measurement for generation and inference."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from threading import Thread
from typing import Any, Iterator

import torch
from transformers import TextIteratorStreamer

from ..config import Config, load_config
from ..data import build_teacher_messages, load_examples, sample_fewshots
from ..peft_utils import adapter_ready, load_chat_model, peft_adapter_metadata
from ..toolcall.loop import ToolLoopConfig, run_tool_loop
from ..utils import load_model, load_tokenizer, pick_device
from .latency import LatencyPhases, ms as _ms, percentile as _percentile
from .paths import (
    new_run_id,
    performance_dir,
    performance_index_path,
    performance_result_path,
    utc_now_iso,
)
from .schema import PerformanceMetrics, PerformanceResult
from .store import append_performance_index, save_performance_result

# Back-compat aliases used by older imports / tests.
_adapter_ready = adapter_ready
_peft_adapter_metadata = peft_adapter_metadata
_load_chat_model = load_chat_model


@torch.inference_mode()
def measure_generation(
    cfg: Config,
    *,
    num_examples: int | None = None,
    warmup_batches: int = 1,
    device: str | None = None,
) -> PerformanceResult:
    """Benchmark the SDFT teacher-generation loop (same path as ``sdft.generate``)."""
    device = device or pick_device()
    examples = load_examples(cfg.data)
    if num_examples is not None:
        examples = examples[: min(num_examples, len(examples))]
    if not examples:
        raise ValueError("no examples available for generation benchmark")

    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = load_model(cfg.model, device)
    model.eval()

    gen = cfg.generation
    do_sample = gen.temperature > 0
    rng_seed = cfg.data.seed
    import random

    rng = random.Random(rng_seed)
    latencies_ms: list[float] = []
    input_tokens = 0
    output_tokens = 0
    samples = 0
    batch_size = gen.batch_size

    batches = list(range(0, len(examples), batch_size))
    for batch_idx, start in enumerate(batches):
        batch = examples[start : start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                build_teacher_messages(
                    sample_fewshots(examples, start + i, gen.num_shots, rng),
                    example["prompt"],
                ),
                tokenize=False,
                add_generation_prompt=True,
            )
            for i, example in enumerate(batch)
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(device)
        batch_input_tokens = int(enc["input_ids"].numel())

        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        batch_output_tokens = int(new_tokens.numel())
        per_sample_ms = elapsed_ms / len(batch)

        if batch_idx >= warmup_batches:
            latencies_ms.extend([per_sample_ms] * len(batch))
            input_tokens += batch_input_tokens
            output_tokens += batch_output_tokens
            samples += len(batch)

    total_tokens = input_tokens + output_tokens
    total_s = sum(latencies_ms) / 1000 if latencies_ms else 0.0
    tps = total_tokens / total_s if total_s > 0 else 0.0

    metrics = PerformanceMetrics(
        latency_ms_mean=statistics.mean(latencies_ms) if latencies_ms else 0.0,
        latency_ms_p50=_percentile(latencies_ms, 50),
        latency_ms_p95=_percentile(latencies_ms, 95),
        tokens_per_second=tps,
        samples=samples,
        batch_size=batch_size,
        input_tokens_total=input_tokens,
        output_tokens_total=output_tokens,
        device=device,
        warmup_samples=warmup_batches * batch_size,
    )
    return PerformanceResult(
        id=new_run_id("bench"),
        run_at=utc_now_iso(),
        benchmark="generate",
        model=cfg.model.name,
        metrics=metrics,
        metadata={
            "num_examples": len(examples),
            "num_shots": gen.num_shots,
            "max_new_tokens": gen.max_new_tokens,
            "dataset": cfg.data.dataset,
        },
    )


@torch.inference_mode()
def measure_inference(
    cfg: Config,
    prompts: list[str],
    *,
    records: list[dict[str, str]] | None = None,
    max_new_tokens: int | None = None,
    batch_size: int = 4,
    warmup_batches: int | None = None,
    device: str | None = None,
) -> PerformanceResult:
    """Benchmark plain single-turn generation on explicit prompts.

    When ``records`` is provided (Alpaca-style ``instruction`` / ``input`` rows,
    parallel to ``prompts``), decoded model text is stored alongside them in
    ``metadata["examples"]`` for UI display.

    ``max_new_tokens`` defaults to ``cfg.generation.max_new_tokens``. Warmup
    defaults to 0 when the whole run fits in one batch (typical web UI
    single-prompt inference); otherwise 1 batch.
    """
    if not prompts:
        raise ValueError("prompts must not be empty")
    if records is not None and len(records) != len(prompts):
        raise ValueError("records must be the same length as prompts")
    if max_new_tokens is None:
        max_new_tokens = cfg.generation.max_new_tokens
    if warmup_batches is None:
        warmup_batches = 0 if len(prompts) <= batch_size else 1

    device = device or pick_device()
    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = load_model(cfg.model, device)
    model.eval()

    latencies_ms: list[float] = []
    input_tokens = 0
    output_tokens = 0
    samples = 0
    examples_out: list[dict[str, str]] = []

    for batch_idx, start in enumerate(range(0, len(prompts), batch_size)):
        batch_prompts = prompts[start : start + batch_size]
        messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(device)
        batch_input_tokens = int(enc["input_ids"].numel())

        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        batch_output_tokens = int(new_tokens.numel())
        per_sample_ms = elapsed_ms / len(batch_prompts)
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for i, text in enumerate(decoded):
            abs_i = start + i
            if records is not None:
                rec = records[abs_i]
                examples_out.append(
                    {
                        "instruction": str(rec.get("instruction", "")),
                        "input": str(rec.get("input", "")),
                        "output": text.strip(),
                    }
                )
            else:
                examples_out.append(
                    {
                        "instruction": batch_prompts[i],
                        "input": "",
                        "output": text.strip(),
                    }
                )

        if batch_idx >= warmup_batches:
            latencies_ms.extend([per_sample_ms] * len(batch_prompts))
            input_tokens += batch_input_tokens
            output_tokens += batch_output_tokens
            samples += len(batch_prompts)

    total_tokens = input_tokens + output_tokens
    total_s = sum(latencies_ms) / 1000 if latencies_ms else 0.0
    tps = total_tokens / total_s if total_s > 0 else 0.0

    return PerformanceResult(
        id=new_run_id("bench"),
        run_at=utc_now_iso(),
        benchmark="inference",
        model=cfg.model.name,
        metrics=PerformanceMetrics(
            latency_ms_mean=statistics.mean(latencies_ms) if latencies_ms else 0.0,
            latency_ms_p50=_percentile(latencies_ms, 50),
            latency_ms_p95=_percentile(latencies_ms, 95),
            tokens_per_second=tps,
            samples=samples,
            batch_size=batch_size,
            input_tokens_total=input_tokens,
            output_tokens_total=output_tokens,
            device=device,
            warmup_samples=warmup_batches * batch_size,
        ),
        metadata={
            "prompt_count": len(prompts),
            "max_new_tokens": max_new_tokens,
            "examples": examples_out,
        },
    )


def _validate_chat_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Normalize OpenAI-style chat messages; require a trailing user turn."""
    allowed = {"system", "user", "assistant"}
    cleaned: list[dict[str, str]] = []
    for i, raw in enumerate(messages):
        if not isinstance(raw, dict):
            raise ValueError(f"messages[{i}] must be a dict")
        role = str(raw.get("role", "")).strip()
        content = str(raw.get("content", "")).strip()
        if role not in allowed:
            raise ValueError(f"messages[{i}].role must be one of {sorted(allowed)}")
        if not content:
            raise ValueError(f"messages[{i}].content must not be empty")
        cleaned.append({"role": role, "content": content})
    if not cleaned:
        raise ValueError("messages must not be empty")
    if cleaned[-1]["role"] != "user":
        raise ValueError("messages must end with a user turn")
    return cleaned


def _chat_examples_summary(messages: list[dict[str, str]], assistant_text: str) -> list[dict[str, str]]:
    """Alpaca-compatible summary of the last turn for existing UI cards."""
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    last_user = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        "",
    )
    if system:
        return [{"instruction": system, "input": last_user, "output": assistant_text}]
    return [{"instruction": last_user, "input": "", "output": assistant_text}]


@torch.inference_mode()
def measure_toolcall_chat(
    cfg: Config,
    messages: list[dict[str, str]],
    *,
    few_shot_k: int = 0,
    cot_line: str | None = None,
    demo_condition: str | None = None,
    device: str | None = None,
    latency_phases: LatencyPhases | None = None,
) -> PerformanceResult:
    """Run one chat turn through the OpenClaw-style tool loop (ablation demo path)."""
    cleaned = _validate_chat_messages(messages)
    history = [m for m in cleaned[:-1] if m["role"] != "system"]
    last_user = cleaned[-1]["content"]
    phases = latency_phases or LatencyPhases()

    device = device or pick_device()
    with phases.span("tokenizer_load"):
        tokenizer = load_tokenizer(cfg.model)
    with phases.span("model_load"):
        model = load_model(cfg.model, device)
        model.eval()

    loop_cfg = ToolLoopConfig(
        max_rounds=cfg.toolcall.max_rounds,
        max_new_tokens=cfg.toolcall.max_new_tokens,
        temperature=cfg.toolcall.temperature,
        top_p=cfg.toolcall.top_p,
        format=cfg.toolcall.format,
        system_prompt=cfg.toolcall.system_prompt,
        max_context_chars=cfg.toolcall.max_context_chars,
        max_obs_chars=cfg.toolcall.max_obs_chars,
        sandbox_timeout_s=cfg.toolcall.sandbox_timeout_s,
        few_shot_k=few_shot_k,
        cot_line=cot_line,
    )

    with phases.span("generate"):
        loop_result = run_tool_loop(
            model,
            tokenizer,
            last_user,
            cfg=loop_cfg,
            device=device,
            history_messages=history or None,
        )
    generate_ms = float(phases.phases[-1]["duration_ms"])

    with phases.span("decode"):
        assistant_text = loop_result.response_text.strip()
        full_messages = [*cleaned, {"role": "assistant", "content": assistant_text}]

        prompt_text = tokenizer.apply_chat_template(
            cleaned,
            tokenize=False,
            add_generation_prompt=True,
        )
        input_tokens = len(tokenizer.encode(prompt_text, add_special_tokens=False))
        output_tokens = len(tokenizer.encode(assistant_text, add_special_tokens=False))

    total_tokens = input_tokens + output_tokens
    total_s = generate_ms / 1000 if generate_ms > 0 else 0.0
    tps = total_tokens / total_s if total_s > 0 else 0.0

    meta: dict[str, Any] = {
        "messages": full_messages,
        "examples": _chat_examples_summary(cleaned, assistant_text),
        "max_new_tokens": cfg.toolcall.max_new_tokens,
        "turn_count": sum(1 for m in cleaned if m["role"] == "user"),
        "chat": True,
        "toolcall": True,
        "tool_call_count": loop_result.tool_call_count,
        "finish_reason": loop_result.finish_reason,
        "few_shot_k": few_shot_k,
        "cot_line": cot_line,
        "toolcall_format": str(cfg.toolcall.format),
        "model_path": cfg.model.name,
        "latency_phases": phases.to_list(),
    }
    if demo_condition:
        meta["demo_condition"] = demo_condition

    return PerformanceResult(
        id=new_run_id("bench"),
        run_at=utc_now_iso(),
        benchmark="inference",
        model=cfg.model.name,
        metrics=PerformanceMetrics(
            latency_ms_mean=generate_ms,
            latency_ms_p50=generate_ms,
            latency_ms_p95=generate_ms,
            tokens_per_second=tps,
            samples=1,
            batch_size=1,
            input_tokens_total=input_tokens,
            output_tokens_total=output_tokens,
            device=device,
            warmup_samples=0,
        ),
        metadata=meta,
    )


def _chat_performance_result(
    *,
    cfg: Config,
    cleaned: list[dict[str, str]],
    assistant_text: str,
    input_tokens: int,
    output_tokens: int,
    generate_ms: float,
    max_new_tokens: int,
    device: str,
    phases: LatencyPhases,
    model_name: str | None = None,
    load_metadata: dict[str, Any] | None = None,
) -> PerformanceResult:
    full_messages = [*cleaned, {"role": "assistant", "content": assistant_text}]
    total_tokens = input_tokens + output_tokens
    total_s = generate_ms / 1000 if generate_ms > 0 else 0.0
    tps = total_tokens / total_s if total_s > 0 else 0.0
    resolved_model = model_name or cfg.model.name
    metadata: dict[str, Any] = {
        "messages": full_messages,
        "examples": _chat_examples_summary(cleaned, assistant_text),
        "max_new_tokens": max_new_tokens,
        "turn_count": sum(1 for m in cleaned if m["role"] == "user"),
        "chat": True,
        "latency_phases": phases.to_list(),
    }
    if load_metadata:
        metadata.update(load_metadata)
    return PerformanceResult(
        id=new_run_id("bench"),
        run_at=utc_now_iso(),
        benchmark="inference",
        model=resolved_model,
        metrics=PerformanceMetrics(
            latency_ms_mean=generate_ms,
            latency_ms_p50=generate_ms,
            latency_ms_p95=generate_ms,
            tokens_per_second=tps,
            samples=1,
            batch_size=1,
            input_tokens_total=input_tokens,
            output_tokens_total=output_tokens,
            device=device,
            warmup_samples=0,
        ),
        metadata=metadata,
    )


@torch.inference_mode()
def measure_chat(
    cfg: Config,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int | None = None,
    device: str | None = None,
    latency_phases: LatencyPhases | None = None,
    adapter_dir: Path | str | None = None,
    model_name: str | None = None,
) -> PerformanceResult:
    """Run one synchronous multi-turn chat completion.

    ``messages`` is OpenAI-style ``[{role, content}, ...]`` and must end with a
    user turn. The assistant reply is appended in ``metadata["messages"]``.
    An Alpaca-style last-turn summary is also stored under ``metadata["examples"]``.

    Wall-clock sub-phases are stored in ``metadata["latency_phases"]``. The
    published ``latency_ms_*`` metrics still reflect only the ``generate`` span
    (``model.generate``), matching the previous single-timer behavior.
    """
    cleaned = _validate_chat_messages(messages)
    if max_new_tokens is None:
        max_new_tokens = cfg.generation.max_new_tokens
    phases = latency_phases or LatencyPhases()

    device = device or pick_device()
    with phases.span("tokenizer_load"):
        tokenizer = load_tokenizer(cfg.model)
        tokenizer.padding_side = "left"
    with phases.span("model_load"):
        model = _load_chat_model(cfg, device, adapter_dir=adapter_dir)
        model.eval()
        load_metadata = _peft_adapter_metadata(model)

    with phases.span("prompt_build"):
        prompt_text = tokenizer.apply_chat_template(
            cleaned,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = enc.to(device)
        input_tokens = int(enc["input_ids"].numel())

    with phases.span("generate"):
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    generate_ms = float(phases.phases[-1]["duration_ms"])

    with phases.span("decode"):
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        output_tokens = int(new_tokens.numel())
        assistant_text = tokenizer.decode(new_tokens[0], skip_special_tokens=True).strip()

    return _chat_performance_result(
        cfg=cfg,
        cleaned=cleaned,
        assistant_text=assistant_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generate_ms=generate_ms,
        max_new_tokens=max_new_tokens,
        device=device,
        phases=phases,
        model_name=model_name,
        load_metadata=load_metadata,
    )


def iter_measure_chat(
    cfg: Config,
    messages: list[dict[str, str]],
    *,
    max_new_tokens: int | None = None,
    device: str | None = None,
    latency_phases: LatencyPhases | None = None,
    adapter_dir: Path | str | None = None,
    model_name: str | None = None,
) -> Iterator[tuple[str, Any]]:
    """Stream one multi-turn chat completion as ``(\"token\", text)`` then ``(\"result\", PerformanceResult)``.

    Uses HuggingFace ``TextIteratorStreamer`` so callers can push decoded text
    deltas to an SSE / WebSocket client while ``model.generate`` runs. Phase
    timing matches ``measure_chat`` (``generate`` wraps the full streamer loop).
    Persist is left to the caller, same as ``measure_chat``.
    """
    cleaned = _validate_chat_messages(messages)
    if max_new_tokens is None:
        max_new_tokens = cfg.generation.max_new_tokens
    phases = latency_phases or LatencyPhases()

    device = device or pick_device()
    with phases.span("tokenizer_load"):
        tokenizer = load_tokenizer(cfg.model)
        tokenizer.padding_side = "left"
    with phases.span("model_load"):
        model = _load_chat_model(cfg, device, adapter_dir=adapter_dir)
        model.eval()
        load_metadata = _peft_adapter_metadata(model)

    with phases.span("prompt_build"):
        prompt_text = tokenizer.apply_chat_template(
            cleaned,
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        enc = enc.to(device)
        input_tokens = int(enc["input_ids"].numel())

    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    gen_kwargs = {
        **enc,
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "streamer": streamer,
    }
    errors: list[BaseException] = []

    def _generate() -> None:
        try:
            with torch.inference_mode():
                model.generate(**gen_kwargs)
        except BaseException as exc:  # noqa: BLE001 — re-raised after streamer drains
            errors.append(exc)

    thread = Thread(target=_generate, daemon=True)
    pieces: list[str] = []
    with phases.span("generate"):
        thread.start()
        for piece in streamer:
            if piece:
                pieces.append(piece)
                yield ("token", piece)
        thread.join()
    if errors:
        raise RuntimeError(f"streaming generate failed: {errors[0]}") from errors[0]
    generate_ms = float(phases.phases[-1]["duration_ms"])

    with phases.span("decode"):
        assistant_text = "".join(pieces).strip()
        output_tokens = len(tokenizer.encode(assistant_text, add_special_tokens=False))

    result = _chat_performance_result(
        cfg=cfg,
        cleaned=cleaned,
        assistant_text=assistant_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        generate_ms=generate_ms,
        max_new_tokens=max_new_tokens,
        device=device,
        phases=phases,
        model_name=model_name,
        load_metadata=load_metadata,
    )
    yield ("result", result)


def geek_jokes_generations_path(run_id: str, root: Path | None = None) -> Path:
    """Sidecar JSONL of geek-jokes benchmark completions."""
    return performance_dir(root) / f"{run_id}_geek_jokes.jsonl"


@torch.inference_mode()
def measure_geek_jokes(
    cfg: Config,
    *,
    num_examples: int | None = None,
    warmup_samples: int = 2,
    device: str | None = None,
    generations_path: Path | None = None,
) -> PerformanceResult:
    """Generate geek-joke completions on the configured JSONL benchmark set."""
    device = device or pick_device()
    examples = load_examples(cfg.data)
    if num_examples is not None:
        examples = examples[: min(num_examples, len(examples))]
    if not examples:
        raise ValueError("no geek-jokes examples found (check data.geek_jokes.jsonl)")

    tokenizer = load_tokenizer(cfg.model)
    tokenizer.padding_side = "left"
    model = load_model(cfg.model, device)
    model.eval()

    gen = cfg.generation
    do_sample = gen.temperature > 0
    batch_size = max(1, gen.batch_size)
    run_id = new_run_id("bench")
    gen_path = generations_path or geek_jokes_generations_path(run_id)

    latencies_ms: list[float] = []
    input_tokens = 0
    output_tokens = 0
    samples = 0
    generation_rows: list[dict[str, Any]] = []

    for batch_idx, start in enumerate(range(0, len(examples), batch_size)):
        batch = examples[start : start + batch_size]
        prompts = [ex["prompt"] for ex in batch]
        messages_batch = [[{"role": "user", "content": p}] for p in prompts]
        texts = [
            tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]
        enc = tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False)
        enc = enc.to(device)
        batch_input_tokens = int(enc["input_ids"].numel())

        t0 = time.perf_counter()
        out = model.generate(
            **enc,
            max_new_tokens=gen.max_new_tokens,
            do_sample=do_sample,
            temperature=gen.temperature if do_sample else None,
            top_p=gen.top_p if do_sample else None,
            pad_token_id=tokenizer.pad_token_id,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        new_tokens = out[:, enc["input_ids"].shape[1] :]
        batch_output_tokens = int(new_tokens.numel())
        per_sample_ms = elapsed_ms / len(batch)
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for example, text in zip(batch, decoded, strict=True):
            generation_rows.append(
                {
                    "prompt": example["prompt"],
                    "reference": example["response"],
                    "generated": text.strip(),
                }
            )

        global_idx_end = start + len(batch)
        if global_idx_end <= warmup_samples:
            continue
        count_start = max(start, warmup_samples)
        counted = global_idx_end - count_start
        if counted <= 0:
            continue
        ratio = counted / len(batch)
        latencies_ms.extend([per_sample_ms] * counted)
        input_tokens += int(batch_input_tokens * ratio)
        output_tokens += int(batch_output_tokens * ratio)
        samples += counted

    gen_path.parent.mkdir(parents=True, exist_ok=True)
    gen_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in generation_rows) + "\n",
        encoding="utf-8",
    )

    total_tokens = input_tokens + output_tokens
    total_s = sum(latencies_ms) / 1000 if latencies_ms else 0.0
    tps = total_tokens / total_s if total_s > 0 else 0.0

    return PerformanceResult(
        id=run_id,
        run_at=utc_now_iso(),
        benchmark="geek_jokes",
        model=cfg.model.name,
        metrics=PerformanceMetrics(
            latency_ms_mean=statistics.mean(latencies_ms) if latencies_ms else 0.0,
            latency_ms_p50=_percentile(latencies_ms, 50),
            latency_ms_p95=_percentile(latencies_ms, 95),
            tokens_per_second=tps,
            samples=samples,
            batch_size=batch_size,
            input_tokens_total=input_tokens,
            output_tokens_total=output_tokens,
            device=device,
            warmup_samples=warmup_samples,
        ),
        metadata={
            "num_examples": len(examples),
            "max_new_tokens": gen.max_new_tokens,
            "dataset": cfg.data.data_files,
            "generations_path": str(gen_path),
        },
    )


def run_benchmark(
    benchmark: str,
    *,
    config_path: str | Path = "configs/default.yaml",
    num_examples: int = 8,
    prompts: list[str] | None = None,
    records: list[dict[str, str]] | None = None,
    messages: list[dict[str, str]] | None = None,
    warmup_batches: int | None = None,
    toolcall_kwargs: dict[str, Any] | None = None,
    persist: bool = True,
    root: Path | None = None,
) -> PerformanceResult:
    """Run a named benchmark and optionally persist results."""
    chat_phases = LatencyPhases() if messages is not None else None
    if chat_phases is not None:
        with chat_phases.span("config_load"):
            cfg = load_config(config_path)
    else:
        cfg = load_config(config_path)

    if benchmark == "generate":
        result = measure_generation(cfg, num_examples=num_examples)
    elif benchmark == "inference":
        if messages is not None:
            assert chat_phases is not None
            if toolcall_kwargs:
                result = measure_toolcall_chat(
                    cfg, messages, **toolcall_kwargs, latency_phases=chat_phases
                )
            else:
                result = measure_chat(cfg, messages, latency_phases=chat_phases)
        else:
            sample_prompts = prompts or [
                "Explain self-distillation fine-tuning in one sentence."
            ]
            infer_kwargs: dict[str, Any] = {}
            if records is not None:
                infer_kwargs["records"] = records
            if warmup_batches is not None:
                infer_kwargs["warmup_batches"] = warmup_batches
            result = measure_inference(cfg, sample_prompts, **infer_kwargs)
    elif benchmark == "geek_jokes":
        result = measure_geek_jokes(cfg, num_examples=num_examples)
    else:
        raise ValueError(
            f"unknown benchmark {benchmark!r} (use generate, inference, or geek_jokes)"
        )

    result.config_path = str(config_path)
    if persist:
        if chat_phases is not None:
            with chat_phases.span("persist"):
                result.metadata["latency_phases"] = chat_phases.to_list()
                persist_performance_result(result, root=root)
            # Rewrite JSON so the on-disk result includes the completed persist span.
            result.metadata["latency_phases"] = chat_phases.to_list()
            save_performance_result(performance_result_path(result.id, root), result)
        else:
            persist_performance_result(result, root=root)
    elif chat_phases is not None:
        result.metadata["latency_phases"] = chat_phases.to_list()
    return result


def persist_performance_result(result: PerformanceResult, *, root: Path | None = None) -> Path:
    """Write full result JSON and append a line to the index."""
    out = performance_result_path(result.id, root)
    save_performance_result(out, result)
    append_performance_index(performance_index_path(root), result)
    return out
