"""Multi-turn tool-calling inference loop for LFM2.5-230M."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .format import (
    DEFAULT_LFM_JSON_SYSTEM,
    DEFAULT_OPENCLAW_SYSTEM,
    ToolCallFormat,
    build_openclaw_prompt,
    detect_tool_call_format,
    format_invalid_action_hint,
    format_tool_observation,
    parse_assistant_action,
    postprocess_assistant_text,
)
from .sandbox import CODE_INTERPRETER_TOOL, execute_code_interpreter


@dataclasses.dataclass
class ToolLoopResult:
    messages: list[dict[str, str]]
    response_text: str
    tool_call_count: int
    finished: bool
    finish_reason: str


@dataclasses.dataclass
class ToolLoopConfig:
    max_rounds: int = 16
    max_new_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    format: ToolCallFormat | str = ToolCallFormat.OPENCLAW
    system_prompt: str | None = None
    max_context_chars: int = 12000
    max_obs_chars: int = 1024
    sandbox_timeout_s: int = 30


def _resolve_format(fmt: ToolCallFormat | str, tokenizer: PreTrainedTokenizerBase) -> ToolCallFormat:
    if isinstance(fmt, ToolCallFormat):
        return fmt
    if fmt == "auto":
        return detect_tool_call_format(tokenizer)
    return ToolCallFormat(fmt)


def _default_system(fmt: ToolCallFormat) -> str:
    if fmt == ToolCallFormat.LFM:
        return DEFAULT_LFM_JSON_SYSTEM
    return DEFAULT_OPENCLAW_SYSTEM


def _render_prompt(
    *,
    tokenizer: PreTrainedTokenizerBase,
    messages: list[dict[str, str]],
    tools: list[dict[str, Any]],
    fmt: ToolCallFormat,
    system_prompt: str,
) -> str:
    if fmt == ToolCallFormat.LFM:
        chat_messages = [{"role": "system", "content": system_prompt}, *messages]
        return tokenizer.apply_chat_template(
            chat_messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
        )

    user_prompt = messages[0]["content"] if messages else ""
    return build_openclaw_prompt(
        user_prompt=user_prompt,
        tools=tools,
        system_prompt=system_prompt,
        history=messages[1:],
    )


@torch.inference_mode()
def run_tool_loop(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    user_prompt: str,
    *,
    cfg: ToolLoopConfig | None = None,
    device: str | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> ToolLoopResult:
    """Run a ReTool-style tool loop until answer, limit, or context overflow."""
    cfg = cfg or ToolLoopConfig()
    fmt = _resolve_format(cfg.format, tokenizer)
    system_prompt = cfg.system_prompt or _default_system(fmt)
    tool_specs = tools or [CODE_INTERPRETER_TOOL]

    if device is None:
        device = str(next(model.parameters()).device)

    messages: list[dict[str, str]] = [{"role": "user", "content": user_prompt}]
    response_parts: list[str] = []
    tool_call_count = 0
    finish_reason = "max_rounds"
    finished = False

    do_sample = cfg.temperature > 0
    eos_id = tokenizer.eos_token_id
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": eos_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = cfg.temperature
        gen_kwargs["top_p"] = cfg.top_p

    for _round in range(cfg.max_rounds):
        prompt = _render_prompt(
            tokenizer=tokenizer,
            messages=messages,
            tools=tool_specs,
            fmt=fmt,
            system_prompt=system_prompt,
        )
        if len(prompt) > cfg.max_context_chars:
            finish_reason = "context_overflow"
            break

        enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        out = model.generate(**enc, **gen_kwargs)
        new_tokens = out[:, enc["input_ids"].shape[1] :]
        raw = tokenizer.decode(new_tokens[0], skip_special_tokens=False)
        if eos_id is not None and eos_id in new_tokens[0].tolist():
            raw = raw.split(tokenizer.eos_token or "")[0]
        assistant_text = postprocess_assistant_text(raw.strip())
        response_parts.append(assistant_text)
        messages.append({"role": "assistant", "content": assistant_text})

        action, content = parse_assistant_action(assistant_text)
        if action == "answer":
            finished = True
            finish_reason = "answer"
            break

        if action == "code":
            result = execute_code_interpreter(content, timeout_s=cfg.sandbox_timeout_s)
            if len(result) > cfg.max_obs_chars:
                result = result[: cfg.max_obs_chars] + f"\n... [truncated]"
            observation = format_tool_observation(result, fmt=fmt)
            tool_call_count += 1
            if fmt == ToolCallFormat.LFM:
                messages.append({"role": "tool", "content": result})
            else:
                messages.append({"role": "tool", "content": observation})
                response_parts.append(observation)
            continue

        hint = format_invalid_action_hint(fmt=fmt)
        if fmt == ToolCallFormat.OPENCLAW:
            response_parts.append(hint)
            messages.append({"role": "tool", "content": hint})
        else:
            messages.append({"role": "tool", "content": hint.strip()})

    return ToolLoopResult(
        messages=messages,
        response_text="".join(response_parts),
        tool_call_count=tool_call_count,
        finished=finished,
        finish_reason=finish_reason,
    )
