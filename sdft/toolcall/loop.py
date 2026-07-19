"""Multi-turn tool-calling inference loop for LFM2.5-230M."""

from __future__ import annotations

import dataclasses
from typing import Any

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .format import (
    DEFAULT_COT_LINE,
    DEFAULT_LFM_JSON_SYSTEM,
    DEFAULT_OPENCLAW_SYSTEM,
    LFM_TOOL_CALL_END,
    LFM_TOOL_CALL_START,
    ToolCallFormat,
    build_openclaw_prompt,
    detect_tool_call_format,
    format_invalid_action_hint,
    format_tool_observation,
    parse_assistant_action,
    postprocess_assistant_text,
    with_cot_line,
)
from .sandbox import CODE_INTERPRETER_TOOL, execute_code_interpreter

# Fixed demos for few-shot eval (must stay out of train and held-out eval).
_ONE_SHOT_QUESTION = "What is 3 + 5?"
_ONE_SHOT_CODE = "print(3 + 5)"
_ONE_SHOT_RESULT = "8"
_ONE_SHOT_ANSWER = "8"


def default_few_shot_messages(
    fmt: ToolCallFormat,
    k: int = 1,
    *,
    cot_line: str | None = None,
) -> list[dict[str, str]]:
    """Return up to ``k`` canned tool-use demonstrations as chat messages."""
    if k <= 0:
        return []
    cot_prefix = f"{cot_line.strip()}\n\n" if cot_line and cot_line.strip() else ""
    if fmt == ToolCallFormat.LFM:
        demo = [
            {"role": "user", "content": _ONE_SHOT_QUESTION},
            {
                "role": "assistant",
                "content": (
                    f"{cot_prefix}"
                    f'{LFM_TOOL_CALL_START}[{{"name": "code_interpreter", '
                    f'"arguments": {{"code": "{_ONE_SHOT_CODE}"}}}}]{LFM_TOOL_CALL_END}'
                ),
            },
            {"role": "tool", "content": _ONE_SHOT_RESULT},
            {"role": "assistant", "content": f"Answer: \\boxed{{{_ONE_SHOT_ANSWER}}}"},
        ]
    else:
        demo = [
            {"role": "user", "content": _ONE_SHOT_QUESTION},
            {
                "role": "assistant",
                "content": (
                    f"{cot_prefix}I'll use the code interpreter.\n\n"
                    "<tool_call>\n"
                    f'{{"name": "code_interpreter", "arguments": {{"code": "{_ONE_SHOT_CODE}"}}}}\n'
                    "</tool_call>\n\n"
                    f"<interpreter>\n{_ONE_SHOT_RESULT}\n</interpreter>\n\n"
                    f"Answer: \\boxed{{{_ONE_SHOT_ANSWER}}}"
                ),
            },
        ]
    if k == 1:
        return demo
    out: list[dict[str, str]] = []
    for _ in range(k):
        out.extend(demo)
    return out


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
    few_shot_k: int = 0
    cot_line: str | None = None


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

    return build_openclaw_prompt(
        user_prompt=messages[0]["content"] if messages else "",
        tools=tools,
        system_prompt=system_prompt,
        history=messages[1:] if len(messages) > 1 else None,
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
    few_shot_messages: list[dict[str, str]] | None = None,
) -> ToolLoopResult:
    """Run a ReTool-style tool loop until answer, limit, or context overflow."""
    cfg = cfg or ToolLoopConfig()
    fmt = _resolve_format(cfg.format, tokenizer)
    base_system = cfg.system_prompt or _default_system(fmt)
    cot = cfg.cot_line
    if cot is True:  # type: ignore[comparison-overlap]
        cot = DEFAULT_COT_LINE
    cot_str = cot if isinstance(cot, str) else None
    system_prompt = with_cot_line(base_system, cot_str)
    tool_specs = tools or [CODE_INTERPRETER_TOOL]

    if device is None:
        device = str(next(model.parameters()).device)

    prefix = few_shot_messages
    if prefix is None and cfg.few_shot_k > 0:
        prefix = default_few_shot_messages(fmt, cfg.few_shot_k, cot_line=cot_str)
    messages: list[dict[str, str]] = list(prefix or [])
    messages.append({"role": "user", "content": user_prompt})
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
