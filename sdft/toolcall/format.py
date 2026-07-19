"""Parse and format tool calls for LFM2.5 native and OpenClaw-RL (ReTool) protocols."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any

try:
    from jinja2 import Template
except ImportError:  # pragma: no cover - optional for LFM-native mode only
    Template = None  # type: ignore[misc, assignment]

LFM_TOOL_CALL_START = "<|tool_call_start|>"
LFM_TOOL_CALL_END = "<|tool_call_end|>"

# OpenClaw-RL ReTool templates (from toolcall-rl/generate_with_retool.py).
OPENCLAW_TOOL_TEMPLATE_JSON = """<|im_start|>system
{%- if messages[0]['role'] == 'system' %}
{{- messages[0]['content'] }}
{%- else %}
You are a helpful assistant.
{%- endif %}
{%- if tools %}
# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{%- for tool in tools %}
{{- tool | tojson }}
{%- endfor %}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{%- endif %}
<|im_end|>
{%- for message in messages %}
{%- if message['role'] == 'user' %}
<|im_start|>user
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'assistant' %}
<|im_start|>assistant
{{- message['content'] }}<|im_end|>
{%- elif message['role'] == 'tool' %}
<|im_start|>tool
{{- message['content'] }}<|im_end|>
{%- endif %}
{%- endfor %}
<|im_start|>assistant
"""

DEFAULT_OPENCLAW_SYSTEM = (
    "You are a helpful assistant that can use Python tools to solve mathematical problems. "
    "When you need to perform calculations, use the code_interpreter tool to execute code and "
    "get results. When you have the final answer, respond with: Answer: \\boxed{answer}"
)

DEFAULT_LFM_JSON_SYSTEM = (
    "Output function calls as JSON inside <|tool_call_start|> and <|tool_call_end|> "
    "tags using the format "
    '[{"name": "code_interpreter", "arguments": {"code": "..."}}]. '
    "When you have the final answer, respond in plain text."
)


class ToolCallFormat(str, Enum):
    """Conversation formatting protocol."""

    OPENCLAW = "openclaw"  # ReTool / Qwen-style <tool_call> JSON + <interpreter> observations
    LFM = "lfm"  # LFM2.5 native apply_chat_template + tool role messages


def detect_tool_call_format(tokenizer) -> ToolCallFormat:
    """Pick OpenClaw JSON vs LFM native based on the model chat template."""
    chat_template = getattr(tokenizer, "chat_template", "") or ""
    if LFM_TOOL_CALL_START in chat_template and "render_tool_calls" in chat_template:
        return ToolCallFormat.LFM
    if "<function=" in chat_template or "<parameter=" in chat_template:
        return ToolCallFormat.OPENCLAW
    return ToolCallFormat.OPENCLAW


def build_openclaw_prompt(
    *,
    user_prompt: str,
    tools: list[dict[str, Any]],
    system_prompt: str | None = None,
    history: list[dict[str, str]] | None = None,
) -> str:
    """Render an OpenClaw-RL ReTool conversation prefix ending at assistant generation."""
    if Template is None:
        raise ImportError(
            "OpenClaw prompt formatting requires jinja2. Install with: uv sync --extra toolcall"
        )
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    if history:
        messages.extend(history)
    return Template(OPENCLAW_TOOL_TEMPLATE_JSON).render(messages=messages, tools=tools)


def format_tool_observation(result: str, *, fmt: ToolCallFormat) -> str:
    """Wrap sandbox output for the active protocol."""
    if fmt == ToolCallFormat.OPENCLAW:
        return f"\n\n<interpreter>\n{result}\n</interpreter>\n\n"
    return result


def format_invalid_action_hint(*, fmt: ToolCallFormat) -> str:
    if fmt == ToolCallFormat.OPENCLAW:
        return (
            "\nMy previous action is invalid. "
            'If I want to execute code, use: <tool_call>\n'
            '{"name": "code_interpreter", "arguments": {"code": "...code..."}}\n'
            "</tool_call>. "
            "For the final answer use: Answer: \\boxed{answer}. Let me try again.\n"
        )
    return (
        "\nInvalid action. Call code_interpreter with JSON like "
        '[{"name": "code_interpreter", "arguments": {"code": "..."}}] '
        "or answer directly.\n"
    )


def _match_answer_boxed(text: str) -> str | None:
    match = re.search(r"Answer:\s*\\boxed\{", text)
    if match:
        start = match.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            return text[start : i - 1]

    # Fallback: last \boxed{...} (with or without Answer: / $ delimiters)
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    start = idx + len("\\boxed{")
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth == 0:
        return text[start : i - 1]
    return None


def _parse_openclaw_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    tool_call_json = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
    if not tool_call_json:
        return None
    try:
        payload = json.loads(tool_call_json.group(1).replace("\n", "\\n"))
    except json.JSONDecodeError:
        return None
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def _parse_lfm_pythonic_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    pattern = (
        rf"{re.escape(LFM_TOOL_CALL_START)}\s*\[(.*?)\]\s*{re.escape(LFM_TOOL_CALL_END)}"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    inner = match.group(1).strip()
    call_match = re.match(r"(\w+)\((.*)\)\s*$", inner, re.DOTALL)
    if not call_match:
        return None
    name = call_match.group(1)
    args_str = call_match.group(2).strip()
    arguments: dict[str, Any] = {}
    if args_str:
        for part in re.finditer(r"(\w+)\s*=\s*(.+?)(?:,\s*|\Z)", args_str + ",", re.DOTALL):
            key = part.group(1)
            raw = part.group(2).strip().rstrip(",")
            if (raw.startswith('"') and raw.endswith('"')) or (
                raw.startswith("'") and raw.endswith("'")
            ):
                arguments[key] = raw[1:-1]
            else:
                try:
                    arguments[key] = json.loads(raw)
                except json.JSONDecodeError:
                    arguments[key] = raw
    return name, arguments


def _parse_lfm_json_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    pattern = (
        rf"{re.escape(LFM_TOOL_CALL_START)}\s*(\[.*?\]|\{{.*?\}})\s*"
        rf"{re.escape(LFM_TOOL_CALL_END)}"
    )
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if isinstance(payload, list) and payload:
        payload = payload[0]
    if not isinstance(payload, dict):
        return None
    name = payload.get("name")
    if not isinstance(name, str):
        return None
    arguments = payload.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {}
    return name, arguments


def parse_assistant_action(text: str) -> tuple[str | None, str]:
    """Return (action, content) where action is 'answer', 'code', or None."""
    boxed = _match_answer_boxed(text)
    if boxed is not None:
        return "answer", boxed.strip()

    for parser in (_parse_openclaw_tool_call, _parse_lfm_json_tool_call, _parse_lfm_pythonic_tool_call):
        parsed = parser(text)
        if parsed and parsed[0] == "code_interpreter":
            code = parsed[1].get("code", "")
            if isinstance(code, str) and code.strip():
                return "code", code.strip()

    code_pattern = r"<code>(.*?)</code>"
    code_match = re.search(code_pattern, text, re.DOTALL)
    if code_match:
        return "code", code_match.group(1).strip()

    python_code_pattern = r"```python\s*(.*?)\s*```"
    python_code_match = re.search(python_code_pattern, text, re.DOTALL)
    if python_code_match:
        return "code", python_code_match.group(1).strip()

    return None, ""


def postprocess_assistant_text(text: str) -> str:
    """Trim generation at the first complete tool call or final answer."""
    if "<tool_call>" in text and "</tool_call>" in text:
        matches = list(re.finditer(r"<tool_call>.*?</tool_call>", text, re.DOTALL))
        if matches:
            return text[: matches[-1].end()]

    if LFM_TOOL_CALL_START in text and LFM_TOOL_CALL_END in text:
        matches = list(
            re.finditer(
                rf"{re.escape(LFM_TOOL_CALL_START)}.*?{re.escape(LFM_TOOL_CALL_END)}",
                text,
                re.DOTALL,
            )
        )
        if matches:
            return text[: matches[-1].end()]

    if "</code>" in text:
        return text.split("</code>")[0] + "</code>"

    if "```python" in text:
        matches = list(re.finditer(r"```python\s*.*?```", text, re.DOTALL))
        if matches:
            return text[: matches[-1].end()]

    if "Answer:" in text and "\\boxed{" in text:
        span_start = text.rfind("Answer:")
        if span_start >= 0:
            sub = text[span_start:]
            boxed = _match_answer_boxed(sub)
            if boxed is not None:
                end = sub.find("\\boxed{") + len("\\boxed{") + len(boxed) + 1
                return text[: span_start + end]

    return text
