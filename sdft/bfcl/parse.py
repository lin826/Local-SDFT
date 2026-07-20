"""Parse model generations into BFCL-style function-call ASTs.

Output shape matches BFCL possible_answer entries::

    [{"func_name": {"arg": value, ...}}, ...]
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from sdft.toolcall.format import LFM_TOOL_CALL_END, LFM_TOOL_CALL_START

_TOOL_BLOCK_RE = re.compile(
    rf"{re.escape(LFM_TOOL_CALL_START)}\s*(.*?)\s*{re.escape(LFM_TOOL_CALL_END)}",
    re.DOTALL,
)
_OPENCLAW_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
# Bare pythonic calls: name(arg=..., arg2=...) possibly dotted names.
_PYTHONIC_CALL_RE = re.compile(
    r"([A-Za-z_][\w.]*)\s*\((.*)\)\s*$",
    re.DOTALL,
)


def _literal(raw: str) -> Any:
    raw = raw.strip()
    if not raw:
        return ""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    try:
        value = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        if (raw.startswith('"') and raw.endswith('"')) or (
            raw.startswith("'") and raw.endswith("'")
        ):
            return raw[1:-1]
        return raw
    return _jsonable(value)


def _jsonable(value: Any) -> Any:
    """Normalize AST literals to JSON-friendly types (sets → sorted lists)."""
    if isinstance(value, set):
        return sorted((_jsonable(v) for v in value), key=lambda x: str(x))
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _split_kwargs(args_str: str) -> dict[str, Any]:
    """Split ``a=1, b="x", c=[1,2]`` into a dict without executing code."""
    args_str = args_str.strip()
    if not args_str:
        return {}
    # Prefer ast parsing of a fake call for nested structures.
    try:
        tree = ast.parse(f"f({args_str})", mode="eval")
        call = tree.body
        if isinstance(call, ast.Call):
            out: dict[str, Any] = {}
            for kw in call.keywords:
                if kw.arg is None:
                    continue
                out[kw.arg] = _jsonable(ast.literal_eval(kw.value))
            return out
    except (SyntaxError, ValueError):
        pass

    out = {}
    depth = 0
    in_str: str | None = None
    start = 0
    i = 0
    while i <= len(args_str):
        ch = args_str[i] if i < len(args_str) else ","
        if in_str:
            if ch == in_str and (i == 0 or args_str[i - 1] != "\\"):
                in_str = None
        else:
            if ch in "\"'":
                in_str = ch
            elif ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                part = args_str[start:i].strip()
                if part and "=" in part:
                    key, raw = part.split("=", 1)
                    out[key.strip()] = _literal(raw)
                start = i + 1
        i += 1
    return out


def _calls_from_pythonic_inner(inner: str) -> list[dict[str, Any]]:
    """Parse ``func(a=1)`` or ``func1(...), func2(...)`` (possibly bracket-wrapped)."""
    inner = inner.strip()
    if inner.startswith("[") and inner.endswith("]"):
        inner = inner[1:-1].strip()
    if not inner:
        return []

    # JSON list/object of tool calls
    if inner.startswith("{") or inner.startswith("["):
        try:
            payload = json.loads(inner)
            return _calls_from_json_payload(payload)
        except json.JSONDecodeError:
            pass

    calls: list[dict[str, Any]] = []
    # Split top-level comma-separated calls carefully.
    depth = 0
    in_str: str | None = None
    start = 0
    for i, ch in enumerate(inner + ","):
        if in_str:
            if ch == in_str and (i == 0 or inner[i - 1] != "\\"):
                in_str = None
            continue
        if ch in "\"'":
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            chunk = inner[start:i].strip()
            start = i + 1
            if not chunk:
                continue
            match = _PYTHONIC_CALL_RE.match(chunk)
            if not match:
                continue
            name = match.group(1)
            args = _split_kwargs(match.group(2))
            calls.append({name: args})
    return calls


def _calls_from_json_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        # {"name": "...", "arguments": {...}} or {func: {args}}
        if "name" in payload:
            args = payload.get("arguments", payload.get("parameters", {}))
            if not isinstance(args, dict):
                args = {}
            return [{str(payload["name"]): args}]
        # Already BFCL-shaped single call
        if len(payload) == 1:
            key = next(iter(payload))
            val = payload[key]
            if isinstance(val, dict):
                return [{str(key): val}]
        return []
    if isinstance(payload, list):
        out: list[dict[str, Any]] = []
        for item in payload:
            out.extend(_calls_from_json_payload(item))
        return out
    return []


def parse_function_calls(text: str) -> list[dict[str, Any]]:
    """Extract function calls from a model completion (any supported format)."""
    if not text or not text.strip():
        return []

    calls: list[dict[str, Any]] = []

    for match in _TOOL_BLOCK_RE.finditer(text):
        calls.extend(_calls_from_pythonic_inner(match.group(1)))

    for match in _OPENCLAW_BLOCK_RE.finditer(text):
        inner = match.group(1).strip()
        try:
            calls.extend(_calls_from_json_payload(json.loads(inner)))
        except json.JSONDecodeError:
            calls.extend(_calls_from_pythonic_inner(inner))

    if calls:
        return calls

    # Fallback: entire response is a bare call / JSON / BFCL list
    stripped = text.strip()
    # Prefer content inside the first [...] that looks like calls
    bracket = re.search(r"\[.+\]", stripped, re.DOTALL)
    if bracket:
        parsed = _calls_from_pythonic_inner(bracket.group(0))
        if parsed:
            return parsed

    try:
        return _calls_from_json_payload(json.loads(stripped))
    except json.JSONDecodeError:
        pass

    return _calls_from_pythonic_inner(stripped)
