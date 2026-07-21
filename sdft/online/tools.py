"""A tiny calculator tool + safe arithmetic evaluation, for the tool-calling demo.

The model learns to answer arithmetic by emitting a tool call:

    <tool>calc("347 + 288")</tool>

We parse the call, evaluate the expression safely (AST, numbers + arithmetic
ops only — never `eval`), and the result is the answer. Because the held-out
problems use numbers that never appear during coaching, getting them right can
only come from learning the *skill* (translate the question into a tool call),
not from memorizing answers.
"""

from __future__ import annotations

import ast
import operator
import re

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# calc("..."), calc('...'), or calc(...) — optionally wrapped in <tool>…</tool>.
_CALL_RE = re.compile(r"calc\(\s*(['\"]?)(?P<expr>[0-9 .+\-*/%x×()]+?)\1\s*\)", re.IGNORECASE)
# A bare arithmetic expression inside a natural-language prompt.
_EXPR_RE = re.compile(r"-?\d+(?:\.\d+)?(?:\s*[-+*/%x×]\s*-?\d+(?:\.\d+)?)+")


def _normalize(expr: str) -> str:
    return expr.replace("×", "*").replace("x", "*").strip()


def safe_eval(expr: str | None) -> float | None:
    """Evaluate an arithmetic expression via AST. Returns None if invalid."""
    if not expr:
        return None
    try:
        node = ast.parse(_normalize(expr), mode="eval").body
        return _ev(node)
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, KeyError, RecursionError):
        return None


def _ev(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("non-numeric constant")
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_ev(node.operand))
    raise ValueError(f"disallowed node: {ast.dump(node)}")


def parse_calc_call(text: str) -> str | None:
    """Return the expression string inside the first calc(...) call, or None."""
    m = _CALL_RE.search(text or "")
    return m.group("expr").strip() if m else None


def extract_arithmetic(text: str) -> str | None:
    """Find the first bare arithmetic expression in a natural-language string."""
    m = _EXPR_RE.search(text or "")
    return m.group(0).strip() if m else None


def run_calc_call(text: str) -> float | None:
    """Parse a calc() tool call from text and evaluate it."""
    return safe_eval(parse_calc_call(text))
