"""Simplified BFCL AST / irrelevance scoring (faithful to possible_answer lists).

Official checker lives in ``bfcl-eval``; this mirrors the core rules for the
local single-turn subset without pulling in that stack.
"""

from __future__ import annotations

from typing import Any


def _normalize_string(value: Any) -> str:
    return str(value).strip().lower()


def _value_matches(actual: Any, possible: list[Any]) -> bool:
    """True if ``actual`` equals any entry in ``possible`` (BFCL-style)."""
    if not possible:
        return False
    # Optional param marker: empty string means "may omit" — handled by caller.
    candidates = [p for p in possible if p != ""]
    if not candidates:
        return True

    for cand in candidates:
        if isinstance(cand, bool) or isinstance(actual, bool):
            if bool(actual) is bool(cand) and type(actual) is type(cand):
                return True
            # bool/int confusion: accept 0/1 vs False/True loosely only if both numeric-ish
            continue
        if isinstance(cand, (int, float)) and isinstance(actual, (int, float)) and not isinstance(
            actual, bool
        ):
            if float(actual) == float(cand):
                return True
            continue
        if isinstance(cand, str) or isinstance(actual, str):
            if _normalize_string(actual) == _normalize_string(cand):
                return True
            continue
        if actual == cand:
            return True
        # Nested list/dict: exact equality after JSON-ish normalization
        if isinstance(cand, (list, dict)) and actual == cand:
            return True
    return False


def _call_name(call: dict[str, Any]) -> str | None:
    if len(call) != 1:
        return None
    return next(iter(call))


def _score_one_call(
    model_call: dict[str, Any],
    possible_call: dict[str, Any],
    func_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    name = _call_name(model_call)
    expected_name = _call_name(possible_call)
    if name is None or expected_name is None:
        return {"valid": False, "error": "malformed_call"}
    if name != expected_name:
        return {"valid": False, "error": "wrong_func_name", "got": name, "want": expected_name}

    actual_args = model_call[name] or {}
    possible_args = possible_call[expected_name] or {}
    if not isinstance(actual_args, dict) or not isinstance(possible_args, dict):
        return {"valid": False, "error": "args_not_dict"}

    required: list[str] = []
    if func_schema:
        params = func_schema.get("parameters") or {}
        required = list(params.get("required") or [])

    for key in required:
        if key not in actual_args:
            return {"valid": False, "error": "missing_required", "param": key}

    for key, possibles in possible_args.items():
        if not isinstance(possibles, list):
            possibles = [possibles]
        if key not in actual_args:
            # Omission allowed when "" is among possible answers (optional param).
            if "" in possibles:
                continue
            return {"valid": False, "error": "missing_param", "param": key}
        if not _value_matches(actual_args[key], possibles):
            return {
                "valid": False,
                "error": "wrong_param_value",
                "param": key,
                "got": actual_args[key],
                "want": possibles,
            }

    # Params not listed in possible_answer / schema are unexpected.
    schema_props: set[str] = set()
    if func_schema:
        props = (func_schema.get("parameters") or {}).get("properties") or {}
        schema_props = set(props.keys())
    allowed = set(possible_args) | schema_props
    for key in actual_args:
        if allowed and key not in allowed:
            return {"valid": False, "error": "unexpected_param", "param": key}

    return {"valid": True, "error": None}


def _find_schema(functions: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for fn in functions:
        if fn.get("name") == name:
            return fn
    return None


def _match_unordered(
    model_calls: list[dict[str, Any]],
    possible_answers: list[dict[str, Any]],
    functions: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(model_calls) != len(possible_answers):
        return {
            "valid": False,
            "error": "wrong_count",
            "got": len(model_calls),
            "want": len(possible_answers),
        }
    remaining = list(range(len(possible_answers)))
    for mcall in model_calls:
        matched = False
        for idx in list(remaining):
            schema = _find_schema(functions, _call_name(mcall) or "")
            result = _score_one_call(mcall, possible_answers[idx], schema)
            if result["valid"]:
                remaining.remove(idx)
                matched = True
                break
        if not matched:
            return {"valid": False, "error": "no_match_for_call", "call": mcall}
    return {"valid": True, "error": None}


def score_bfcl_example(
    *,
    category: str,
    model_calls: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]] | None,
    functions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Return ``{acc: bool, error: str|None, n_calls: int}`` for one example."""
    if category == "irrelevance":
        ok = len(model_calls) == 0
        return {
            "acc": ok,
            "error": None if ok else "unexpected_function_call",
            "n_calls": len(model_calls),
        }

    if ground_truth is None:
        return {"acc": False, "error": "missing_ground_truth", "n_calls": len(model_calls)}

    if category in ("parallel", "parallel_multiple"):
        result = _match_unordered(model_calls, ground_truth, functions)
        return {"acc": bool(result["valid"]), "error": result.get("error"), "n_calls": len(model_calls)}

    if category == "multiple":
        # Exactly one call; must match the single possible-answer entry.
        if len(model_calls) != 1:
            return {
                "acc": False,
                "error": "wrong_count",
                "n_calls": len(model_calls),
            }
        if len(ground_truth) != 1:
            # Some GT rows may list one preferred call.
            pass
        schema = _find_schema(functions, _call_name(model_calls[0]) or "")
        result = _score_one_call(model_calls[0], ground_truth[0], schema)
        return {"acc": bool(result["valid"]), "error": result.get("error"), "n_calls": 1}

    # simple (and default): exactly one function call
    if len(model_calls) != 1:
        return {"acc": False, "error": "wrong_count", "n_calls": len(model_calls)}
    if not ground_truth:
        return {"acc": False, "error": "empty_ground_truth", "n_calls": 1}
    schema = _find_schema(functions, _call_name(model_calls[0]) or "")
    result = _score_one_call(model_calls[0], ground_truth[0], schema)
    return {"acc": bool(result["valid"]), "error": result.get("error"), "n_calls": 1}
