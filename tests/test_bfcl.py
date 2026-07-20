"""Unit tests for local BFCL subset (parser / scorer / config) — no model weights."""

from __future__ import annotations

from pathlib import Path

from sdft.bfcl.ast_score import score_bfcl_example
from sdft.bfcl.data import extract_user_text, functions_to_tools
from sdft.bfcl.parse import parse_function_calls
from sdft.config import load_config


def test_parse_lfm_pythonic_tool_call():
    text = (
        "<|tool_call_start|>[calculate_triangle_area(base=10, height=5)]"
        "<|tool_call_end|>"
    )
    calls = parse_function_calls(text)
    assert calls == [{"calculate_triangle_area": {"base": 10, "height": 5}}]


def test_parse_parallel_pythonic_calls():
    text = (
        "<|tool_call_start|>["
        'spotify.play(artist="Taylor Swift", duration=20), '
        'spotify.play(artist="Maroon 5", duration=15)'
        "]<|tool_call_end|>"
    )
    calls = parse_function_calls(text)
    assert len(calls) == 2
    assert calls[0] == {"spotify.play": {"artist": "Taylor Swift", "duration": 20}}
    assert calls[1] == {"spotify.play": {"artist": "Maroon 5", "duration": 15}}


def test_parse_lfm_json_tool_call():
    text = (
        '<|tool_call_start|>[{"name": "calc", "arguments": {"x": 1}}]'
        "<|tool_call_end|>"
    )
    assert parse_function_calls(text) == [{"calc": {"x": 1}}]


def test_parse_openclaw_tool_call():
    text = '<tool_call>{"name": "foo", "arguments": {"a": "b"}}</tool_call>'
    assert parse_function_calls(text) == [{"foo": {"a": "b"}}]


def test_score_simple_ast_match():
    functions = [
        {
            "name": "calculate_triangle_area",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer"},
                    "height": {"type": "integer"},
                    "unit": {"type": "string"},
                },
                "required": ["base", "height"],
            },
        }
    ]
    gt = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    ok = score_bfcl_example(
        category="simple",
        model_calls=[{"calculate_triangle_area": {"base": 10, "height": 5}}],
        ground_truth=gt,
        functions=functions,
    )
    assert ok["acc"] is True

    bad = score_bfcl_example(
        category="simple",
        model_calls=[{"calculate_triangle_area": {"base": 10, "height": 9}}],
        ground_truth=gt,
        functions=functions,
    )
    assert bad["acc"] is False
    assert bad["error"] == "wrong_param_value"


def test_score_irrelevance():
    ok = score_bfcl_example(
        category="irrelevance",
        model_calls=[],
        ground_truth=None,
        functions=[{"name": "bmi"}],
    )
    assert ok["acc"] is True
    bad = score_bfcl_example(
        category="irrelevance",
        model_calls=[{"bmi": {"weight": 70, "height": 1.8}}],
        ground_truth=None,
        functions=[{"name": "bmi"}],
    )
    assert bad["acc"] is False


def test_score_parallel_unordered():
    functions = [
        {
            "name": "spotify.play",
            "parameters": {
                "properties": {
                    "artist": {"type": "string"},
                    "duration": {"type": "integer"},
                },
                "required": ["artist", "duration"],
            },
        }
    ]
    gt = [
        {"spotify.play": {"artist": ["Taylor Swift"], "duration": [20]}},
        {"spotify.play": {"artist": ["Maroon 5"], "duration": [15]}},
    ]
    # Reverse order should still match.
    calls = [
        {"spotify.play": {"artist": "Maroon 5", "duration": 15}},
        {"spotify.play": {"artist": "Taylor Swift", "duration": 20}},
    ]
    result = score_bfcl_example(
        category="parallel",
        model_calls=calls,
        ground_truth=gt,
        functions=functions,
    )
    assert result["acc"] is True


def test_extract_user_text_and_tools():
    question = [[{"role": "user", "content": "Find area"}]]
    assert extract_user_text(question) == "Find area"
    tools = functions_to_tools(
        [
            {
                "name": "f",
                "description": "d",
                "parameters": {"type": "dict", "properties": {}, "required": []},
            }
        ]
    )
    assert tools[0]["parameters"]["type"] == "object"


def test_load_bfcl_configs():
    root = Path(__file__).resolve().parents[1]
    cfg230 = load_config(root / "configs/bfcl_eval.yaml")
    assert cfg230.model.name == "LiquidAI/LFM2.5-230M"
    assert "simple" in cfg230.bfcl_eval.categories
    assert cfg230.bfcl_eval.num_examples == 32

    cfg12 = load_config(root / "configs/bfcl_eval_1_2b.yaml")
    assert cfg12.model.name == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert cfg12.model.dtype == "float16"

    suite = load_config(root / "configs/compare/batch1_1_2b_sdft.yaml")
    assert suite.training.batch_size == 1
    assert "self_attn" in suite.lora.target_modules

    bfcl_train = load_config(root / "configs/compare/bfcl_grpo.yaml")
    assert bfcl_train.grpo.reward == "bfcl"


def test_parse_empty_is_no_calls():
    assert parse_function_calls("") == []
    assert parse_function_calls("I cannot help with that.") == []
