"""Unit tests for BFCL train data / reward / configs — no model weights."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdft.bfcl.ast_score import score_bfcl_example
from sdft.bfcl.data import load_bfcl_category
from sdft.bfcl.parse import parse_function_calls
from sdft.bfcl.train_data import (
    IRRELEVANCE_GOLD,
    build_grpo_row,
    gold_response_for_row,
    ground_truth_to_model_calls,
    is_rendered_prompt,
    load_bfcl_train_eval_split,
    model_calls_to_lfm_tool_text,
    pick_arg_value,
    split_category_rows,
)
from sdft.config import load_config
from sdft.rewards import bfcl_reward, resolve_reward


def test_pick_arg_value_skips_optional_empty():
    assert pick_arg_value([10, ""]) == 10
    assert pick_arg_value(["units", ""]) == "units"
    assert pick_arg_value(["", ""]) is None
    assert pick_arg_value(3) == 3


def test_ground_truth_to_lfm_tool_text_roundtrip():
    gt = [
        {
            "calculate_triangle_area": {
                "base": [10],
                "height": [5],
                "unit": ["units", ""],
            }
        }
    ]
    calls = ground_truth_to_model_calls(gt)
    assert calls == [{"calculate_triangle_area": {"base": 10, "height": 5, "unit": "units"}}]
    text = model_calls_to_lfm_tool_text(calls)
    assert "<|tool_call_start|>" in text
    parsed = parse_function_calls(text)
    assert parsed == calls
    ok = score_bfcl_example(
        category="simple",
        model_calls=parsed,
        ground_truth=gt,
        functions=[
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
        ],
    )
    assert ok["acc"] is True


def test_irrelevance_gold_has_no_tool_calls():
    row = {"category": "irrelevance", "ground_truth": None, "function": []}
    text = gold_response_for_row(row)
    assert text == IRRELEVANCE_GOLD
    assert parse_function_calls(text) == []


def test_split_category_rows_no_overlap():
    rows = [{"id": f"simple_{i}"} for i in range(10)]
    train, ev = split_category_rows(rows, num_eval=3, num_train=4)
    assert [r["id"] for r in ev] == ["simple_0", "simple_1", "simple_2"]
    assert [r["id"] for r in train] == ["simple_3", "simple_4", "simple_5", "simple_6"]
    assert {r["id"] for r in train}.isdisjoint({r["id"] for r in ev})


def test_load_bfcl_train_eval_split_guard(tmp_path):
    cache = Path(__file__).resolve().parents[1] / "data" / "bfcl"
    if not (cache / "BFCL_v3_simple.json").is_file():
        pytest.skip("BFCL cache not present")
    split = load_bfcl_train_eval_split(
        categories=("simple", "irrelevance"),
        num_train_per_cat=4,
        num_eval_per_cat=8,
        cache_dir=cache,
    )
    assert len(split["train"]) == 8
    assert len(split["eval"]) == 16
    assert set(split["train_ids"]).isdisjoint(set(split["eval_ids"]))
    # Eval ids match first-N of the raw category file.
    first_simple = load_bfcl_category("simple", cache_dir=cache, num_examples=8)
    assert [r["id"] for r in first_simple] == split["per_category"]["simple"]["eval_ids"]


def test_bfcl_reward_scores_ast():
    functions = [
        {
            "name": "calculate_triangle_area",
            "parameters": {
                "type": "dict",
                "properties": {
                    "base": {"type": "integer"},
                    "height": {"type": "integer"},
                },
                "required": ["base", "height"],
            },
        }
    ]
    gt = [{"calculate_triangle_area": {"base": [10], "height": [5]}}]
    good = (
        '<|tool_call_start|>[{"name": "calculate_triangle_area", '
        '"arguments": {"base": 10, "height": 5}}]<|tool_call_end|>'
    )
    bad = (
        '<|tool_call_start|>[{"name": "calculate_triangle_area", '
        '"arguments": {"base": 10, "height": 9}}]<|tool_call_end|>'
    )
    scores = bfcl_reward(
        [good, bad, ""],
        bfcl_category=["simple", "simple", "irrelevance"],
        bfcl_functions=[json.dumps(functions)] * 3,
        bfcl_ground_truth=[json.dumps(gt), json.dumps(gt), ""],
    )
    assert scores[0] == 1.0
    assert scores[1] == -1.0
    assert scores[2] == 1.0  # irrelevance + empty calls


def test_resolve_reward_bfcl():
    assert resolve_reward("bfcl") is bfcl_reward


def test_is_rendered_prompt():
    assert is_rendered_prompt("<|im_start|>user\nhello<|im_end|>")
    assert not is_rendered_prompt("Find the area of a triangle")


def test_load_bfcl_train_configs():
    root = Path(__file__).resolve().parents[1]
    gold = load_config(root / "configs/compare/bfcl_sft_gold.yaml")
    assert gold.training.batch_size == 1
    assert gold.training.output_dir.endswith("bfcl-sft-gold")

    sdft = load_config(root / "configs/compare/bfcl_sdft.yaml")
    assert sdft.training.target_field == "sdft_response"

    grpo = load_config(root / "configs/compare/bfcl_grpo.yaml")
    assert grpo.grpo.reward == "bfcl"
    assert grpo.grpo.num_generations == 2
    assert grpo.grpo.batch_size == 2

    g12 = load_config(root / "configs/compare/bfcl_1_2b_grpo.yaml")
    assert g12.model.name == "LiquidAI/LFM2.5-1.2B-Instruct"
    assert g12.model.dtype == "float16"
    assert g12.grpo.reward == "bfcl"


def test_build_grpo_row_metadata():
    class _Tok:
        def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=True):
            return "<|im_start|>user\nprompt<|im_end|>\n<|im_start|>assistant\n"

    row = {
        "id": "simple_0",
        "category": "simple",
        "question": [[{"role": "user", "content": "Find area"}]],
        "function": [
            {
                "name": "calculate_triangle_area",
                "description": "d",
                "parameters": {
                    "type": "dict",
                    "properties": {"base": {"type": "integer"}, "height": {"type": "integer"}},
                    "required": ["base", "height"],
                },
            }
        ],
        "ground_truth": [{"calculate_triangle_area": {"base": [10], "height": [5]}}],
    }
    out = build_grpo_row(row, _Tok())
    assert out is not None
    assert out["bfcl_category"] == "simple"
    assert "calculate_triangle_area" in out["bfcl_functions"]
    assert is_rendered_prompt(out["prompt"])
    parsed_gt = json.loads(out["bfcl_ground_truth"])
    assert parsed_gt[0]["calculate_triangle_area"]["base"] == [10]
