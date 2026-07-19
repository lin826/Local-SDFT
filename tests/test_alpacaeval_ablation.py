"""Unit tests for AlpacaEval ablation prompt building."""

from __future__ import annotations

import pytest

from sdft.alpacaeval_ablation import (
    DEFAULT_COT_LINE,
    apply_cot_to_user,
    build_eval_messages,
    build_perf_chat_messages,
    get_ablation_arm,
    normalize_instruction,
    verify_no_eval_leakage,
)


def test_build_eval_messages_zs_user_only():
    msgs = build_eval_messages("How do I sew a button?", get_ablation_arm("ZS"))
    assert msgs == [{"role": "user", "content": "How do I sew a button?"}]


def test_build_eval_messages_cot_appends_to_user():
    msgs = build_eval_messages("How do I sew a button?", get_ablation_arm("CoT"))
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"].endswith(DEFAULT_COT_LINE)
    assert "system" not in [m["role"] for m in msgs]


def test_build_eval_messages_sys_helpful():
    msgs = build_eval_messages("Hi", get_ablation_arm("SysHelpful"))
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_build_perf_chat_multi_turn_cot_on_latest_user():
    history = [{"role": "user", "content": "First question"}]
    msgs = build_perf_chat_messages(
        get_ablation_arm("CoT"),
        history,
        "Follow up",
    )
    user_contents = [m["content"] for m in msgs if m["role"] == "user"]
    assert all(c.endswith(DEFAULT_COT_LINE) for c in user_contents)


def test_leakage_guard_raises_on_exact_match():
    demo = [{"prompt": "Eval question", "response": "answer"}]
    with pytest.raises(ValueError, match="leakage"):
        verify_no_eval_leakage(demo, ["Eval question"])


def test_normalize_instruction_collapses_whitespace():
    assert normalize_instruction("  Hello \n World ") == "hello world"
