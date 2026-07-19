"""Unit tests for tool-call parse/format (no model weights)."""

from __future__ import annotations

import pytest

from sdft.toolcall.format import (
    ToolCallFormat,
    build_openclaw_prompt,
    format_tool_observation,
    parse_assistant_action,
    postprocess_assistant_text,
)
from sdft.toolcall.loop import default_few_shot_messages
from sdft.toolcall.sandbox import execute_code_interpreter
from sdft.toolcall.scoring import extract_boxed_answer, score_openclaw_solution


def test_parse_openclaw_json_tool_call():
    text = (
        'Let me compute.\n<tool_call>\n'
        '{"name": "code_interpreter", "arguments": {"code": "print(6*7)"}}\n'
        "</tool_call>"
    )
    action, content = parse_assistant_action(text)
    assert action == "code"
    assert content == "print(6*7)"


def test_parse_lfm_pythonic_tool_call():
    text = '<|tool_call_start|>[code_interpreter(code="print(1+1)")]<|tool_call_end|>'
    action, content = parse_assistant_action(text)
    assert action == "code"
    assert content == "print(1+1)"


def test_parse_lfm_json_tool_call():
    text = (
        '<|tool_call_start|>[{"name": "code_interpreter", '
        '"arguments": {"code": "2+2"}}]<|tool_call_end|>'
    )
    action, content = parse_assistant_action(text)
    assert action == "code"
    assert "2+2" in content


def test_parse_answer_boxed():
    text = r"Therefore Answer: \boxed{\dfrac{10}{3}}"
    action, content = parse_assistant_action(text)
    assert action == "answer"
    assert content == r"\dfrac{10}{3}"


def test_postprocess_trims_after_tool_call():
    raw = (
        '<tool_call>{"name": "code_interpreter", "arguments": {"code": "1"}}</tool_call>'
        " extra hallucination"
    )
    trimmed = postprocess_assistant_text(raw)
    assert trimmed.endswith("</tool_call>")
    assert "hallucination" not in trimmed


def test_build_openclaw_prompt_includes_tools():
    prompt = build_openclaw_prompt(
        user_prompt="What is 2+2?",
        tools=[{"type": "function", "function": {"name": "code_interpreter"}}],
        system_prompt="You are helpful.",
    )
    assert "<tools>" in prompt
    assert "code_interpreter" in prompt
    assert "What is 2+2?" in prompt
    assert "<|im_start|>assistant" in prompt


def test_format_tool_observation_openclaw():
    obs = format_tool_observation("42", fmt=ToolCallFormat.OPENCLAW)
    assert "<interpreter>" in obs
    assert "42" in obs


def test_sandbox_executes_safe_code():
    out = execute_code_interpreter("print(2 + 2)")
    assert "4" in out


def test_sandbox_blocks_os_import():
    out = execute_code_interpreter("import os\nprint(os.getcwd())")
    assert out.startswith("Error:")


def test_score_openclaw_solution_correct():
    solution = r"Step by step... Answer: \boxed{73}"
    result = score_openclaw_solution(solution, "73")
    assert result["acc"] is True
    assert result["score"] == 1.0
    assert result["pred"] == "73"


def test_score_openclaw_solution_wrong():
    solution = r"Answer: \boxed{42}"
    result = score_openclaw_solution(solution, "73")
    assert result["acc"] is False
    assert result["score"] == -1.0


def test_parse_dollar_boxed_answer():
    text = r"Thus the answer is $\boxed{0}$"
    action, content = parse_assistant_action(text)
    assert action == "answer"
    assert content == "0"


def test_score_dollar_boxed():
    solution = r"Therefore $\boxed{73}$"
    result = score_openclaw_solution(solution, "73", strict_box_verify=False)
    assert result["acc"] is True

    text = r"Answer: \boxed{\dfrac{1}{2}}"
    assert extract_boxed_answer(text) == r"\dfrac{1}{2}"


def test_default_few_shot_messages_openclaw():
    msgs = default_few_shot_messages(ToolCallFormat.OPENCLAW, 1)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"
    assert "<tool_call>" in msgs[1]["content"]
    assert r"\boxed{8}" in msgs[1]["content"]


def test_default_few_shot_messages_lfm():
    msgs = default_few_shot_messages(ToolCallFormat.LFM, 1)
    assert any(m["role"] == "tool" for m in msgs)
    assert any(m["role"] == "assistant" and "tool_call_start" in m["content"] for m in msgs)
