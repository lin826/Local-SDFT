"""Unit tests for local AlpacaEval judge helpers (no model load)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdft.alpacaeval_local_judge import (
    DEFAULT_LOCAL_JUDGE,
    _should_swap,
    _winrate_from_preferences,
    default_local_judge_model,
    parse_pairwise_verdict,
)
from sdft.alpacaeval_score import resolve_judge_mode


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Output (a)", 1),
        ("Output (b)", 2),
        ("output (A)", 1),
        ("I prefer Output (b) because it is clearer.", 2),
        ("a", 1),
        ("b", 2),
        ("(a)", 1),
        ("(b)", 2),
        ("m", 1),
        ("M", 2),
        ("Output (a)\nOutput (b)", 1),  # first match wins
        ("", None),
        ("neither is good", None),
        ("The better answer is unclear.", None),
    ],
)
def test_parse_pairwise_verdict(text: str, expected: int | None):
    assert parse_pairwise_verdict(text) == expected


def test_default_local_judge_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ALPACA_EVAL_LOCAL_JUDGE", raising=False)
    assert default_local_judge_model() == DEFAULT_LOCAL_JUDGE
    monkeypatch.setenv("ALPACA_EVAL_LOCAL_JUDGE", "Qwen/Qwen2.5-7B-Instruct")
    assert default_local_judge_model() == "Qwen/Qwen2.5-7B-Instruct"


def test_resolve_judge_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("JUDGE", raising=False)
    assert resolve_judge_mode() == "local"
    assert resolve_judge_mode("openai") == "openai"
    monkeypatch.setenv("JUDGE", "openai")
    assert resolve_judge_mode() == "openai"
    with pytest.raises(ValueError, match="unknown judge"):
        resolve_judge_mode("claude")


def test_should_swap_deterministic():
    assert _should_swap("hello") == _should_swap("hello")
    # Different instructions should not all swap the same way.
    flips = {_should_swap(f"q{i}") for i in range(40)}
    assert flips == {True, False}


def test_winrate_from_preferences():
    # All model wins (preference=2)
    m = _winrate_from_preferences([2.0, 2.0, 2.0, 2.0])
    assert m["win_rate"] == 100.0
    assert m["n_wins"] == 4
    assert m["n_total"] == 4
    # Mixed: 2 wins, 1 draw, 1 ref win → scores 1+1+0.5+0 = 2.5/4 = 62.5%
    m2 = _winrate_from_preferences([2.0, 2.0, 1.5, 1.0])
    assert m2["win_rate"] == pytest.approx(62.5)
    assert m2["n_draws"] == 1


def test_load_gpt4_turbo_reference_from_json(tmp_path: Path):
    from sdft.alpacaeval_local_judge import load_gpt4_turbo_reference

    path = tmp_path / "ref.json"
    path.write_text(
        json.dumps(
            [
                {
                    "instruction": "Say hi",
                    "output": "Hello!",
                    "generator": "gpt4_1106_preview",
                },
                {
                    "instruction": "Say bye",
                    "output": "Bye!",
                    "generator": "gpt4_1106_preview",
                },
            ]
        ),
        encoding="utf-8",
    )
    rows = load_gpt4_turbo_reference(json_path=path, num_examples=1)
    assert len(rows) == 1
    assert rows[0]["instruction"] == "Say hi"
    assert rows[0]["output"] == "Hello!"
