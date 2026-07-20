"""Unit tests for AlpacaEval scoring helpers (no OpenAI / alpaca_eval API calls)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sdft.alpacaeval_score import (
    extract_model_metrics,
    require_openai_api_key,
    to_model_outputs,
    write_model_outputs,
)


def test_to_model_outputs_shape():
    rows = to_model_outputs(
        ["How do I sew a button?", "Make juice"],
        ["Step one…", "Blend apples."],
        generator="sdft",
    )
    assert rows == [
        {
            "instruction": "How do I sew a button?",
            "output": "Step one…",
            "generator": "sdft",
        },
        {
            "instruction": "Make juice",
            "output": "Blend apples.",
            "generator": "sdft",
        },
    ]


def test_to_model_outputs_length_mismatch():
    with pytest.raises(ValueError, match="length mismatch"):
        to_model_outputs(["a"], ["b", "c"], generator="x")


def test_write_model_outputs(tmp_path: Path):
    path = write_model_outputs(
        tmp_path / "out.json",
        to_model_outputs(["Q?"], ["A."], generator="zs"),
    )
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded[0]["generator"] == "zs"
    assert loaded[0]["instruction"] == "Q?"


def test_require_openai_api_key_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        require_openai_api_key()


def test_require_openai_api_key_present(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENAI_API_KEY", " sk-test ")
    assert require_openai_api_key() == "sk-test"


def test_extract_model_metrics_from_dict():
    leaderboard = {
        "sdft": {
            "win_rate": 12.5,
            "standard_error": 1.1,
            "length_controlled_winrate": 10.0,
            "n_total": 16,
            "avg_length": 200,
        }
    }
    metrics = extract_model_metrics(leaderboard, "sdft")
    assert metrics["win_rate"] == 12.5
    assert metrics["length_controlled_winrate"] == 10.0
    assert metrics["n_total"] == 16


def test_extract_model_metrics_missing_name():
    with pytest.raises(KeyError, match="missing"):
        extract_model_metrics({"other": {"win_rate": 1.0}}, "sdft")
