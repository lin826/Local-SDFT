"""Smoke tests for multi-turn Performance chat UI (mocked model)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sdft.records.schema import PerformanceMetrics, PerformanceResult
from web.app import app


def _fake_chat_result(messages: list[dict[str, str]], run_id: str = "bench-test-chat") -> PerformanceResult:
    full = list(messages) + [{"role": "assistant", "content": f"echo:{messages[-1]['content']}"}]
    last_user = messages[-1]["content"]
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    return PerformanceResult(
        id=run_id,
        run_at="2026-07-19T00:00:00Z",
        benchmark="inference",
        model="mock",
        metrics=PerformanceMetrics(
            latency_ms_mean=12.0,
            latency_ms_p50=12.0,
            latency_ms_p95=12.0,
            tokens_per_second=100.0,
            samples=1,
            batch_size=1,
            input_tokens_total=8,
            output_tokens_total=4,
            device="cpu",
        ),
        metadata={
            "messages": full,
            "examples": [
                {
                    "instruction": system or last_user,
                    "input": last_user if system else "",
                    "output": full[-1]["content"],
                }
            ],
            "max_new_tokens": 64,
            "turn_count": sum(1 for m in messages if m["role"] == "user"),
            "chat": True,
        },
        config_path="configs/default.yaml",
    )


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sdft-test'\n", encoding="utf-8")
    (tmp_path / "outputs" / "benchmarks").mkdir(parents=True)
    monkeypatch.setattr(
        "sdft.records.paths.project_root",
        lambda start=None: tmp_path.resolve(),
    )
    with TestClient(app) as c:
        yield c


def test_perf_page_shows_chat_ui(client: TestClient):
    resp = client.get("/perf")
    assert resp.status_code == 200
    assert b'action="/perf/chat"' in resp.content
    assert b"Chat inference" in resp.content
    assert b"Start generate" in resp.content


def test_multi_turn_chat_transcript(client: TestClient):
    call_count = {"n": 0}
    saved: dict[str, PerformanceResult] = {}

    def fake_run(benchmark, **kwargs):
        call_count["n"] += 1
        messages = kwargs["messages"]
        run_id = f"bench-turn-{call_count['n']}"
        result = _fake_chat_result(messages, run_id=run_id)
        from sdft.records.store import append_performance_index, save_performance_result
        from sdft.records.paths import performance_index_path, performance_result_path

        save_performance_result(performance_result_path(result.id), result)
        append_performance_index(performance_index_path(), result)
        saved[result.id] = result
        return result

    with patch("web.app.run_benchmark", side_effect=fake_run):
        r1 = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "instruction": "You are a witty PhD comic narrator.",
                "user_message": "Tell a joke about grading.",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
        assert r1.status_code == 303
        loc1 = r1.headers["location"]
        assert "continue=bench-turn-1" in loc1
        assert "sent=1" in loc1

        page1 = client.get(loc1)
        assert page1.status_code == 200
        body1 = page1.text
        assert "Tell a joke about grading." in body1
        assert "echo:Tell a joke about grading." in body1
        assert "You are a witty PhD comic narrator." in body1

        history = [
            {"role": "user", "content": "Tell a joke about grading."},
            {"role": "assistant", "content": "echo:Tell a joke about grading."},
        ]
        r2 = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "instruction": "You are a witty PhD comic narrator.",
                "user_message": "Make it shorter.",
                "messages_json": json.dumps(history),
            },
            follow_redirects=False,
        )
        assert r2.status_code == 303
        loc2 = r2.headers["location"]
        assert "continue=bench-turn-2" in loc2

        page2 = client.get(loc2)
        assert page2.status_code == 200
        body2 = page2.text
        assert "Tell a joke about grading." in body2
        assert "Make it shorter." in body2
        assert "echo:Make it shorter." in body2

        detail = client.get("/perf/bench-turn-2")
        assert detail.status_code == 200
        assert "Transcript" in detail.text
        assert "Make it shorter." in detail.text
        assert "echo:Make it shorter." in detail.text
        assert "Continue chat" in detail.text

        assert call_count["n"] == 2
        second_msgs = saved["bench-turn-2"].metadata["messages"]
        roles = [m["role"] for m in second_msgs]
        assert roles == ["system", "user", "assistant", "user", "assistant"]


def test_generate_still_background(client: TestClient):
    with patch("web.app.run_benchmark") as mocked:
        r = client.post(
            "/perf/run",
            data={
                "benchmark": "generate",
                "config_path": "configs/default.yaml",
                "num_examples": "2",
            },
            follow_redirects=False,
        )
        assert r.status_code == 303
        assert r.headers["location"] == "/perf?started=1"
        assert mocked.call_count <= 1
