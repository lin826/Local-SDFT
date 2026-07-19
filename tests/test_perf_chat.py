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
    assert b'hx-post="/perf/chat"' in resp.content
    assert b'hx-target="#chat-panel"' in resp.content
    assert b'id="chat-panel"' in resp.content
    assert b"htmx.org" in resp.content
    assert b"Chat inference" in resp.content
    assert b"Demo condition" in resp.content
    assert b"name=\"demo_condition\"" in resp.content
    assert b"Start generate" in resp.content
    assert b"without a full page reload" in resp.content


def test_htmx_chat_returns_partial_not_redirect(client: TestClient):
    def fake_measure_chat(cfg, messages, **kwargs):
        return _fake_chat_result(messages, run_id="bench-htmx-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "Be brief.",
                "user_message": "Hello HTMX",
                "messages_json": "[]",
            },
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )

    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "location" not in {k.lower() for k in resp.headers.keys()} or resp.headers.get("location") is None
    body = resp.text
    assert 'id="chat-panel"' in body
    assert "Hello HTMX" in body
    assert "echo:Hello HTMX" in body
    assert 'name="messages_json"' in body
    assert "bench-htmx-1" in body
    assert "<html" not in body.lower()
    assert resp.headers.get("HX-Push-Url")
    assert "continue=bench-htmx-1" in resp.headers["HX-Push-Url"]
    assert "sent=1" in resp.headers["HX-Push-Url"]


def test_multi_turn_chat_transcript(client: TestClient):
    call_count = {"n": 0}
    saved: dict[str, PerformanceResult] = {}

    def fake_measure_chat(cfg, messages, **kwargs):
        call_count["n"] += 1
        run_id = f"bench-turn-{call_count['n']}"
        result = _fake_chat_result(messages, run_id=run_id)
        from sdft.records.store import append_performance_index, save_performance_result
        from sdft.records.paths import performance_index_path, performance_result_path

        save_performance_result(performance_result_path(result.id), result)
        append_performance_index(performance_index_path(), result)
        saved[result.id] = result
        return result

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", side_effect=lambda r: saved.setdefault(r.id, r)
    ):
        r1 = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
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
        assert "condition=plain" in loc1

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
                "demo_condition": "plain",
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


def test_sdft_condition_rejected_without_checkpoint(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("web.app.merged_checkpoint_available", lambda: False)
    resp = client.post(
        "/perf/chat",
        data={
            "config_path": "configs/openclaw_demo_eval.yaml",
            "demo_condition": "SDFT-ZS",
            "instruction": "",
            "user_message": "What is 2+2?",
            "messages_json": "[]",
        },
    )
    assert resp.status_code == 400
    assert "SDFT checkpoint missing" in resp.json()["detail"]


def test_toolcall_condition_uses_measure_toolcall_chat(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    captured: dict[str, object] = {}

    def fake_toolcall(cfg, messages, **kwargs):
        captured["kwargs"] = kwargs
        captured["model"] = cfg.model.name
        return _fake_chat_result(messages, run_id="bench-tool-zs")

    monkeypatch.setattr("web.app.measure_toolcall_chat", fake_toolcall)
    monkeypatch.setattr(
        "web.app.persist_performance_result",
        lambda result: None,
    )

    with patch("web.app.load_config") as load_cfg:
        from sdft.config import load_config as real_load

        cfg = real_load("configs/openclaw_demo_eval.yaml")
        load_cfg.return_value = cfg

        r = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/openclaw_demo_eval.yaml",
                "demo_condition": "OS+CoT",
                "instruction": "ignored",
                "user_message": "What is 3+5?",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert r.status_code == 303
    assert "condition=OS%2BCoT" in r.headers["location"] or "condition=OS+CoT" in r.headers["location"]
    assert captured["kwargs"]["few_shot_k"] == 1
    assert captured["kwargs"]["cot_line"] is not None
    assert captured["kwargs"]["demo_condition"] == "OS+CoT"
    assert captured["model"] == "LiquidAI/LFM2.5-230M"
