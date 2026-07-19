"""Smoke tests for multi-turn Performance chat UI (mocked model)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sdft.records.schema import PerformanceMetrics, PerformanceResult
from web.app import CONFIG_OPTIONS, app
from web.demo_conditions import build_design_summary


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
    body = resp.content
    assert b'action="/perf/chat"' in body
    assert b'hx-post="/perf/chat"' in body
    assert b'hx-target="#chat-panel"' in body
    assert b'id="chat-panel"' in body
    assert b"htmx.org" in body
    assert b"Plain chat inference" in body
    assert b'name="demo_condition"' in body
    assert b'value="plain"' in body
    assert b"configs/lfm25_alpacaeval2_trained.yaml" in body
    assert b"configs/default.yaml" in body
    assert b"openclaw" not in body.lower()
    assert b'data-toolcall=' not in body
    assert b"syncInstructionField()" not in body
    assert b"Start generate" in body
    assert b"without a full page reload" not in body  # removed OpenClaw-specific howto line


def test_config_options_include_alpacaeval_sdft():
    assert CONFIG_OPTIONS == [
        "configs/default.yaml",
        "configs/lfm25_alpacaeval2_trained.yaml",
    ]
    assert "configs/openclaw_demo_eval.yaml" not in CONFIG_OPTIONS
    assert "configs/geek_jokes.yaml" not in CONFIG_OPTIONS
    assert "configs/geek_jokes_trained.yaml" not in CONFIG_OPTIONS
    assert "configs/geek_jokes_bench.yaml" not in CONFIG_OPTIONS


def test_build_design_summary_variants():
    base = build_design_summary(
        demo_condition="plain",
        config_path="configs/default.yaml",
        model_path="LiquidAI/LFM2.5-230M",
    )
    assert "base LFM2.5-230M" in base["variant"]
    assert "no GPT-4 judge" in base["eval_surface"]

    sdft = build_design_summary(
        demo_condition="plain",
        config_path="configs/lfm25_alpacaeval2_trained.yaml",
        model_path="outputs/lfm25-230m-alpacaeval2-sdft-merged",
    )
    assert "SDFT merge" in sdft["variant"]
    assert sdft["config_path"] == "configs/lfm25_alpacaeval2_trained.yaml"


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


def test_chat_persists_design_summary(client: TestClient):
    saved: dict[str, PerformanceResult] = {}

    def fake_measure_chat(cfg, messages, **kwargs):
        return _fake_chat_result(messages, run_id="bench-design-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result",
        side_effect=lambda r: saved.setdefault(r.id, r),
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/lfm25_alpacaeval2_trained.yaml",
                "demo_condition": "plain",
                "instruction": "",
                "user_message": "How do I sew a button?",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    meta = saved["bench-design-1"].metadata
    assert "design_summary" in meta
    assert meta["design_summary"]["config_path"] == "configs/lfm25_alpacaeval2_trained.yaml"
    assert "SDFT merge" in meta["design_summary"]["variant"]


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


def test_run_detail_shows_design_summary(client: TestClient):
    from sdft.records.store import save_performance_result
    from sdft.records.paths import performance_result_path

    result = _fake_chat_result(
        [{"role": "user", "content": "How do I make apple juice?"}],
        run_id="bench-detail-design",
    )
    result.metadata["design_summary"] = build_design_summary(
        demo_condition="plain",
        config_path="configs/default.yaml",
        model_path="LiquidAI/LFM2.5-230M",
    )
    save_performance_result(performance_result_path(result.id), result)

    detail = client.get(f"/perf/{result.id}")
    assert detail.status_code == 200
    assert "Design summary" in detail.text
    assert "apple juice" in detail.text or "AlpacaEval-style" in detail.text
    assert "eval surface" in detail.text.lower()


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


def test_unknown_demo_condition_rejected(client: TestClient):
    resp = client.post(
        "/perf/chat",
        data={
            "config_path": "configs/default.yaml",
            "demo_condition": "ZS",
            "instruction": "",
            "user_message": "Hello",
            "messages_json": "[]",
        },
    )
    assert resp.status_code == 400
    assert "unknown demo condition" in resp.json()["detail"]
