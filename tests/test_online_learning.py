"""Tests for online-learning metadata, stats, and web routes (mocked train/infer)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sdft.online_learning import (
    aggregate_turn_latencies,
    create_session,
    load_session,
    run_online_turn,
)
from sdft.online_learning.schema import OnlineTurn, TurnLatency
from sdft.online_learning.stats import turn_latency_from_phases
from sdft.records.paths import collected_records_path
from sdft.records.store import load_collected_records
from web.app import app


@pytest.fixture()
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'sdft-test'\n", encoding="utf-8")
    (tmp_path / "data" / "collected").mkdir(parents=True)
    (tmp_path / "outputs" / "online-learning").mkdir(parents=True)
    monkeypatch.setattr(
        "sdft.records.paths.project_root",
        lambda start=None: tmp_path.resolve(),
    )
    monkeypatch.setattr(
        "sdft.online_learning.paths.project_root",
        lambda start=None: tmp_path.resolve(),
    )
    with TestClient(app) as c:
        yield c


def test_aggregate_turn_latencies_stats():
    turns = [
        OnlineTurn(
            turn_index=1,
            instruction="a",
            input="",
            output="b",
            sdft_response="sdft-b",
            record_id="rec-1",
            created_at="t1",
            latency=TurnLatency(total_ms=3000, generate_ms=800, inference_ms=200, train_ms=2000),
        ),
        OnlineTurn(
            turn_index=2,
            instruction="c",
            input="",
            output="d",
            sdft_response="sdft-d",
            record_id="rec-2",
            created_at="t2",
            latency=TurnLatency(total_ms=5000, generate_ms=1200, inference_ms=300, train_ms=3500),
        ),
    ]
    summary = aggregate_turn_latencies(turns)
    assert summary["turn_count"] == 2
    assert summary["generate_ms"]["count"] == 2
    assert summary["generate_ms"]["mean"] == 1000.0
    assert summary["inference_ms"]["count"] == 2
    assert summary["inference_ms"]["mean"] == 250.0
    assert summary["train_ms"]["p50"] == 2000.0
    assert summary["total_ms"]["p95"] == 5000.0


def test_turn_latency_from_phases():
    phases = [
        {"name": "generate_sdft", "start_ms": 0, "end_ms": 800, "duration_ms": 800},
        {"name": "train_update", "start_ms": 800, "end_ms": 2800, "duration_ms": 2000},
        {"name": "inference_preview", "start_ms": 2800, "end_ms": 3000, "duration_ms": 200},
        {"name": "record_collect", "start_ms": 3000, "end_ms": 3005, "duration_ms": 5},
    ]
    lat = turn_latency_from_phases(phases, input_tokens=12, output_tokens=34)
    assert lat.generate_ms == 800.0
    assert lat.inference_ms == 200.0
    assert lat.train_ms == 2000.0
    assert lat.total_ms == 3005.0
    assert lat.input_tokens == 12
    assert lat.output_tokens == 34


def test_run_online_turn_persists_session_and_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "data" / "collected").mkdir(parents=True)
    monkeypatch.setattr("sdft.records.paths.project_root", lambda start=None: tmp_path.resolve())
    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())

    session = create_session("configs/online_learning.yaml")

    def fake_generate(cfg, *, instruction, user_input="", fewshot_examples=None, **kwargs):
        return "sdft rewrite text", 10, 5

    def fake_preview(cfg, adapter_dir, instruction, user_input="", **kwargs):
        return "preview text", 8, 4

    def fake_train(cfg, adapter_dir, examples, **kwargs):
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert all("sdft_response" in ex for ex in examples)
        assert all("output" not in ex for ex in examples)

    with patch("sdft.online_learning.loop.generate_sdft_response", side_effect=fake_generate), patch(
        "sdft.online_learning.loop.generate_preview", side_effect=fake_preview
    ), patch("sdft.online_learning.loop.run_train_step", side_effect=fake_train):
        turn = run_online_turn(
            session.id,
            instruction="Tell a joke",
            output="Why did the PhD cross the road?",
            preview=True,
        )

    loaded = load_session(session.id)
    assert loaded.turn_count == 1
    assert loaded.latency_summary["turn_count"] == 1
    assert loaded.latency_summary["generate_ms"]["count"] == 1
    assert loaded.latency_summary["train_ms"]["count"] == 1
    assert turn.sdft_response == "sdft rewrite text"
    assert turn.output == "Why did the PhD cross the road?"
    assert turn.preview == "preview text"
    assert turn.latency.generate_ms is not None
    assert turn.latency.inference_ms is not None
    assert turn.latency.train_ms is not None

    records = load_collected_records(collected_records_path())
    assert len(records) == 1
    assert records[0].metadata["online_session_id"] == session.id
    assert records[0].metadata["sdft_response"] == "sdft rewrite text"
    assert "latency" in records[0].metadata
    assert records[0].metadata["latency"]["total_ms"] > 0


def test_data_page_shows_online_learning_ui(client: TestClient):
    resp = client.get("/data?new=1")
    assert resp.status_code == 200
    body = resp.text
    assert "Online learning" in body
    assert "tiny SDFT" in body
    assert 'action="/data/turn"' in body
    assert "configs/online_learning.yaml" in body
    assert "Running tiny SDFT update" in body
    assert "collection only" in body


def test_online_turn_route_mocked(client: TestClient):
    page = client.get("/data?new=1")
    assert page.status_code == 200
    session_id = page.text.split("session_id")[1].split('value="')[1].split('"')[0]

    def fake_run(session_id, **kwargs):
        from sdft.online_learning.schema import OnlineTurn, TurnLatency
        from sdft.online_learning.session import load_session, save_session
        from sdft.records.paths import utc_now_iso

        session = load_session(session_id)
        turn = OnlineTurn(
            turn_index=1,
            instruction=kwargs["instruction"],
            input=kwargs.get("input_text") or kwargs.get("input") or "",
            output=kwargs.get("output") or "",
            sdft_response="mock sdft target",
            preview="mock preview",
            record_id="rec-test-1",
            created_at=utc_now_iso(),
            latency=TurnLatency(
                total_ms=1234,
                generate_ms=500,
                inference_ms=400,
                train_ms=300,
                output_tokens=7,
            ),
            latency_phases=[
                {"name": "generate_sdft", "start_ms": 0, "end_ms": 500, "duration_ms": 500},
                {"name": "train_update", "start_ms": 500, "end_ms": 800, "duration_ms": 300},
                {"name": "inference_preview", "start_ms": 800, "end_ms": 1200, "duration_ms": 400},
            ],
        )
        session.turns.append(turn)
        save_session(session)
        return turn

    with patch("web.app.run_online_turn", side_effect=fake_run):
        resp = client.post(
            "/data/turn",
            data={
                "session_id": session_id,
                "config_path": "configs/online_learning.yaml",
                "instruction": "Hello",
                "preview": "1",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert f"session={session_id}" in resp.headers["location"]
    assert "turn=1" in resp.headers["location"]

    detail = client.get(f"/data/{session_id}")
    assert detail.status_code == 200
    assert "Latency summary" in detail.text
    assert "SDFT generate" in detail.text
    assert "Per-turn latencies" in detail.text


def test_online_session_detail_404(client: TestClient):
    resp = client.get("/data/ol-does-not-exist")
    assert resp.status_code == 404
