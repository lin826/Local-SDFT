"""Tests for online-learning metadata, tone feedback, stats, and web routes."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sdft.online_learning import (
    aggregate_tone_counts,
    aggregate_turn_latencies,
    build_session,
    build_train_examples,
    classify_tone,
    create_session,
    load_session,
    resolve_tone,
    run_online_turn,
)
from sdft.online_learning.paths import online_session_path
from sdft.online_learning.session import session_persisted
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


def test_classify_tone_heuristics():
    assert classify_tone("Thanks, that was perfect!") == ("positive", 1)
    assert classify_tone("That's wrong, try again") == ("negative", -1)
    assert classify_tone("Tell me another joke") == ("neutral", 0)
    assert classify_tone("yes") == ("positive", 1)
    assert classify_tone("no") == ("negative", -1)


def test_resolve_tone_manual_override():
    tone, reward, source = resolve_tone("anything", override="negative")
    assert tone == "negative"
    assert reward == -1
    assert source == "manual"


def test_build_train_examples_positive_reinforces_prev():
    from sdft.config import load_config

    cfg = load_config("configs/online_learning.yaml")
    prior = [
        OnlineTurn(
            turn_index=1,
            instruction="joke",
            input="",
            output="",
            sdft_response="sdft-1",
            assistant_reply="reply-1",
            record_id="r1",
            created_at="t1",
            latency=TurnLatency(total_ms=1),
        )
    ]
    rows, action, trained_on = build_train_examples(
        cfg,
        prior_turns=prior,
        instruction="Thanks!",
        user_input="",
        sdft_response="sdft-2",
        feedback_tone="positive",
        feedback_reward=1,
    )
    assert action == "reinforce_prev"
    assert any(r["sdft_response"] == "reply-1" for r in rows)
    assert sum(1 for r in rows if r.get("sdft_response") == "reply-1") == 2
    assert trained_on[1]["role"] == "reinforce_prev"


def test_build_train_examples_negative_rewrites_prev():
    from sdft.config import load_config

    cfg = load_config("configs/online_learning.yaml")
    prior = [
        OnlineTurn(
            turn_index=1,
            instruction="joke",
            input="",
            output="",
            sdft_response="sdft-1",
            assistant_reply="bad-reply",
            record_id="r1",
            created_at="t1",
            latency=TurnLatency(total_ms=1),
        )
    ]
    rows, action, trained_on = build_train_examples(
        cfg,
        prior_turns=prior,
        instruction="Wrong!",
        user_input="",
        sdft_response="sdft-2",
        feedback_tone="negative",
        feedback_reward=-1,
        prev_rewrite="fresh rewrite",
    )
    assert action == "rewrite_prev"
    assert any(r["sdft_response"] == "fresh rewrite" for r in rows)
    assert not any(r["sdft_response"] == "bad-reply" for r in rows)


def test_build_train_examples_neutral_skips_prev():
    from sdft.config import load_config

    cfg = load_config("configs/online_learning.yaml")
    prior = [
        OnlineTurn(
            turn_index=1,
            instruction="joke",
            input="",
            output="",
            sdft_response="sdft-1",
            assistant_reply="reply-1",
            record_id="r1",
            created_at="t1",
            latency=TurnLatency(total_ms=1),
        )
    ]
    rows, action, _ = build_train_examples(
        cfg,
        prior_turns=prior,
        instruction="Another one please",
        user_input="",
        sdft_response="sdft-2",
        feedback_tone="neutral",
        feedback_reward=0,
    )
    assert action == "neutral_skip_prev"
    assert not any(r["sdft_response"] == "reply-1" for r in rows)
    assert rows[-1]["sdft_response"] == "sdft-2"


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
            latency=TurnLatency(total_ms=3000, tone_ms=1, generate_ms=800, inference_ms=200, train_ms=2000),
        ),
        OnlineTurn(
            turn_index=2,
            instruction="Thanks!",
            input="",
            output="d",
            sdft_response="sdft-d",
            record_id="rec-2",
            created_at="t2",
            feedback_tone="positive",
            feedback_reward=1,
            latency=TurnLatency(total_ms=5000, tone_ms=2, generate_ms=1200, inference_ms=300, train_ms=3500),
        ),
    ]
    summary = aggregate_turn_latencies(turns)
    assert summary["turn_count"] == 2
    assert summary["tone_ms"]["count"] == 2
    assert summary["generate_ms"]["count"] == 2
    assert summary["generate_ms"]["mean"] == 1000.0
    assert summary["inference_ms"]["count"] == 2
    assert summary["inference_ms"]["mean"] == 250.0
    assert summary["train_ms"]["p50"] == 2000.0
    assert summary["total_ms"]["p95"] == 5000.0
    assert summary["tone_counts"]["positive"] == 1
    assert summary["tone_counts"]["first_turn"] == 1


def test_aggregate_tone_counts():
    turns = [
        OnlineTurn(
            turn_index=1,
            instruction="a",
            input="",
            output="",
            sdft_response="x",
            record_id="r",
            created_at="t",
            latency=TurnLatency(total_ms=1),
        ),
        OnlineTurn(
            turn_index=2,
            instruction="bad",
            input="",
            output="",
            sdft_response="x",
            record_id="r2",
            created_at="t2",
            feedback_tone="negative",
            feedback_reward=-1,
            latency=TurnLatency(total_ms=1),
        ),
    ]
    counts = aggregate_tone_counts(turns)
    assert counts["first_turn"] == 1
    assert counts["negative"] == 1


def test_turn_latency_from_phases():
    phases = [
        {"name": "tone_classify", "start_ms": 0, "end_ms": 2, "duration_ms": 2},
        {"name": "generate_sdft", "start_ms": 2, "end_ms": 802, "duration_ms": 800},
        {"name": "train_update", "start_ms": 802, "end_ms": 2802, "duration_ms": 2000},
        {"name": "inference_reply", "start_ms": 2802, "end_ms": 3002, "duration_ms": 200},
        {"name": "record_collect", "start_ms": 3002, "end_ms": 3007, "duration_ms": 5},
    ]
    lat = turn_latency_from_phases(phases, input_tokens=12, output_tokens=34)
    assert lat.tone_ms == 2.0
    assert lat.generate_ms == 800.0
    assert lat.inference_ms == 200.0
    assert lat.train_ms == 2000.0
    assert lat.total_ms == 3007.0
    assert lat.input_tokens == 12
    assert lat.output_tokens == 34


def test_run_online_turn_update_then_infer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    (tmp_path / "data" / "collected").mkdir(parents=True)
    monkeypatch.setattr("sdft.records.paths.project_root", lambda start=None: tmp_path.resolve())
    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())

    session = create_session("configs/online_learning.yaml")
    call_order: list[str] = []

    def fake_generate(cfg, *, instruction, user_input="", fewshot_examples=None, **kwargs):
        call_order.append(f"generate:{instruction[:20]}")
        return "sdft rewrite text", 10, 5

    def fake_preview(cfg, adapter_dir, instruction, user_input="", **kwargs):
        call_order.append(f"infer:{instruction[:20]}")
        return "assistant reply", 8, 4

    def fake_train(cfg, adapter_dir, examples, **kwargs):
        call_order.append("train")
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
        assert all("sdft_response" in ex for ex in examples)

    with patch("sdft.online_learning.loop.generate_sdft_response", side_effect=fake_generate), patch(
        "sdft.online_learning.loop.generate_preview", side_effect=fake_preview
    ), patch("sdft.online_learning.loop.run_train_step", side_effect=fake_train):
        turn1 = run_online_turn(session.id, instruction="Tell a joke", preview=True)
        turn2 = run_online_turn(session.id, instruction="Thanks, perfect!", preview=True)

    assert call_order.index("train") < call_order.index(f"infer:Thanks, perfect!")
    assert turn1.feedback_tone is None
    assert turn1.assistant_reply == "assistant reply"
    assert turn2.feedback_tone == "positive"
    assert turn2.feedback_reward == 1
    assert turn2.preference_action == "reinforce_prev"

    loaded = load_session(session.id)
    assert loaded.turn_count == 2
    assert loaded.latency_summary["tone_counts"]["positive"] == 1
    assert loaded.latency_summary["tone_counts"]["first_turn"] == 1

    records = load_collected_records(collected_records_path())
    assert len(records) == 2
    assert records[1].metadata["feedback_tone"] == "positive"
    assert records[1].metadata["assistant_reply"] == "assistant reply"
    assert "trained_on" in records[1].metadata


def test_list_sessions_with_adapter_filters_ready(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())

    from sdft.online_learning.session import adapter_ready, list_sessions_with_adapter, save_session

    session = create_session("configs/online_learning.yaml")
    assert list_sessions_with_adapter(limit=10) == []

    adapter = Path(session.adapter_dir)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_session(session)

    ready = list_sessions_with_adapter(limit=10)
    assert len(ready) == 1
    assert ready[0].id == session.id
    assert adapter_ready(adapter)


def test_build_session_is_ephemeral(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'x'\n", encoding="utf-8")
    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())

    session = build_session("configs/online_learning.yaml")
    assert session.id.startswith("ol-")
    assert not session_persisted(session.id)
    assert not online_session_path(session.id).parent.exists()


def test_data_page_ephemeral_until_first_turn(client: TestClient, tmp_path: Path):
    resp = client.get("/data")
    assert resp.status_code == 200
    session_id = resp.text.split("session_id")[1].split('value="')[1].split('"')[0]
    assert not session_persisted(session_id)

    resp_new = client.get("/data?new=1")
    assert resp_new.status_code == 200
    new_session_id = resp_new.text.split("session_id")[1].split('value="')[1].split('"')[0]
    assert new_session_id != session_id
    assert not session_persisted(new_session_id)


def test_data_page_does_not_reuse_recent_session(client: TestClient, tmp_path: Path):
    saved = create_session("configs/online_learning.yaml")
    page = client.get("/data")
    assert page.status_code == 200
    session_id = page.text.split("session_id")[1].split('value="')[1].split('"')[0]
    assert session_id != saved.id


def test_data_page_loads_saved_session(client: TestClient, tmp_path: Path):
    saved = create_session("configs/online_learning.yaml")
    page = client.get(f"/data?session={saved.id}")
    assert page.status_code == 200
    session_id = page.text.split("session_id")[1].split('value="')[1].split('"')[0]
    assert session_id == saved.id


def test_data_page_shows_online_learning_ui(client: TestClient):
    resp = client.get("/data")
    assert resp.status_code == 200
    body = resp.text
    assert "Online learning" in body
    assert "tone" in body.lower()
    assert 'action="/data/turn"' in body
    assert "configs/online_learning.yaml" in body
    assert "Updating adapter" in body
    assert 'name="message"' in body
    assert "Gold output" not in body
    assert 'name="tags"' not in body
    assert "Export collected" not in body
    assert "Session stats" not in body


def test_online_turn_route_mocked(client: TestClient):
    page = client.get("/data")
    assert page.status_code == 200
    session_id = page.text.split("session_id")[1].split('value="')[1].split('"')[0]
    assert not session_persisted(session_id)

    def fake_run(session_id, **kwargs):
        from sdft.online_learning.schema import OnlineTurn, TurnLatency
        from sdft.online_learning.session import resolve_session, save_session
        from sdft.records.paths import utc_now_iso

        session = resolve_session(
            session_id,
            config_path=kwargs.get("config_path") or "configs/online_learning.yaml",
        )
        turn = OnlineTurn(
            turn_index=1,
            instruction=kwargs["instruction"],
            input=kwargs.get("input_text") or kwargs.get("input") or "",
            output=kwargs.get("output") or "",
            sdft_response="mock sdft target",
            assistant_reply="mock assistant reply",
            preview="mock assistant reply",
            record_id="rec-test-1",
            created_at=utc_now_iso(),
            feedback_tone=None,
            preference_action="first_turn",
            latency=TurnLatency(
                total_ms=1234,
                tone_ms=1,
                generate_ms=500,
                inference_ms=400,
                train_ms=300,
                output_tokens=7,
            ),
            latency_phases=[
                {"name": "generate_sdft", "start_ms": 0, "end_ms": 500, "duration_ms": 500},
                {"name": "train_update", "start_ms": 500, "end_ms": 800, "duration_ms": 300},
                {"name": "inference_reply", "start_ms": 800, "end_ms": 1200, "duration_ms": 400},
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
                "message": "Hello",
                "preview": "1",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert f"session={session_id}" in resp.headers["location"]
    assert "turn=1" in resp.headers["location"]
    assert session_persisted(session_id)

    detail = client.get(f"/data/{session_id}")
    assert detail.status_code == 200
    assert "Latency summary" in detail.text
    assert "Feedback tone counts" in detail.text
    assert "Tone classify" in detail.text
    assert "Per-turn latencies" in detail.text


def test_online_turn_htmx_returns_panel_fragment_not_redirect(client: TestClient):
    page = client.get("/data")
    session_id = page.text.split("session_id")[1].split('value="')[1].split('"')[0]

    def fake_run(session_id, **kwargs):
        from sdft.online_learning.schema import OnlineTurn, TurnLatency
        from sdft.online_learning.session import resolve_session, save_session
        from sdft.records.paths import utc_now_iso

        session = resolve_session(
            session_id,
            config_path=kwargs.get("config_path") or "configs/online_learning.yaml",
        )
        turn = OnlineTurn(
            turn_index=1,
            instruction=kwargs["instruction"],
            input="",
            output="",
            sdft_response="mock sdft target",
            assistant_reply="mock assistant reply",
            preview="mock assistant reply",
            record_id="rec-test-htmx",
            created_at=utc_now_iso(),
            feedback_tone=None,
            preference_action="first_turn",
            latency=TurnLatency(total_ms=100),
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
                "message": "Hello HTMX",
                "preview": "1",
            },
            headers={"HX-Request": "true"},
            follow_redirects=False,
        )

    assert resp.status_code == 200
    assert resp.headers.get("HX-Redirect") is None
    assert resp.headers.get("location") is None
    body = resp.text
    assert 'id="data-live"' in body
    assert "Hello HTMX" in body
    assert "mock assistant reply" in body
    assert "Turn 1 logged" in body
    assert "1 turn(s)" in body
    assert 'name="message"' in body
    assert 'value="Hello HTMX"' not in body
    assert resp.headers.get("HX-Push-Url") == f"/data?session={session_id}&turn=1"


def test_online_session_detail_404(client: TestClient):
    resp = client.get("/data/ol-does-not-exist")
    assert resp.status_code == 404
