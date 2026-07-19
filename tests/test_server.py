import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sdft.config import Config
from sdft.online.controller import OnlineController
from sdft.online.serve import create_app


@pytest.fixture()
def client(tmp_path):
    cfg = Config()
    cfg.online.backend = "echo"
    cfg.online.db_path = str(tmp_path / "online.db")
    cfg.online.adapters_dir = str(tmp_path / "adapters")
    cfg.online.min_new_demos = 2
    cfg.online.eval_every_n_updates = 0
    ctrl = OnlineController.build(cfg)
    yield TestClient(create_app(ctrl))
    ctrl.store.close()


def test_chat_completion(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hello world"}]})
    assert r.status_code == 200
    d = r.json()
    assert d["choices"][0]["message"]["content"] == "echo: hello world"
    assert d["sdft"]["turn_type"] == "main"


def test_feedback_flow_and_update(client):
    d = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "deploy?"}],
        "conversation_id": "w1"}).json()
    client.post("/v1/feedback", json={
        "conversation_id": "w1", "message_id": d["message_id"],
        "corrected_text": "Use `make deploy-prod`."})
    assert client.get("/v1/stats").json()["pending_demos"] == 1
    d = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "rollback?"}],
        "conversation_id": "w1"}).json()
    r = client.post("/v1/feedback", json={
        "conversation_id": "w1", "message_id": d["message_id"],
        "corrected_text": "Use `make rollback`."})
    assert r.json()["update_ran"] is True
    assert client.get("/v1/stats").json()["active_adapter"] == 1
    assert client.post("/v1/rollback").json()["version"] == 0


class TestOpenClawHarness:
    def test_side_turn_not_recorded(self, client):
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "heartbeat"}],
            "conversation_id": "s1", "turn_type": "side"})
        assert r.json()["sdft"]["turn_type"] == "side"
        # side turns are not logged as conversation history
        assert client.get("/v1/stats").json()["conversations"] == 0

    def test_headers_supply_session_and_turn(self, client):
        r = client.post("/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Session-Id": "hsess", "X-Turn-Type": "main"})
        assert r.json()["conversation_id"] == "hsess"

    def test_session_done_closes(self, client):
        client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "tell me about logging"}],
            "conversation_id": "d1"})
        r = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "thanks, bye"}],
            "conversation_id": "d1", "session_done": True})
        assert r.status_code == 200
        # closing harvested at least one accepted self-demonstration
        assert client.get("/v1/stats").json()["demonstrations"] >= 1


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "SDFT" in r.text or "sdft" in r.text
