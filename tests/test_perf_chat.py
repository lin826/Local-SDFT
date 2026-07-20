"""Smoke tests for multi-turn Performance chat UI (mocked model)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from sdft.online_learning import create_session
from sdft.records.schema import PerformanceMetrics, PerformanceResult
from web.app import CONFIG_OPTIONS, app
from web.demo_conditions import (
    FIXED_SYSTEM_INSTRUCTIONS,
    NO_SYSTEM_INSTRUCTION_HINT,
    build_design_summary,
    config_ignores_user_instruction,
    fixed_system_instruction,
)


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
            "latency_phases": [
                {"name": "config_load", "start_ms": 0.0, "end_ms": 5.0, "duration_ms": 5.0},
                {"name": "tokenizer_load", "start_ms": 5.0, "end_ms": 40.0, "duration_ms": 35.0},
                {"name": "model_load", "start_ms": 40.0, "end_ms": 1240.0, "duration_ms": 1200.0},
                {"name": "prompt_build", "start_ms": 1240.0, "end_ms": 1248.0, "duration_ms": 8.0},
                {"name": "generate", "start_ms": 1248.0, "end_ms": 1260.0, "duration_ms": 12.0},
                {"name": "decode", "start_ms": 1260.0, "end_ms": 1261.0, "duration_ms": 1.0},
                {"name": "persist", "start_ms": 1261.0, "end_ms": 1265.0, "duration_ms": 4.0},
            ],
        },
        config_path="configs/default.yaml",
    )


def _fake_measure_chat_with_phases(cfg, messages, **kwargs):
    """Mock measure_chat that advances a shared LatencyPhases clock."""
    run_id = kwargs.pop("_run_id", "bench-test-chat")
    result = _fake_chat_result(messages, run_id=run_id)
    phases = kwargs.get("latency_phases")
    if phases is not None:
        for name in ("tokenizer_load", "model_load", "prompt_build", "generate", "decode"):
            with phases.span(name):
                pass
        result.metadata["latency_phases"] = phases.to_list()
    return result


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


def test_perf_online_adapter_option_and_chat(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Online session adapters appear in /perf Config and load via measure_chat."""
    from sdft.online_learning.session import save_session
    from web.perf_models import online_config_value

    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())
    session = create_session("configs/online_learning.yaml")
    adapter = Path(session.adapter_dir)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_session(session)

    online_value = online_config_value(session.id)
    page = client.get("/perf")
    assert page.status_code == 200
    assert f"Online: {session.id}".encode() in page.content
    assert online_value.encode() in page.content

    captured: dict[str, object] = {}

    def fake_measure_chat(cfg, messages, **kwargs):
        captured["adapter_dir"] = kwargs.get("adapter_dir")
        captured["model_name"] = kwargs.get("model_name")
        captured["base_model"] = cfg.model.name
        return _fake_chat_result(messages, run_id="bench-online-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": online_value,
                "demo_condition": "plain",
                "instruction": "ignored",
                "user_message": "Try online adapter",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert captured["adapter_dir"] == adapter
    assert captured["base_model"] == session.model
    assert str(captured["model_name"]) == str(adapter)


def test_normalize_perf_config_keeps_online_session_outside_dropdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Valid online sessions must not be dropped when omitted from the dropdown cap."""
    from sdft.online_learning.session import save_session
    from web.perf_models import ONLINE_PERF_LIMIT, normalize_perf_config_path, online_config_value

    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())

    hidden = create_session("configs/online_learning.yaml", session_id="ol-hidden-adapter-test")
    adapter = Path(hidden.adapter_dir)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_session(hidden)

    for i in range(ONLINE_PERF_LIMIT):
        session = create_session("configs/online_learning.yaml", session_id=f"ol-dropdown-filler-{i}")
        filler_adapter = Path(session.adapter_dir)
        filler_adapter.mkdir(parents=True, exist_ok=True)
        (filler_adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
        save_session(session)

    online_value = online_config_value(hidden.id)
    from web.perf_models import known_perf_config_values

    assert online_value not in known_perf_config_values()
    assert normalize_perf_config_path(online_value) == online_value


def test_normalize_perf_config_rejects_missing_online_session():
    from web.demo_conditions import DEFAULT_CONFIG
    from web.perf_models import normalize_perf_config_path

    assert normalize_perf_config_path("online:ol-does-not-exist") == DEFAULT_CONFIG


def test_perf_infer_link_from_session_detail(client: TestClient, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from sdft.online_learning.session import save_session
    from web.perf_models import online_config_value

    monkeypatch.setattr("sdft.online_learning.paths.project_root", lambda start=None: tmp_path.resolve())
    session = create_session("configs/online_learning.yaml")
    adapter = Path(session.adapter_dir)
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    save_session(session)

    detail = client.get(f"/data/{session.id}")
    assert detail.status_code == 200
    assert b"Infer on Performance" in detail.content
    assert online_config_value(session.id).encode() in detail.content


def test_perf_page_shows_chat_ui(client: TestClient):
    resp = client.get("/perf")
    assert resp.status_code == 200
    body = resp.content
    assert b'action="/perf/chat"' in body
    assert b'data-stream-url="/perf/chat/stream"' in body
    assert b'id="chat-panel"' in body
    assert b"Plain chat inference" in body
    assert b'name="demo_condition"' in body
    assert b'value="plain"' in body
    assert b"configs/lfm25_alpacaeval2_trained.yaml" in body
    assert b"configs/default.yaml" in body
    assert b'name="prompt_strategy"' in body
    assert b"Prompt strategy" in body
    assert b"openclaw" not in body.lower()
    assert b'data-toolcall=' not in body
    assert b"syncInstructionField()" not in body
    assert b"Start generate" in body
    assert b"without a full page reload" not in body  # removed OpenClaw-specific howto line
    assert b"AlpacaEval-faithful ZS" in body
    assert b"streams tokens" in body or b"Streaming" in body
    idx = body.index(b'id="instruction"')
    tag_end = body.index(b">", idx) + 1
    close = body.index(b"</textarea>", tag_end)
    textarea_body = body[tag_end:close]
    assert textarea_body.strip() == b""
    tag = body[idx:tag_end]
    assert b"readonly" in tag
    assert b'name="instruction"' not in tag


def test_format_sse_framing():
    from web.app import format_sse

    raw = format_sse("token", {"text": "hi"})
    assert raw.startswith("event: token\n")
    assert 'data: {"text": "hi"}' in raw
    assert raw.endswith("\n\n")


def _parse_sse_events(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        event = "message"
        data_lines: list[str] = []
        for line in block.split("\n"):
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if data_lines:
            events.append((event, json.loads("\n".join(data_lines))))
    return events


def test_chat_stream_sse_tokens_and_done(client: TestClient):
    """POST /perf/chat/stream yields token events then done with panel HTML (mocked)."""

    def fake_iter_measure_chat(cfg, messages, **kwargs):
        phases = kwargs.get("latency_phases")
        if phases is not None:
            for name in ("tokenizer_load", "model_load", "prompt_build", "generate", "decode"):
                with phases.span(name):
                    pass
        yield ("token", "Hel")
        yield ("token", "lo!")
        result = _fake_chat_result(messages, run_id="bench-stream-1")
        result.metadata["messages"][-1]["content"] = "Hello!"
        if phases is not None:
            result.metadata["latency_phases"] = phases.to_list()
        yield ("result", result)

    with patch("web.app.iter_measure_chat", side_effect=fake_iter_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ), patch(
        "web.app.save_performance_result", lambda path, r: None
    ), patch(
        "web.app.load_config",
        return_value=type(
            "Cfg",
            (),
            {"model": type("M", (), {"name": "mock"})()},
        )(),
    ):
        resp = client.post(
            "/perf/chat/stream",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "",
                "user_message": "Hi stream",
                "messages_json": "[]",
            },
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    events = _parse_sse_events(resp.text)
    kinds = [k for k, _ in events]
    assert kinds[:2] == ["token", "token"]
    assert events[0][1]["text"] == "Hel"
    assert events[1][1]["text"] == "lo!"
    assert kinds[-1] == "done"
    done = events[-1][1]
    assert done["run_id"] == "bench-stream-1"
    assert 'id="chat-panel"' in done["html"]
    assert "Hello!" in done["html"]
    assert 'class="latency-gantt"' in done["html"]
    assert "continue=bench-stream-1" in done["continue_url"]


def test_chat_stream_error_event(client: TestClient):
    def boom(*args, **kwargs):
        raise RuntimeError("mock generate failed")
        yield  # make this a generator  # noqa: unreachable

    with patch("web.app.iter_measure_chat", side_effect=boom), patch(
        "web.app.load_config",
        return_value=type(
            "Cfg",
            (),
            {"model": type("M", (), {"name": "mock"})()},
        )(),
    ):
        resp = client.post(
            "/perf/chat/stream",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "",
                "user_message": "fail please",
                "messages_json": "[]",
            },
        )

    assert resp.status_code == 200
    events = _parse_sse_events(resp.text)
    assert any(k == "error" for k, _ in events)
    err = next(d for k, d in events if k == "error")
    assert "mock generate failed" in err["detail"]


def test_iter_measure_chat_yields_tokens_with_mock_streamer(monkeypatch):
    """Unit-test streamer helper without a real model/GPU."""
    from sdft.records import benchmark as bench
    from sdft.config import Config, GenerateConfig, ModelConfig, DataConfig

    class FakeStreamer:
        def __iter__(self):
            yield "tok"
            yield "en"

    class FakeTok:
        padding_side = "right"
        pad_token_id = 0

        def apply_chat_template(self, *a, **k):
            return "<prompt>"

        def __call__(self, text, return_tensors=None, add_special_tokens=False):
            import torch

            class Enc(dict):
                def to(self, device):
                    return self

            return Enc(input_ids=torch.tensor([[1, 2, 3]]))

        def encode(self, text, add_special_tokens=False):
            return list(range(max(1, len(text))))

    class FakeModel:
        def eval(self):
            return self

        def generate(self, **kwargs):
            return None

    monkeypatch.setattr(bench, "load_tokenizer", lambda model: FakeTok())
    monkeypatch.setattr(bench, "load_model", lambda model, device: FakeModel())
    monkeypatch.setattr(bench, "pick_device", lambda: "cpu")
    monkeypatch.setattr(bench, "TextIteratorStreamer", lambda *a, **k: FakeStreamer())

    cfg = Config(
        model=ModelConfig(name="mock"),
        data=DataConfig(dataset="dummy"),
        generation=GenerateConfig(max_new_tokens=8),
    )
    events = list(
        bench.iter_measure_chat(
            cfg,
            [{"role": "user", "content": "hi"}],
        )
    )
    assert events[0] == ("token", "tok")
    assert events[1] == ("token", "en")
    assert events[2][0] == "result"
    assert events[2][1].metadata["messages"][-1]["content"] == "token"
    assert events[2][1].metrics.output_tokens_total == 5  # len("token")


def test_config_options_ignore_user_instruction():
    for cfg in CONFIG_OPTIONS:
        assert config_ignores_user_instruction(cfg)


def test_alpacaeval_configs_have_no_fixed_system_instruction():
    for cfg in CONFIG_OPTIONS:
        assert fixed_system_instruction(cfg) == ""


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
    assert "AlpacaEval-faithful ZS" in base["system_instruction"]
    assert base["prompt_strategy"] == "ZS"

    sdft = build_design_summary(
        demo_condition="plain",
        config_path="configs/lfm25_alpacaeval2_trained.yaml",
        model_path="outputs/lfm25-230m-alpacaeval2-sdft-merged",
    )
    assert "SDFT merge" in sdft["variant"]
    assert sdft["config_path"] == "configs/lfm25_alpacaeval2_trained.yaml"
    assert "AlpacaEval-faithful ZS" in sdft["system_instruction"]

    online = build_design_summary(
        demo_condition="plain",
        config_path="online:ol-test123",
        model_path="outputs/online-learning/ol-test123/adapter",
        online_session_id="ol-test123",
        adapter_dir="outputs/online-learning/ol-test123/adapter",
    )
    assert "online LoRA" in online["variant"]
    assert online["online_session_id"] == "ol-test123"
    assert online["adapter_dir"].endswith("/adapter")


def test_htmx_chat_returns_partial_not_redirect(client: TestClient):
    def fake_measure_chat(cfg, messages, **kwargs):
        return _fake_measure_chat_with_phases(cfg, messages, _run_id="bench-htmx-1", **kwargs)

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ), patch(
        "web.app.save_performance_result", lambda path, r: None
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
    assert 'class="latency-gantt"' in body
    assert 'data-phase="model_load"' in body
    assert 'data-phase="generate"' in body
    assert "Latency phases" in body


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
    assert "AlpacaEval-faithful ZS" in meta["design_summary"]["system_instruction"]


def test_chat_ignores_user_instruction_for_alpacaeval_configs(client: TestClient):
    captured: list[list[dict[str, str]]] = []

    def fake_measure_chat(cfg, messages, **kwargs):
        captured.append(list(messages))
        return _fake_chat_result(messages, run_id="bench-ignore-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "You must always reply in rhyme.",
                "user_message": "How do I sew a button?",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert captured
    roles = [m["role"] for m in captured[0]]
    assert roles == ["user"]
    assert captured[0][0]["content"] == "How do I sew a button?"


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
        assert "AlpacaEval-faithful ZS" in body1
        assert "You are a witty PhD comic narrator." not in body1

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
        assert roles == ["user", "assistant", "user", "assistant"]


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
    assert 'class="latency-gantt"' in detail.text
    assert 'data-phase="model_load"' in detail.text
    assert "1200" in detail.text  # model_load duration
    assert "ms generate" in detail.text
    assert "ms total" in detail.text


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


def test_fixed_system_instruction_shown_readonly_and_used(client: TestClient, monkeypatch):
    fixed_text = "You are a helpful benchmark assistant."
    monkeypatch.setitem(FIXED_SYSTEM_INSTRUCTIONS, "configs/default.yaml", fixed_text)
    captured: list[list[dict[str, str]]] = []

    def fake_measure_chat(cfg, messages, **kwargs):
        captured.append(list(messages))
        return _fake_chat_result(messages, run_id="bench-fixed-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ):
        page = client.get("/perf")
        assert page.status_code == 200
        assert fixed_text in page.text
        assert NO_SYSTEM_INSTRUCTION_HINT not in page.text

        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "Ignore me.",
                "user_message": "Hello fixed",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert captured
    assert captured[0][0] == {"role": "system", "content": fixed_text}
    assert captured[0][1] == {"role": "user", "content": "Hello fixed"}


def test_unknown_prompt_strategy_rejected(client: TestClient):
    resp = client.post(
        "/perf/chat",
        data={
            "config_path": "configs/default.yaml",
            "demo_condition": "plain",
            "prompt_strategy": "NOT_AN_ARM",
            "instruction": "",
            "user_message": "Hello",
            "messages_json": "[]",
        },
    )
    assert resp.status_code == 400
    assert "unknown ablation arm" in resp.json()["detail"]


def test_cot_strategy_appends_cue_to_user_message(client: TestClient):
    captured: list[list[dict[str, str]]] = []

    def fake_measure_chat(cfg, messages, **kwargs):
        captured.append(list(messages))
        return _fake_chat_result(messages, run_id="bench-cot-1")

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", lambda r: None
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "prompt_strategy": "CoT",
                "instruction": "",
                "user_message": "How do I sew a button?",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    assert captured
    assert captured[0][0]["role"] == "user"
    assert "Let's think step by step." in captured[0][0]["content"]


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


def test_empty_assistant_response_renders_refusal_fallback(client: TestClient):
    from sdft.records.paths import performance_result_path
    from sdft.records.store import save_performance_result

    result = PerformanceResult(
        id="bench-empty-assistant",
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
            output_tokens_total=0,
            device="cpu",
        ),
        metadata={
            "messages": [
                {"role": "user", "content": "Tell me something unsafe."},
                {"role": "assistant", "content": ""},
            ],
            "examples": [
                {
                    "instruction": "",
                    "input": "Tell me something unsafe.",
                    "output": "",
                }
            ],
            "chat": True,
        },
        config_path="configs/default.yaml",
    )
    save_performance_result(performance_result_path(result.id), result)

    detail = client.get(f"/perf/{result.id}")
    assert detail.status_code == 200
    assert "sorry, but I can" in detail.text
    assert "assist with that." in detail.text

    perf = client.get("/perf")
    assert perf.status_code == 200
    assert "sorry, but I can" in perf.text
    assert "assist with that." in perf.text


def test_latency_phases_structure():
    import time

    from sdft.records.benchmark import LatencyPhases

    clock = LatencyPhases()
    with clock.span("tokenizer_load"):
        time.sleep(0.001)
    with clock.span("model_load"):
        time.sleep(0.001)
    phases = clock.to_list()
    assert [p["name"] for p in phases] == ["tokenizer_load", "model_load"]
    for p in phases:
        assert set(p) == {"name", "start_ms", "end_ms", "duration_ms"}
        assert p["end_ms"] >= p["start_ms"]
        assert p["duration_ms"] >= 0
        assert abs((p["end_ms"] - p["start_ms"]) - p["duration_ms"]) < 0.01
    assert phases[0]["start_ms"] >= 0
    assert phases[1]["start_ms"] >= phases[0]["end_ms"] - 0.5


def test_chat_persists_latency_phases(client: TestClient):
    saved: dict[str, PerformanceResult] = {}

    def fake_measure_chat(cfg, messages, **kwargs):
        return _fake_measure_chat_with_phases(cfg, messages, _run_id="bench-phases-1", **kwargs)

    def capture_persist(r):
        saved[r.id] = r

    with patch("web.app.measure_chat", side_effect=fake_measure_chat), patch(
        "web.app.persist_performance_result", side_effect=capture_persist
    ), patch(
        "web.app.save_performance_result",
        side_effect=lambda path, r: saved.setdefault(r.id, r),
    ):
        resp = client.post(
            "/perf/chat",
            data={
                "config_path": "configs/default.yaml",
                "demo_condition": "plain",
                "instruction": "",
                "user_message": "phase check",
                "messages_json": "[]",
            },
            follow_redirects=False,
        )
    assert resp.status_code == 303
    phases = saved["bench-phases-1"].metadata["latency_phases"]
    names = [p["name"] for p in phases]
    assert names[0] == "config_load"
    assert "model_load" in names
    assert "generate" in names
    assert names[-1] == "persist"
    for p in phases:
        assert {"name", "start_ms", "end_ms", "duration_ms"} <= set(p)

