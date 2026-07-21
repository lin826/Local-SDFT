import pytest

from sdft.config import OnlineConfig
from sdft.online.events import Correction, Message
from sdft.online.signals import SignalExtractor, auto_topic
from sdft.online.store import SQLiteStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


def add_conversation(store, conv_id, turns):
    """turns: list of (role, content). Returns list[Message]."""
    msgs = []
    for role, content in turns:
        msgs.append(store.add_message(Message(conversation_id=conv_id, role=role, content=content)))
    return msgs


class TestCorrectionSignal:
    def test_correction_builds_demo_with_context(self, store):
        msgs = add_conversation(store, "c1", [
            ("user", "What is our deploy command?"),
            ("assistant", "I think it is deploy.sh"),
        ])
        corr = Correction(
            conversation_id="c1",
            message_id=msgs[1].id,
            original=msgs[1].content,
            corrected="Our deploy command is `make deploy-prod`.",
        )
        extractor = SignalExtractor(store, OnlineConfig())
        demo = extractor.on_correction(corr)

        assert demo is not None
        assert demo.source == "correction"
        assert demo.messages == [{"role": "user", "content": "What is our deploy command?"}]
        assert demo.demonstration == "Our deploy command is `make deploy-prod`."
        assert demo.weight == OnlineConfig().correction_weight
        # persisted
        assert store.untrained_demonstrations()[0].id == demo.id

    def test_correction_uses_full_prior_context(self, store):
        msgs = add_conversation(store, "c1", [
            ("user", "I work on the payments service."),
            ("assistant", "Got it, the payments service."),
            ("user", "What is its retry policy?"),
            ("assistant", "Retries happen 3 times."),
        ])
        corr = Correction(
            conversation_id="c1", message_id=msgs[3].id,
            original=msgs[3].content, corrected="Retries: 5 attempts, exponential backoff.",
        )
        demo = SignalExtractor(store, OnlineConfig()).on_correction(corr)
        assert len(demo.messages) == 3
        assert demo.messages[-1]["content"] == "What is its retry policy?"

    def test_unknown_message_returns_none(self, store):
        add_conversation(store, "c1", [("user", "hi")])
        corr = Correction(conversation_id="c1", message_id="nope", original="a", corrected="b")
        assert SignalExtractor(store, OnlineConfig()).on_correction(corr) is None

    def test_correction_excluded_from_accepted(self, store):
        msgs = add_conversation(store, "c1", [
            ("user", "q1"),
            ("assistant", "a reply that will be corrected soon"),
        ])
        store.add_correction(Correction(
            conversation_id="c1", message_id=msgs[1].id, original="x", corrected="y",
        ))
        extractor = SignalExtractor(store, OnlineConfig())
        assert extractor.close_conversation("c1") == []


class TestAcceptedSignal:
    def test_close_conversation_harvests_replies(self, store):
        add_conversation(store, "c1", [
            ("user", "Tell me about our logging setup."),
            ("assistant", "We use structured JSON logs shipped to Loki."),
            ("user", "And alerting?"),
            ("assistant", "Alerts fire from Loki rules into PagerDuty."),
        ])
        extractor = SignalExtractor(store, OnlineConfig())
        demos = extractor.close_conversation("c1")

        assert len(demos) == 2
        assert all(d.source == "accepted" for d in demos)
        assert all(d.weight == OnlineConfig().accepted_weight for d in demos)
        # Each demo's context ends right before its own reply.
        by_demo_text = {d.demonstration: d for d in demos}
        d1 = by_demo_text["We use structured JSON logs shipped to Loki."]
        assert d1.messages == [{"role": "user", "content": "Tell me about our logging setup."}]
        d2 = by_demo_text["Alerts fire from Loki rules into PagerDuty."]
        assert len(d2.messages) == 3

    def test_cap_keeps_most_recent(self, store):
        cfg = OnlineConfig(max_accepted_per_conversation=1)
        add_conversation(store, "c1", [
            ("user", "first question here"),
            ("assistant", "the first answer, now stale"),
            ("user", "second question here"),
            ("assistant", "the freshest answer we keep"),
        ])
        demos = SignalExtractor(store, cfg).close_conversation("c1")
        assert len(demos) == 1
        assert demos[0].demonstration == "the freshest answer we keep"

    def test_short_replies_skipped(self, store):
        add_conversation(store, "c1", [
            ("user", "ok?"),
            ("assistant", "OK"),
        ])
        assert SignalExtractor(store, OnlineConfig()).close_conversation("c1") == []


class TestAutoTopic:
    def test_from_first_user_message(self):
        msgs = [
            Message(conversation_id="c", role="user", content="How do I deploy the API?"),
        ]
        assert auto_topic(msgs) == "how-do-i-deploy-the-api"

    def test_no_user_message(self):
        msgs = [Message(conversation_id="c", role="assistant", content="hi")]
        assert auto_topic(msgs) == "general"
