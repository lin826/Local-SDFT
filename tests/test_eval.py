from sdft.online.echo import EchoTrainer
from sdft.config import Config
from sdft.online.events import Demonstration
from sdft.online.probes import (
    ProbeEvaluator,
    _word_overlap,
    personalization_score,
    run_general_probes,
)


class KeywordBackend(EchoTrainer):
    """Echo backend that answers known probe questions correctly."""

    ANSWERS = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
    }

    def generate(self, messages, **overrides):
        q = messages[-1]["content"]
        return self.ANSWERS.get(q, "echo reply")


def test_general_probes_scoring():
    backend = KeywordBackend(Config())
    result = run_general_probes(backend, probes=[
        ("What is the capital of France?", "paris"),
        ("What is 2 + 2?", "4"),
        ("What is the largest ocean on Earth?", "pacific"),
    ])
    assert result["hits"] == 2
    assert result["accuracy"] == 2 / 3


def test_word_overlap():
    assert _word_overlap("deploy with make deploy-prod", "use make deploy-prod") > 0.5
    assert _word_overlap("totally unrelated words here", "quantum flux capacitor") == 0.0
    assert _word_overlap("", "anything") == 0.0


def test_personalization_score():
    class RecallBackend(EchoTrainer):
        def generate(self, messages, **overrides):
            return "The deploy command is make deploy-prod."

    demo = Demonstration(
        source="correction",
        conversation_id="c",
        messages=[{"role": "user", "content": "deploy command?"}],
        demonstration="The deploy command is make deploy-prod.",
    )
    result = personalization_score(RecallBackend(Config()), [demo])
    assert result["n"] == 1
    assert result["overlap"] > 0.8


def test_probe_evaluator_degradation():
    ev = ProbeEvaluator(threshold_drop=0.2)

    class Ctl:
        backend = KeywordBackend(Config())

    ev.baseline = 1.0
    out = ev(Ctl())
    assert out["probe_accuracy"] < 1.0  # KeywordBackend only knows 2 of 8
    assert out["degraded"] is True
