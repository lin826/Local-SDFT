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


def test_house_style_shaper_yields_full_marks():
    from sdft.online.reward import get_reward_fn, get_shaper
    rfn = get_reward_fn("house_style"); shp = get_shaper("house_style")
    for raw in ["TCP is reliable. UDP is fast. Pick per need.",
                "I'm ready to dive in", "one two three four five six seven"]:
        assert rfn("q", shp("q", raw)) >= 0.99


def test_style_shapers_pass_their_rewards():
    from sdft.online.reward import get_reward_fn, get_shaper
    for name in ("five_words", "terse", "house_style", "direct"):
        rfn, shp = get_reward_fn(name), get_shaper(name)
        assert shp is not None, name
        assert rfn("q", shp("q", "some rambly model answer here that is long")) >= 0.99, name


def test_controller_set_task_switches(tmp_path):
    from sdft.config import Config
    from sdft.online.controller import OnlineController
    cfg = Config(); cfg.online.backend = "echo"
    cfg.online.db_path = str(tmp_path / "d.db"); cfg.online.adapters_dir = str(tmp_path / "a")
    c = OnlineController.build(cfg)
    try:
        assert c._reward_fn is None
        c.set_task("five_words")
        assert c._reward_fn is not None and c._shaper is not None
        c.set_task(None)
        assert c._reward_fn is None
    finally:
        c.store.close()
