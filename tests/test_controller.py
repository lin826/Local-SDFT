import pytest

from sdft.config import Config
from sdft.online.controller import OnlineController


def make_cfg(tmp_path, **online_overrides) -> Config:
    cfg = Config()
    cfg.online.backend = "echo"
    cfg.online.db_path = str(tmp_path / "online.db")
    cfg.online.adapters_dir = str(tmp_path / "adapters")
    cfg.online.min_new_demos = 2
    cfg.online.steps_per_update = 2
    cfg.online.demos_per_step = 1
    cfg.online.eval_every_n_updates = 0
    for k, v in online_overrides.items():
        setattr(cfg.online, k, v)
    return cfg


@pytest.fixture()
def ctrl(tmp_path):
    c = OnlineController.build(make_cfg(tmp_path))
    yield c
    c.store.close()


class TestChat:
    def test_chat_logs_turns(self, ctrl):
        mid, reply = ctrl.chat("c1", "hello")
        assert reply == "echo: hello"
        msgs = ctrl.store.conversation_messages("c1")
        assert [m.role for m in msgs] == ["user", "assistant"]
        assert msgs[1].id == mid

    def test_base_adapter_on_init(self, ctrl):
        vs = ctrl.store.list_adapter_versions()
        assert len(vs) == 1 and vs[0].version == 0 and vs[0].active


class TestCorrectionUpdate:
    def test_correction_triggers_update_at_threshold(self, ctrl):
        mid, _ = ctrl.chat("c1", "deploy command?")
        demo = ctrl.correct("c1", mid, "Use `make deploy-prod`.")
        assert demo is not None and demo.source == "correction"
        assert ctrl.maybe_update() is None  # 1 < min_new_demos=2
        mid2, _ = ctrl.chat("c1", "rollback command?")
        ctrl.correct("c1", mid2, "Use `make rollback`.")
        run = ctrl.maybe_update()
        assert run is not None and run.adapter_version == 1
        assert ctrl.store.get_active_adapter().version == 1
        assert len(ctrl.buffer.pending()) == 0


class TestRollback:
    def test_rollback_restores_previous(self, ctrl):
        for q, a in [("q1", "a1"), ("q2", "a2")]:
            mid, _ = ctrl.chat("c1", q)
            ctrl.correct("c1", mid, a)
        assert ctrl.maybe_update() is not None
        av = ctrl.rollback()
        assert av.version == 0
        assert ctrl.store.get_active_adapter().version == 0

    def test_rollback_at_base_none(self, ctrl):
        assert ctrl.rollback() is None


class TestRewardPath:
    def test_shaper_task_always_harvests(self, tmp_path):
        # five_words has a shaper, so even a non-passing sample is reshaped into a
        # guaranteed-correct target and harvested.
        cfg = make_cfg(tmp_path, reward_fn="five_words", reward_num_samples=3)
        c = OnlineController.build(cfg)
        try:
            c.chat("c1", "name three primary colors please")
            assert len(c.buffer.pending()) == 1
            # the harvested target actually satisfies the reward
            from sdft.online.reward import get_reward_fn
            demo = c.buffer.pending()[0]
            assert get_reward_fn("five_words")("q", demo.demonstration) == 1.0
        finally:
            c.store.close()

    def test_reward_without_shaper_skips_zero(self, tmp_path):
        # A reward with NO shaper and zero score harvests nothing.
        from sdft.online import reward
        reward._REGISTRY["zeroonly"] = lambda p, r: 0.0
        try:
            cfg = make_cfg(tmp_path, reward_fn="zeroonly", reward_num_samples=2)
            c = OnlineController.build(cfg)
            try:
                c.chat("c1", "hello")
                assert len(c.buffer.pending()) == 0
            finally:
                c.store.close()
        finally:
            reward._REGISTRY.pop("zeroonly", None)

    def test_reward_fn_keeps_positive(self, tmp_path):
        # A reward that always passes -> best sample harvested as a demo.
        from sdft.online import reward

        @reward.reward("always")
        def _always(prompt, reply):
            return 1.0

        cfg = make_cfg(tmp_path, reward_fn="always", reward_num_samples=2)
        c = OnlineController.build(cfg)
        try:
            c.chat("c1", "hello there")
            assert len(c.buffer.pending()) == 1
        finally:
            c.store.close()


class TestEvalHook:
    def test_degraded_rolls_back(self, tmp_path):
        cfg = make_cfg(tmp_path, eval_every_n_updates=1)
        c = OnlineController.build(cfg)
        c.eval_hook = lambda ctrl: {"degraded": True}
        try:
            for q, a in [("q", "a"), ("q2", "a2")]:
                mid, _ = c.chat("c1", q)
                c.correct("c1", mid, a)
            c.maybe_update()
            assert c.store.get_active_adapter().version == 0
        finally:
            c.store.close()


class TestStats:
    def test_stats_shape(self, ctrl):
        ctrl.chat("c1", "hi")
        s = ctrl.stats()
        assert s["conversations"] == 1
        assert s["active_adapter"] == 0
        assert s["updates_total"] == 0
