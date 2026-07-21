import pytest

from sdft.config import OnlineConfig
from sdft.online.buffer import ReplayBuffer
from sdft.online.events import Demonstration
from sdft.online.store import SQLiteStore


@pytest.fixture()
def store(tmp_path):
    s = SQLiteStore(tmp_path / "test.db")
    yield s
    s.close()


def make_demo(store, topic="t", weight=1.0, trained=0, source="correction"):
    d = Demonstration(
        source=source,
        conversation_id="c",
        messages=[{"role": "user", "content": "q"}],
        demonstration="a",
        topic=topic,
        weight=weight,
    )
    store.add_demonstration(d)
    if trained:
        store.mark_trained([d.id] * trained)
    return d


class TestShouldUpdate:
    def test_threshold(self, store):
        buf = ReplayBuffer(store, OnlineConfig(min_new_demos=2))
        assert not buf.should_update()
        make_demo(store)
        assert not buf.should_update()
        make_demo(store)
        assert buf.should_update()


class TestSampleBatch:
    def test_mix_of_new_and_replay(self, store):
        cfg = OnlineConfig(replay_ratio=0.5, max_per_topic_per_batch=10)
        buf = ReplayBuffer(store, cfg, seed=0)
        for _ in range(4):
            make_demo(store, trained=2)   # replay pool
        for _ in range(4):
            make_demo(store, trained=0)   # new pool
        batch = buf.sample_batch(4)
        assert len(batch) == 4
        n_new = sum(1 for d in batch if d.times_trained == 0)
        assert n_new == 2  # replay_ratio 0.5 of 4

    def test_topic_cap(self, store):
        cfg = OnlineConfig(replay_ratio=0.0, max_per_topic_per_batch=1)
        buf = ReplayBuffer(store, cfg, seed=0)
        for _ in range(3):
            make_demo(store, topic="same")
        make_demo(store, topic="other")
        batch = buf.sample_batch(2)
        topics = [d.topic for d in batch]
        assert topics.count("same") <= 1

    def test_no_replay_pool_yet(self, store):
        buf = ReplayBuffer(store, OnlineConfig(replay_ratio=0.5), seed=0)
        for _ in range(3):
            make_demo(store)
        batch = buf.sample_batch(2)
        assert len(batch) == 2
        assert all(d.times_trained == 0 for d in batch)

    def test_mark_trained_moves_pools(self, store):
        buf = ReplayBuffer(store, OnlineConfig(), seed=0)
        d = make_demo(store)
        assert len(buf.pending()) == 1
        buf.mark_trained([d])
        assert len(buf.pending()) == 0
        assert len(store.trained_demonstrations()) == 1
