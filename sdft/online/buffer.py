"""Replay buffer: mixes fresh demonstrations with replayed past ones.

Keeps updates stable (mitigates catastrophic forgetting) by ensuring each
training batch blends new signals with a recency- and weight-biased sample of
previously trained demonstrations, with a per-topic cap for domain diversity.
"""

from __future__ import annotations

import math
import random
import time

from ..config import OnlineConfig
from .events import Demonstration
from .store import SQLiteStore

RECENCY_TAU_SECONDS = 7 * 24 * 3600  # recency bias half-life-ish scale


class ReplayBuffer:
    def __init__(self, store: SQLiteStore, config: OnlineConfig, seed: int = 42):
        self.store = store
        self.config = config
        self._rng = random.Random(seed)

    def pending(self) -> list[Demonstration]:
        """Demonstrations not yet trained on."""
        return self.store.untrained_demonstrations()

    def should_update(self) -> bool:
        return len(self.pending()) >= self.config.min_new_demos

    def sample_batch(self, n: int) -> list[Demonstration]:
        """Sample n demonstrations: replay_ratio from history, the rest new."""
        n_replay = min(
            round(n * self.config.replay_ratio),
            len(self.store.trained_demonstrations()),
        )
        n_new = n - n_replay

        batch: list[Demonstration] = []
        new_pool = self.store.untrained_demonstrations()
        batch += self._weighted_sample(
            new_pool, n_new, [d.weight for d in new_pool]
        )

        replay_pool = self.store.trained_demonstrations()
        now = time.time()
        replay_weights = [
            d.weight * math.exp(-(now - d.created_at) / RECENCY_TAU_SECONDS)
            for d in replay_pool
        ]
        batch += self._weighted_sample(replay_pool, n_replay, replay_weights)

        self._rng.shuffle(batch)
        return batch

    def mark_trained(self, demos: list[Demonstration]) -> None:
        self.store.mark_trained([d.id for d in demos])

    def _weighted_sample(
        self, pool: list[Demonstration], k: int, weights: list[float]
    ) -> list[Demonstration]:
        """Weighted sampling without replacement, honoring the per-topic cap."""
        cap = self.config.max_per_topic_per_batch
        chosen: list[Demonstration] = []
        remaining = list(zip(pool, weights))
        while len(chosen) < k and remaining:
            total = sum(w for _, w in remaining)
            if total <= 0:
                pick = self._rng.randrange(len(remaining))
            else:
                r = self._rng.random() * total
                acc, pick = 0.0, len(remaining) - 1
                for i, (_, w) in enumerate(remaining):
                    acc += w
                    if r <= acc:
                        pick = i
                        break
            item, _ = remaining.pop(pick)
            if sum(1 for c in chosen if c.topic == item.topic) >= cap:
                continue
            chosen.append(item)
        return chosen
