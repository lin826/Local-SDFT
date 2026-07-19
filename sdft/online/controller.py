"""Online learning controller: serve, collect signals, run SDFT updates.

Single process owns one model; an RLock serializes serving vs. training.
Updates fire when enough new demonstrations accrue (or on demand), run a few
LoRA gradient steps, and save a versioned adapter that can be rolled back.

Two signal paths feed the demonstration buffer:
  * corrections / accepted replies  (SignalExtractor)
  * reward-selected on-policy samples (when online.reward_fn is set): sample N
    rollouts, keep the best-rewarded as a self-demonstration (RAFT-style RL).
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from ..config import Config
from .buffer import ReplayBuffer
from .echo import create_backend
from .events import AdapterVersion, Correction, Demonstration, Message, TrainingRun, now
from .signals import SignalExtractor, auto_topic
from .store import SQLiteStore

log = logging.getLogger(__name__)

EvalHook = Callable[["OnlineController"], dict | None]


class OnlineController:
    def __init__(self, cfg: Config, backend, store: SQLiteStore):
        self.cfg = cfg
        self.backend = backend
        self.store = store
        self.buffer = ReplayBuffer(store, cfg.online)
        self.extractor = SignalExtractor(store, cfg.online)
        self._lock = threading.RLock()
        self.eval_hook: EvalHook | None = None
        self._reward_fn = None
        if cfg.online.reward_fn:
            from .reward import get_reward_fn

            self._reward_fn = get_reward_fn(cfg.online.reward_fn)
        self._updates = self.store.count_training_runs()
        self._ensure_base_version()

    @classmethod
    def build(cls, cfg: Config) -> "OnlineController":
        store = SQLiteStore(cfg.online.db_path)
        backend = create_backend(cfg)
        backend.load()
        return cls(cfg, backend, store)

    # ---- serving ---------------------------------------------------------

    def chat(self, conversation_id: str, user_text: str, **gen_overrides) -> tuple[str, str]:
        user_msg = self.store.add_message(
            Message(conversation_id=conversation_id, role="user", content=user_text)
        )
        history = [m.to_chat() for m in self.store.conversation_messages(conversation_id)]
        with self._lock:
            reply = self.backend.generate(history, **gen_overrides)
        asst = self.store.add_message(
            Message(conversation_id=conversation_id, role="assistant",
                    content=reply, reply_to=user_msg.id)
        )
        # Reward-driven RL path: harvest a self-demonstration from the best rollout.
        if self._reward_fn is not None:
            self._reward_harvest(conversation_id, history, reply)
        return asst.id, reply

    def _reward_harvest(self, conversation_id: str, history: list[dict], served_reply: str) -> None:
        o = self.cfg.online
        prompt_text = history[-1]["content"] if history else ""
        # Sample candidates from a (optionally) instruction-conditioned model — the
        # SDFT teacher hint — so a cold-start model can produce passing samples.
        sample_ctx = history
        if o.coach_instruction:
            sample_ctx = [{"role": "system", "content": o.coach_instruction}, *history]
        with self._lock:
            samples = self.backend.sample(sample_ctx, n=o.reward_num_samples)
        samples.append(served_reply)
        scored = [(s, self._reward_fn(prompt_text, s)) for s in samples]
        best, best_r = max(scored, key=lambda x: x[1])
        if best_r <= 0:
            return  # nothing worth imitating this turn
        demo = Demonstration(
            source="accepted",
            conversation_id=conversation_id,
            messages=history,
            demonstration=best,
            topic=auto_topic([Message(conversation_id=conversation_id, role="user",
                                      content=prompt_text)]),
            weight=o.correction_weight * best_r,
        )
        self.store.add_demonstration(demo)

    # ---- feedback --------------------------------------------------------

    def correct(self, conversation_id: str, message_id: str, corrected: str) -> Demonstration | None:
        msg = self.store.get_message(message_id)
        if msg is None or msg.role != "assistant":
            return None
        corr = self.store.add_correction(
            Correction(conversation_id=conversation_id, message_id=message_id,
                       original=msg.content, corrected=corrected)
        )
        return self.extractor.on_correction(corr)

    def close_conversation(self, conversation_id: str) -> list[Demonstration]:
        return self.extractor.close_conversation(conversation_id)

    # ---- updates ---------------------------------------------------------

    def maybe_update(self, force: bool = False) -> TrainingRun | None:
        if not force and not self.buffer.should_update():
            return None
        return self.run_update()

    def run_update(self) -> TrainingRun | None:
        o = self.cfg.online
        with self._lock:
            n = o.steps_per_update * o.demos_per_step
            demos = self.buffer.sample_batch(n)
            if not demos:
                return None
            chunks = [demos[i: i + o.demos_per_step] for i in range(0, len(demos), o.demos_per_step)]
            metrics_list = [self.backend.train_on_demos(c) for c in chunks]
            self.buffer.mark_trained(demos)

            agg = _mean_metrics(metrics_list)
            version = max((v.version for v in self.store.list_adapter_versions()), default=-1) + 1
            path = self._adapter_path(version)
            self.backend.save_adapter(path)

            run = TrainingRun(steps=len(chunks), demo_ids=[d.id for d in demos],
                              metrics=agg, adapter_version=version, finished_at=now())
            self.store.record_training_run(run)
            self.store.add_adapter_version(AdapterVersion(
                version=version, path=path, training_run_id=run.id,
                note=f"update {version}: loss {agg.get('loss', float('nan')):.4f}"))
            self.store.set_active_adapter(version)
            self._updates += 1
        self._maybe_eval(run)
        return run

    def rollback(self, version: int | None = None) -> AdapterVersion | None:
        versions = self.store.list_adapter_versions()
        if not versions:
            return None
        if version is None:
            active = next((v for v in versions if v.active), versions[-1])
            earlier = [v for v in versions if v.version < active.version]
            if not earlier:
                return None
            target = earlier[-1]
        else:
            target = next((v for v in versions if v.version == version), None)
            if target is None:
                return None
        with self._lock:
            self.backend.load_adapter(target.path)
        self.store.set_active_adapter(target.version)
        return target

    # ---- eval hook -------------------------------------------------------

    def _maybe_eval(self, run: TrainingRun) -> None:
        every = self.cfg.online.eval_every_n_updates
        if not self.eval_hook or every <= 0 or self._updates % every != 0:
            return
        with self._lock:
            result = self.eval_hook(self) or {}
        run.metrics.update({f"eval/{k}": v for k, v in result.items() if isinstance(v, (int, float))})
        if result.get("degraded") and run.adapter_version:
            log.warning("probe degradation after v%d; rolling back", run.adapter_version)
            self.rollback(run.adapter_version - 1)

    # ---- misc ------------------------------------------------------------

    def stats(self) -> dict:
        active = self.store.get_active_adapter()
        return {
            "conversations": len(self.store.list_conversations()),
            "demonstrations": len(self.store.all_demonstrations()),
            "pending_demos": len(self.buffer.pending()),
            "updates_total": self._updates,
            "active_adapter": active.version if active else None,
            "adapter_versions": len(self.store.list_adapter_versions()),
        }

    def _adapter_path(self, version: int) -> str:
        return str(Path(self.cfg.online.adapters_dir) / f"v{version}")

    def _ensure_base_version(self) -> None:
        if self.store.list_adapter_versions():
            return
        path = self._adapter_path(0)
        with self._lock:
            self.backend.save_adapter(path)
        self.store.add_adapter_version(
            AdapterVersion(version=0, path=path, note="base adapter", active=True))
        self.store.set_active_adapter(0)


_SUM_KEYS = {"trained"}


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if not metrics_list:
        return {}
    keys = {k for m in metrics_list for k, v in m.items() if isinstance(v, (int, float))}
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and m[k] == m[k]]
        if vals:
            out[k] = sum(vals) if k in _SUM_KEYS else sum(vals) / len(vals)
    return out
