"""Torch SDFT trainer for the online loop.

Reuses the repo's model/device helpers (sdft.utils) and LoRA target spec
(sdft.config.LoraConfig) so the online path shares conventions with the
offline pipeline. Unlike the reference two-model setup, teacher and student
are the *same* model — the teacher just sees a demonstration-conditioned
prompt, so only one model lives in memory. Loss is per-token forward KL over
the full vocabulary (GKD / on-policy SDFT default).
"""

from __future__ import annotations

import logging

from ..config import Config
from ..utils import load_model, load_tokenizer, pick_device
from .events import Demonstration
from .teacher import build_student_messages, build_teacher_messages

log = logging.getLogger(__name__)

FALLBACK_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
)


class TorchTrainer:
    name = "torch"

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self._torch = None
        self.device = "cpu"

    # ---- loading ---------------------------------------------------------

    def load(self) -> None:
        import torch
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model

        self._torch = torch
        self.device = pick_device()
        log.info("torch device: %s", self.device)

        self.tokenizer = load_tokenizer(self.cfg.model)
        if self.tokenizer.chat_template is None:
            log.info("no chat template; installing ChatML fallback")
            self.tokenizer.chat_template = FALLBACK_CHAT_TEMPLATE

        base = load_model(self.cfg.model, self.device)
        peft_cfg = PeftLoraConfig(
            r=self.cfg.lora.r,
            lora_alpha=self.cfg.lora.alpha,
            lora_dropout=self.cfg.lora.dropout,
            target_modules=self.cfg.lora.target_modules,
            task_type="CAUSAL_LM",
        )
        self.model = get_peft_model(base, peft_cfg)
        self.model.print_trainable_parameters()
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        o = self.cfg.online
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable, lr=o.lr, weight_decay=o.weight_decay
        )

    # ---- tokenization ----------------------------------------------------

    def _chat_ids(self, messages: list[dict[str, str]]) -> list[int]:
        out = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True
        )
        ids = out["input_ids"] if hasattr(out, "keys") else out
        ids = list(ids)
        if ids and isinstance(ids[0], (list, tuple)):
            ids = list(ids[0])
        max_len = self.cfg.online.max_prompt_tokens
        if len(ids) > max_len:
            ids = ids[-max_len:]
        return ids

    def _tensor(self, ids: list[int]):
        return self._torch.tensor([ids], device=self.device)

    # ---- serving ---------------------------------------------------------

    def generate(self, messages: list[dict[str, str]], **overrides) -> str:
        torch = self._torch
        o = self.cfg.online
        temperature = overrides.get("temperature", o.serve_temperature)
        max_new = overrides.get("max_new_tokens", o.serve_max_new_tokens)
        input_ids = self._tensor(self._chat_ids(messages))
        kwargs = dict(max_new_tokens=max_new)
        if temperature and temperature > 0:
            kwargs.update(do_sample=True, temperature=temperature, min_p=o.serve_min_p)
        else:
            kwargs.update(do_sample=False)
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(input_ids=input_ids, **kwargs)
        new_ids = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

    def sample(self, messages: list[dict[str, str]], n: int = 1) -> list[str]:
        """On-policy samples used for reward-selected self-distillation."""
        torch = self._torch
        o = self.cfg.online
        input_ids = self._tensor(self._chat_ids(messages))
        self.model.eval()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=o.max_completion_tokens,
                do_sample=True,
                temperature=o.sample_temperature,
                top_p=o.sample_top_p,
                num_return_sequences=n,
            )
        return [
            self.tokenizer.decode(out[i, input_ids.shape[1]:], skip_special_tokens=True).strip()
            for i in range(out.shape[0])
        ]

    # ---- SDFT update -----------------------------------------------------

    def train_on_demos(self, demos: list[Demonstration]) -> dict[str, float]:
        torch = self._torch
        o = self.cfg.online
        self.model.train()
        self.model.config.use_cache = False
        self.optimizer.zero_grad()

        totals = {"loss": 0.0, "kl_to_base": 0.0, "completion_tokens": 0.0}
        trained = 0
        use_sft = self.cfg.online.loss_type == "sft"
        for demo in demos:
            step = self._sft_loss_for_demo(demo) if use_sft else self._loss_for_demo(demo)
            if step is None:
                continue
            loss, aux = step
            (loss / len(demos)).backward()
            for k in totals:
                totals[k] += aux[k]
            trained += 1

        if trained == 0:
            self.optimizer.zero_grad()
            self.model.config.use_cache = True
            self.model.eval()
            return {"loss": float("nan"), "trained": 0.0, **totals}

        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], o.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.model.config.use_cache = True
        self.model.eval()
        return {
            "loss": totals["loss"] / trained,
            "kl_to_base": totals["kl_to_base"] / trained,
            "completion_tokens": totals["completion_tokens"] / trained,
            "trained": float(trained),
        }

    def _loss_for_demo(self, demo: Demonstration):
        torch = self._torch
        o = self.cfg.online

        student_ids = self._chat_ids(build_student_messages(demo.messages))
        completion = self._sample_ids(student_ids)
        n_skip = o.num_loss_tokens_to_skip
        if completion.numel() <= n_skip:
            return None

        teacher_ids = self._chat_ids(
            build_teacher_messages(demo.messages, demo.demonstration, o.reinstruct_template)
        )

        s_input = self._tensor(student_ids + completion.tolist())
        s_logits = self.model(input_ids=s_input).logits[0, len(student_ids) - 1: -1]
        s_logp = torch.log_softmax(s_logits.float(), dim=-1)

        with torch.no_grad():
            t_input = self._tensor(teacher_ids + completion.tolist())
            t_logits = self.model(input_ids=t_input).logits[0, len(teacher_ids) - 1: -1]
            t_logp = torch.log_softmax(t_logits.float(), dim=-1)

        mask = torch.ones(completion.numel(), device=self.device)
        mask[:n_skip] = 0.0
        per_token = (t_logp.exp() * (t_logp - s_logp)).sum(-1)  # forward KL(teacher||student)
        loss = (per_token * mask).sum() / mask.sum()
        aux = {"loss": loss.item(), "kl_to_base": 0.0,
               "completion_tokens": float(completion.numel())}

        if o.beta_kl_base > 0:
            with torch.no_grad(), self.model.disable_adapter():
                b_logits = self.model(input_ids=s_input).logits[0, len(student_ids) - 1: -1]
                b_logp = torch.log_softmax(b_logits.float(), dim=-1)
            kl_base = ((s_logp.exp() * (s_logp - b_logp)).sum(-1) * mask).sum() / mask.sum()
            loss = loss + o.beta_kl_base * kl_base
            aux["kl_to_base"] = kl_base.item()

        return loss, aux

    def _sft_loss_for_demo(self, demo: Demonstration):
        """Completion-only NLL on the demonstration text (RAFT / stable supervised).

        Used for reward-selected samples and shaped targets, where the
        demonstration itself is the thing to imitate.
        """
        torch = self._torch
        prompt_ids = self._chat_ids(build_student_messages(demo.messages))
        comp_ids = self.tokenizer(demo.demonstration, add_special_tokens=False)["input_ids"]
        if not comp_ids:
            return None
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            comp_ids = list(comp_ids) + [eos]

        input_ids = self._tensor(prompt_ids + list(comp_ids))
        logits = self.model(input_ids=input_ids).logits[0, len(prompt_ids) - 1: -1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        targets = torch.tensor(list(comp_ids), device=self.device)
        nll = -logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1).mean()
        return nll, {"loss": nll.item(), "kl_to_base": 0.0,
                     "completion_tokens": float(len(comp_ids))}

    def _sample_ids(self, prompt_ids: list[int]):
        torch = self._torch
        o = self.cfg.online
        input_ids = self._tensor(prompt_ids)
        was_training = self.model.training
        self.model.eval()
        self.model.config.use_cache = True
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=o.max_completion_tokens,
                do_sample=True,
                temperature=o.sample_temperature,
                top_p=o.sample_top_p,
            )
        self.model.config.use_cache = False
        if was_training:
            self.model.train()
        return out[0, input_ids.shape[1]:]

    # ---- adapters --------------------------------------------------------

    def save_adapter(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load_adapter(self, path: str) -> None:
        import os

        from peft.utils.save_and_load import set_peft_model_state_dict

        st = os.path.join(path, "adapter_model.safetensors")
        if os.path.exists(st):
            from safetensors.torch import load_file

            state = load_file(st)
        else:
            state = self._torch.load(
                os.path.join(path, "adapter_model.bin"), map_location="cpu", weights_only=True
            )
        set_peft_model_state_dict(self.model, state, adapter_name="default")
