"""Local self-distillation fine-tuning (SDFT) of Liquid AI LFM2.5-230M with LoRA.

Recipe (Shenfeld et al., 2026, "Self-Distillation Enables Continual Learning";
https://self-distillation.github.io/SDFT, arXiv:2601.19897):

1. generate: the model rewrites each training target in its own distribution,
   using a few in-context demonstrations drawn from the original dataset.
2. train:    LoRA SFT on (prompt -> model-generated response), or gold SFT via
   ``--target gold``.
3. grpo:     optional LoRA GRPO baseline (``python -m sdft.grpo_train``).
4. merge:    fold the LoRA adapter back into the base weights.

See ``docs/architecture.md`` for the package map and web/online-learning entry points.
"""

__version__ = "0.1.0"
