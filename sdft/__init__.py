"""Local self-distillation fine-tuning (SDFT) of Liquid AI LFM2.5-230M with LoRA.

Recipe (Yang et al., 2024, "Self-Distillation Bridges Distribution Gap in
Language Model Fine-Tuning"):

1. generate: the model rewrites each training target in its own distribution,
   using a few in-context demonstrations drawn from the original dataset.
2. train:    LoRA SFT on (prompt -> model-generated response).
3. merge:    fold the LoRA adapter back into the base weights.
"""

__version__ = "0.1.0"
