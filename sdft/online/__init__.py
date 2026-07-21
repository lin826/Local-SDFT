"""Online SDFT: learn from live user interaction while serving.

The offline pipeline (sdft.generate/train/merge) distills once from a dataset.
This package keeps distilling *during serving* — corrections, accepted replies,
and reward-selected on-policy samples become demonstrations that drive small
LoRA updates, with versioned adapters and rollback. Designed for edge devices
(laptop/phone) running a small model like LFM2.5-230M.
"""

from .controller import OnlineController
from .events import Correction, Demonstration, Message

__all__ = ["OnlineController", "Correction", "Demonstration", "Message"]
