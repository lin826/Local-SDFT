"""Tests for device / encoding helpers in ``sdft.utils``."""

from __future__ import annotations

import torch
import torch.nn as nn

from sdft.utils import model_device, to_model_device


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))


def test_model_device_and_to_model_device_cpu() -> None:
    model = _Tiny()
    assert model_device(model).type == "cpu"
    batch = {"input_ids": torch.tensor([[1, 2, 3]]), "meta": "x"}
    moved = to_model_device(batch, model)
    assert torch.is_tensor(moved["input_ids"])
    assert moved["input_ids"].device.type == "cpu"
    assert moved["meta"] == "x"


def test_to_model_device_batch_encoding_like() -> None:
    model = _Tiny()

    class _Enc(dict):
        def to(self, device):  # noqa: ANN001
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in self.items()}

    enc = _Enc(input_ids=torch.tensor([[9]]))
    out = to_model_device(enc, model)
    assert out["input_ids"].device.type == "cpu"
