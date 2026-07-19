"""Per-token divergence losses for SDFT, in backend-agnostic numpy.

Backends (torch / MLX) re-implement the same math natively; these reference
implementations exist for unit tests and numerical cross-checks. Semantics
match TRL's GKD and the Self-Distillation repo's default (alpha=0):

- forward KL  = KL(teacher || student)   (beta = 0)  <- SDFT default
- reverse KL  = KL(student || teacher)   (beta = 1)
- generalized JSD with mixture m = beta * teacher + (1 - beta) * student
"""

from __future__ import annotations

import numpy as np


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    m = logits.max(axis=axis, keepdims=True)
    shifted = logits - m
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _masked_mean(per_token: np.ndarray, mask: np.ndarray | None) -> float:
    if mask is None:
        return float(per_token.mean())
    mask = mask.astype(per_token.dtype)
    denom = mask.sum()
    if denom == 0:
        raise ValueError("loss mask sums to zero")
    return float((per_token * mask).sum() / denom)


def forward_kl(
    student_logp: np.ndarray,
    teacher_logp: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """KL(teacher || student), summed over vocab, masked-averaged over tokens.

    student_logp, teacher_logp: [T, V] log-probabilities; mask: [T].
    """
    teacher_p = np.exp(teacher_logp)
    per_token = (teacher_p * (teacher_logp - student_logp)).sum(axis=-1)
    return _masked_mean(per_token, mask)


def reverse_kl(
    student_logp: np.ndarray,
    teacher_logp: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """KL(student || teacher)."""
    student_p = np.exp(student_logp)
    per_token = (student_p * (student_logp - teacher_logp)).sum(axis=-1)
    return _masked_mean(per_token, mask)


def generalized_jsd(
    student_logp: np.ndarray,
    teacher_logp: np.ndarray,
    beta: float,
    mask: np.ndarray | None = None,
) -> float:
    """Generalized Jensen-Shannon divergence, TRL GKD semantics.

    beta = 0 -> forward KL; beta = 1 -> reverse KL (explicit branches, since
    the mixture limit is degenerate). Otherwise:
        m   = beta * teacher + (1 - beta) * student          (in prob space)
        jsd = beta * KL(teacher || m) + (1 - beta) * KL(student || m)
    """
    if beta == 0.0:
        return forward_kl(student_logp, teacher_logp, mask)
    if beta == 1.0:
        return reverse_kl(student_logp, teacher_logp, mask)

    log_m = np.logaddexp(
        np.log(beta) + teacher_logp, np.log1p(-beta) + student_logp
    )
    teacher_p = np.exp(teacher_logp)
    student_p = np.exp(student_logp)
    kl_teacher = (teacher_p * (teacher_logp - log_m)).sum(axis=-1)
    kl_student = (student_p * (student_logp - log_m)).sum(axis=-1)
    per_token = beta * kl_teacher + (1.0 - beta) * kl_student
    return _masked_mean(per_token, mask)
