import numpy as np
import pytest

from sdft.online import loss


def make_logps(weights: list[list[float]]) -> np.ndarray:
    """Turn unnormalized weights into log-probabilities, shape [T, V]."""
    logits = np.log(np.asarray(weights, dtype=np.float64))
    return loss.log_softmax(logits, axis=-1)


class TestLogSoftmax:
    def test_rows_sum_to_one(self):
        logits = np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 3.0]])
        logp = loss.log_softmax(logits)
        np.testing.assert_allclose(np.exp(logp).sum(-1), [1.0, 1.0], rtol=1e-12)

    def test_extreme_values_stable(self):
        logits = np.array([[1000.0, -1000.0, 0.0]])
        logp = loss.log_softmax(logits)
        assert np.isfinite(logp).all()
        assert logp[0, 0] == pytest.approx(0.0, abs=1e-12)


class TestForwardKL:
    def test_identical_distributions_zero(self):
        p = make_logps([[0.2, 0.5, 0.3]])
        assert loss.forward_kl(p, p) == pytest.approx(0.0, abs=1e-12)

    def test_hand_computed(self):
        # V = 2, single token. teacher = [0.25, 0.75], student = [0.5, 0.5]
        teacher = make_logps([[0.25, 0.75]])
        student = make_logps([[0.5, 0.5]])
        expected = 0.25 * np.log(0.25 / 0.5) + 0.75 * np.log(0.75 / 0.5)
        assert loss.forward_kl(student, teacher) == pytest.approx(expected, rel=1e-10)

    def test_mask(self):
        teacher = make_logps([[0.25, 0.75], [0.9, 0.1]])
        student = make_logps([[0.5, 0.5], [0.9, 0.1]])
        # Second token: identical -> contributes 0. Masking the first -> loss 0.
        mask = np.array([0.0, 1.0])
        assert loss.forward_kl(student, teacher, mask) == pytest.approx(0.0, abs=1e-12)

    def test_zero_mask_raises(self):
        p = make_logps([[0.5, 0.5]])
        with pytest.raises(ValueError):
            loss.forward_kl(p, p, mask=np.array([0.0]))


class TestReverseKL:
    def test_hand_computed(self):
        teacher = make_logps([[0.25, 0.75]])
        student = make_logps([[0.5, 0.5]])
        expected = 0.5 * np.log(0.5 / 0.25) + 0.5 * np.log(0.5 / 0.75)
        assert loss.reverse_kl(student, teacher) == pytest.approx(expected, rel=1e-10)

    def test_asymmetry(self):
        teacher = make_logps([[0.1, 0.9]])
        student = make_logps([[0.5, 0.5]])
        fwd = loss.forward_kl(student, teacher)
        rev = loss.reverse_kl(student, teacher)
        assert fwd != pytest.approx(rev)


class TestGeneralizedJSD:
    def test_endpoints_match_kls(self):
        teacher = make_logps([[0.3, 0.7], [0.6, 0.4]])
        student = make_logps([[0.5, 0.5], [0.2, 0.8]])
        assert loss.generalized_jsd(student, teacher, 0.0) == pytest.approx(
            loss.forward_kl(student, teacher)
        )
        assert loss.generalized_jsd(student, teacher, 1.0) == pytest.approx(
            loss.reverse_kl(student, teacher)
        )

    def test_half_is_symmetric(self):
        teacher = make_logps([[0.3, 0.7]])
        student = make_logps([[0.6, 0.4]])
        assert loss.generalized_jsd(student, teacher, 0.5) == pytest.approx(
            loss.generalized_jsd(teacher, student, 0.5), rel=1e-10
        )

    def test_identical_zero(self):
        p = make_logps([[0.3, 0.7]])
        assert loss.generalized_jsd(p, p, 0.3) == pytest.approx(0.0, abs=1e-12)
