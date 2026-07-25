import numpy as np
import pytest
from qec_decoder import metrics


def test_logical_error_rate_half():
    preds = np.array([[0], [1], [0], [1]], dtype=bool)
    labels = np.array([[0], [0], [1], [1]], dtype=bool)
    assert metrics.logical_error_rate(preds, labels) == 0.5


def test_binomial_uncertainty():
    assert metrics.binomial_uncertainty(0.1, 100) == pytest.approx(0.03, abs=1e-9)


def test_epsilon_per_cycle_formula():
    p_L, t = 0.1, 3
    expected = (1 - (1 - 2 * p_L) ** (1 / t)) / 2
    assert metrics.epsilon_per_cycle(p_L, t) == pytest.approx(expected)


def test_fit_epsilon_recovers_slope():
    # construct p_L(t) from a known epsilon via (1-2pL) = (1-2eps)^t
    eps = 0.02
    pL_by_t = {t: (1 - (1 - 2 * eps) ** t) / 2 for t in [5, 10, 20, 40]}
    assert metrics.fit_epsilon_per_cycle(pL_by_t) == pytest.approx(eps, abs=1e-6)


def test_suppression_factor_constant_ratio():
    # eps halves every +2 in distance -> Lambda = 2
    eps_by_d = {3: 0.08, 5: 0.04, 7: 0.02, 9: 0.01}
    assert metrics.suppression_factor(eps_by_d) == pytest.approx(2.0, abs=1e-6)


def test_fidelity():
    assert metrics.fidelity(0.1) == pytest.approx(0.9)


def test_latency_stats():
    out = metrics.latency_stats([0.001, 0.002, 0.003, 0.004])
    assert out["mean_ms"] == pytest.approx(2.5)
    assert out["p95_ms"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))
