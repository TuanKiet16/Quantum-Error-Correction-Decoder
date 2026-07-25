import numpy as np


def logical_error_rate(preds: np.ndarray, labels: np.ndarray) -> float:
    preds = np.asarray(preds).astype(bool)
    labels = np.asarray(labels).astype(bool)
    # a shot is wrong if any observable is mispredicted
    wrong = np.any(preds != labels, axis=tuple(range(1, preds.ndim)))
    return float(np.mean(wrong))


def binomial_uncertainty(p_L: float, N: int) -> float:
    return float(np.sqrt(p_L * (1 - p_L) / N))


def epsilon_per_cycle(p_L: float, rounds: int) -> float:
    t = rounds
    # For p_L > 0.5 the base (1 - 2 p_L) is negative and the fractional power is
    # complex; a decoder that bad has no meaningful per-cycle rate, so clamp the
    # base at 0 — epsilon saturates at 0.5 (maximal per-cycle error).
    base = max(1 - 2 * p_L, 0.0)
    return float((1 - base ** (1 / t)) / 2)


def fit_epsilon_per_cycle(pL_by_t: dict) -> float:
    ts = np.array(sorted(pL_by_t))
    pL = np.array([pL_by_t[t] for t in ts])
    y = np.log(1 - 2 * pL)          # = t * ln(1 - 2 eps)
    slope = np.polyfit(ts, y, 1)[0]  # slope = ln(1 - 2 eps)
    return float((1 - np.exp(slope)) / 2)


def suppression_factor(eps_by_d: dict) -> float:
    ds = np.array(sorted(eps_by_d))
    x = (ds + 1) / 2
    y = np.log(np.array([eps_by_d[d] for d in ds]))
    slope = np.polyfit(x, y, 1)[0]   # m = -ln Lambda
    return float(np.exp(-slope))


def fidelity(p_L: float) -> float:
    return float(1 - p_L)


def latency_stats(times_s) -> dict:
    ms = np.asarray(times_s) * 1000.0
    return {"mean_ms": float(np.mean(ms)), "p95_ms": float(np.percentile(ms, 95))}
