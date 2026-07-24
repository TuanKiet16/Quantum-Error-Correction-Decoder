from qec_decoder import baseline


def test_evaluate_returns_sane_rate():
    out = baseline.evaluate(d=3, p=0.01, shots=500, seed=1)
    assert 0.0 <= out["logical_error_rate"] <= 0.5
    assert out["latency_ms"] > 0.0


def test_low_p_lower_error_than_high_p():
    lo = baseline.evaluate(d=3, p=0.002, shots=2000, seed=1)["logical_error_rate"]
    hi = baseline.evaluate(d=3, p=0.03, shots=2000, seed=1)["logical_error_rate"]
    assert lo <= hi
