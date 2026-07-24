import json
from qec_decoder import precompute


def test_sweep_mwpm_structure(tmp_path):
    res = precompute.sweep_mwpm(ds=[3], ps=[0.005, 0.02], shots=500, seed=1)
    assert res["points"] and "epsilon_d" in res["points"][0]
    out = precompute.write_threshold(res, str(tmp_path / "threshold.json"))
    assert json.load(open(out))["points"]
