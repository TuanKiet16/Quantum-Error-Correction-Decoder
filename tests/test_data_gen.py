import numpy as np
from qec_decoder import data_gen


def test_generate_shapes_and_determinism():
    dets1, obs1 = data_gen.generate(d=3, p=0.01, shots=100, seed=1)
    dets2, obs2 = data_gen.generate(d=3, p=0.01, shots=100, seed=1)
    assert dets1.shape[0] == 100 and obs1.shape[0] == 100
    assert dets1.dtype == bool and obs1.dtype == bool
    assert obs1.shape[1] >= 1
    assert np.array_equal(dets1, dets2)  # seeded determinism


def test_save_npz_roundtrip(tmp_path):
    dets, obs = data_gen.generate(d=3, p=0.01, shots=50, seed=1)
    path = data_gen.save_npz(str(tmp_path), 3, 0.01, dets, obs)
    loaded = np.load(path)
    assert np.array_equal(loaded["detection_events"], dets)
    assert int(loaded["d"]) == 3
