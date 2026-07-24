import numpy as np
from qec_decoder.seed import set_seed, SEED


def test_seed_makes_numpy_deterministic():
    set_seed()
    a = np.random.rand(5)
    set_seed()
    b = np.random.rand(5)
    assert np.allclose(a, b)
    assert SEED == 20240724
