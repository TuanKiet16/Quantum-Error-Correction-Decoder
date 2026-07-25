import numpy as np
import pytest
from qec_decoder.models import patching


def test_patch_qubit_cap_enforced():
    with pytest.raises(ValueError):
        patching.make_patches(np.zeros((2, 30), bool), patch_qubits=20)


def test_patch_shapes_and_padding():
    dets = np.ones((4, 20), dtype=bool)
    out = patching.make_patches(dets, patch_qubits=12)
    assert out.shape == (4, 2, 12)          # ceil(20/12)=2 patches
    assert out[:, 0, :].sum() == 4 * 12     # first patch full of ones
    assert out[:, 1, 8:].sum() == 0         # padding zeros (20-12=8 real)


def test_n_patches():
    assert patching.n_patches(20, 12) == 2
    assert patching.n_patches(8, 12) == 1
