import numpy as np
import pennylane as qml
import pytest
from qec_decoder.models import encoding


def test_angle_encode_flips_qubit_on_one():
    dev = qml.device("lightning.qubit", wires=2)

    @qml.qnode(dev)
    def circ(f):
        encoding.angle_encode(f, wires=[0, 1])
        return [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))]

    z0, z1 = circ(np.array([1.0, 0.0]))
    assert z0 == pytest.approx(-1.0, abs=1e-6)  # RY(pi) -> |1>
    assert z1 == pytest.approx(1.0, abs=1e-6)   # RY(0)  -> |0>
