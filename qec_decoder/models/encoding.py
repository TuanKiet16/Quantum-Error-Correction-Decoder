import numpy as np
import pennylane as qml


def angle_encode(features, wires) -> None:
    for f, w in zip(features, wires):
        qml.RY(np.pi * f, wires=w)
