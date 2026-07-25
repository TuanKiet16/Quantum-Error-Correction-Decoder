"""Select the PennyLane simulator backend for the QCNN circuits.

Default is CPU `lightning.qubit` with adjoint differentiation — unchanged from
the original build, so existing tests and HPC runs behave identically. Set
`QEC_QML_DEVICE=default.qubit` to switch to the Torch-native statevector, which
runs on whatever device the model's tensors live on (e.g. CUDA on Kaggle) and
differentiates via backprop. Any `lightning.*` name keeps adjoint diff.
"""
import os
import pennylane as qml

ENV_VAR = "QEC_QML_DEVICE"


def make_device(q: int):
    """Return (device, diff_method) for a q-wire QCNN circuit."""
    name = os.environ.get(ENV_VAR, "lightning.qubit")
    dev = qml.device(name, wires=q)
    # default.qubit is Torch-differentiable (runs on the tensor's device);
    # the lightning family uses the adjoint method on its own C++/CUDA backend.
    diff_method = "backprop" if name == "default.qubit" else "adjoint"
    return dev, diff_method
