import pennylane as qml
import torch
import torch.nn as nn
from qec_decoder.models import patching, encoding
from qec_decoder.models.qdevice import make_device


def _conv_layer(params, wires):
    # parameterized 2-qubit gates on neighboring pairs
    for i in range(0, len(wires) - 1, 2):
        _two_qubit(params[i], wires[i], wires[i + 1])
    for i in range(1, len(wires) - 1, 2):
        _two_qubit(params[i], wires[i], wires[i + 1])


def _two_qubit(p, a, b):
    qml.RY(p[0], wires=a)
    qml.RY(p[1], wires=b)
    qml.CNOT(wires=[a, b])


def _pool_layer(params, wires):
    # measure-out odd wires by conditioning even wires, keep even wires
    kept = wires[::2]
    for j, src in enumerate(wires[1::2]):
        tgt = wires[2 * j]
        qml.CRZ(params[j], wires=[src, tgt])
    return kept


class QCNNCong(nn.Module):
    def __init__(self, n_detectors: int, patch_qubits: int = 12):
        super().__init__()
        self.patch_qubits = patch_qubits
        self.n_patches = patching.n_patches(n_detectors, patch_qubits)
        q = patch_qubits
        dev, diff_method = make_device(q)

        def circuit(inputs, conv1, pool1, conv2):
            # Index the feature (last) axis so a batched [B, q] input broadcasts
            # per-wire; angle_encode's zip otherwise iterates the batch axis.
            encoding.angle_encode([inputs[..., i] for i in range(q)], wires=range(q))
            wires = list(range(q))
            _conv_layer(conv1, wires)
            wires = _pool_layer(pool1, wires)
            if len(wires) > 1:
                _conv_layer(conv2, wires)
            return qml.expval(qml.PauliZ(wires[0]))

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
        weight_shapes = {"conv1": (q, 2), "pool1": (q // 2,), "conv2": (q // 2, 2)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.head = nn.Linear(self.n_patches, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_detectors] -> patches [B, n_patches, q]
        patches = patching.make_patches(x.detach().cpu().numpy(), self.patch_qubits)
        patches = torch.tensor(patches, dtype=torch.float32, device=x.device)
        B, K, q = patches.shape
        flat = patches.reshape(B * K, q)
        out = self.qlayer(flat)                 # [B*K]
        out = out.reshape(B, K)
        return self.head(out)
