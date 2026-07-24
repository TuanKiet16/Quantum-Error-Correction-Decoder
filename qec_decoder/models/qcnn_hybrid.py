import pennylane as qml
import torch
import torch.nn as nn
from qec_decoder.models import encoding
from qec_decoder.models.qcnn_cong import _conv_layer, _pool_layer


class QCNNHybrid(nn.Module):
    def __init__(self, n_detectors: int, n_qubits: int = 10):
        super().__init__()
        assert 10 <= n_qubits <= 16
        self.n_qubits = n_qubits
        self.reduce = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_qubits),
        )
        self.to_features = nn.Linear(8 * n_qubits, n_qubits)

        q = n_qubits
        dev = qml.device("lightning.qubit", wires=q)

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

        qnode = qml.QNode(circuit, dev, interface="torch", diff_method="adjoint")
        weight_shapes = {"conv1": (q, 2), "pool1": (q // 2,), "conv2": (q // 2, 2)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        self.head = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.reduce(x.unsqueeze(1))            # [B, 8, n_qubits]
        h = h.flatten(1)
        feat = torch.sigmoid(self.to_features(h))  # [B, n_qubits] in [0,1]
        q_out = self.qlayer(feat).reshape(-1, 1)   # [B, 1]
        return self.head(q_out)
