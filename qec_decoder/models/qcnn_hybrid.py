import pennylane as qml
import torch
import torch.nn as nn
from qec_decoder.models import encoding
from qec_decoder.models.qcnn_cong import _conv_layer, _pool_layer
from qec_decoder.models.qdevice import make_device


class QCNNHybrid(nn.Module):
    def __init__(self, n_detectors: int, n_qubits: int = 10, detector_order=None):
        super().__init__()
        assert 10 <= n_qubits <= 16
        perm = (torch.arange(n_detectors) if detector_order is None
                else torch.as_tensor(detector_order, dtype=torch.long))
        self.register_buffer("det_perm", perm)   # geometry ordering, saved in ckpt
        self.n_qubits = n_qubits
        # Classical conv reduces each sample to one quantum circuit (not per
        # patch), so the quantum batch equals the sample batch.
        self.circuits_per_sample = 1
        self.reduce = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(n_qubits),
        )
        self.to_features = nn.Linear(8 * n_qubits, n_qubits)

        q = n_qubits
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
            # Read every surviving wire, not just one, so the quantum head
            # exposes a feature vector to the classifier.
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        self.n_meas = q // 2            # _pool_layer keeps wires[::2]
        qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
        weight_shapes = {"conv1": (q, 2), "pool1": (q // 2,), "conv2": (q // 2, 2)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        hidden = 32
        self.head = nn.Sequential(
            nn.Linear(self.n_meas, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x[:, self.det_perm]                    # reorder detectors by geometry
        h = self.reduce(x.unsqueeze(1))            # [B, 8, n_qubits]
        h = h.flatten(1)
        feat = torch.sigmoid(self.to_features(h))  # [B, n_qubits] in [0,1]
        q_out = self.qlayer(feat)                  # [B, n_meas]
        return self.head(q_out)
