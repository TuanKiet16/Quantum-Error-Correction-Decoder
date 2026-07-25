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
    def __init__(self, n_detectors: int, patch_qubits: int = 12,
                 detector_order=None):
        super().__init__()
        perm = (torch.arange(n_detectors) if detector_order is None
                else torch.as_tensor(detector_order, dtype=torch.long))
        self.register_buffer("det_perm", perm)   # geometry ordering, saved in ckpt
        self.patch_qubits = patch_qubits
        self.n_patches = patching.n_patches(n_detectors, patch_qubits)
        # One quantum circuit runs per patch, so a batch of B samples issues
        # B * n_patches circuits — the driver of GPU memory. Training reads this
        # to bound its micro-batch size at large d.
        self.circuits_per_sample = self.n_patches
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
            # Multi-observable readout: one expval per surviving wire, so each
            # patch yields a feature vector instead of a single scalar — the old
            # single-Z readout compressed a 12-detector patch to one number.
            return [qml.expval(qml.PauliZ(w)) for w in wires]

        self.n_meas = q // 2            # _pool_layer keeps wires[::2]
        qnode = qml.QNode(circuit, dev, interface="torch", diff_method=diff_method)
        weight_shapes = {"conv1": (q, 2), "pool1": (q // 2,), "conv2": (q // 2, 2)}
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)
        # MLP head over the full per-patch feature vectors (n_patches * n_meas),
        # replacing the linear head that saw one number per patch.
        hidden = 64
        self.head = nn.Sequential(
            nn.Linear(self.n_patches * self.n_meas, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_detectors] -> patches [B, n_patches, q]
        x = x[:, self.det_perm]     # reorder so patches are lattice neighbourhoods
        patches = patching.make_patches(x.detach().cpu().numpy(), self.patch_qubits)
        patches = torch.tensor(patches, dtype=torch.float32, device=x.device)
        B, K, q = patches.shape
        flat = patches.reshape(B * K, q)
        out = self.qlayer(flat)                 # [B*K, n_meas]
        out = out.reshape(B, K * self.n_meas)   # concat patch features per sample
        return self.head(out)
