"""Render the QCNN quantum circuits to PDF via PennyLane's matplotlib drawer.

Both QCNN decoders share the same conv/pool/conv quantum block; they differ
only in wire count (Cong=12, Hybrid=10) and how features reach it. We draw the
quantum circuit each model actually executes.
"""
import os
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt

from qec_decoder.models import encoding
from qec_decoder.models.qcnn_cong import _conv_layer, _pool_layer

OUT_DIR = "figures"


def make_qnode(q):
    dev = qml.device("lightning.qubit", wires=q)

    def circuit(inputs, conv1, pool1, conv2):
        encoding.angle_encode([inputs[..., i] for i in range(q)], wires=range(q))
        wires = list(range(q))
        _conv_layer(conv1, wires)
        wires = _pool_layer(pool1, wires)
        if len(wires) > 1:
            _conv_layer(conv2, wires)
        return qml.expval(qml.PauliZ(wires[0]))

    return qml.QNode(circuit, dev)


def draw(q, title, fname):
    qnode = make_qnode(q)
    rng = np.random.default_rng(20240724)
    inputs = rng.random(q)
    conv1 = rng.random((q, 2))
    pool1 = rng.random(q // 2)
    conv2 = rng.random((q // 2, 2))

    fig, ax = qml.draw_mpl(qnode, decimals=2, style="pennylane")(
        inputs, conv1, pool1, conv2
    )
    fig.suptitle(title, fontsize=14)
    path = os.path.join(OUT_DIR, fname)
    fig.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    draw(12, "QCNN-Cong quantum circuit (12 qubits)", "qcnn_cong_circuit.pdf")
    draw(10, "QCNN-Hybrid quantum head (10 qubits)", "qcnn_hybrid_circuit.pdf")


if __name__ == "__main__":
    main()
