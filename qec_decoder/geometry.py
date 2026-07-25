"""Order detectors by their physical location on the surface-code lattice.

Stim assigns each detector a coordinate (x, y, t) — its spatial position and the
measurement round. The raw detector index order is not spatially coherent, so a
decoder that treats the syndrome as a flat vector (Conv1d, or QCNN patches cut
from contiguous slices) sees neighbours scattered far apart. Sorting detectors by
(t, y, x) makes adjacent entries physically adjacent, giving those models a
locality inductive bias without any fragile 2-D grid reconstruction.

The ordering depends only on the code distance (noise strength does not move
detectors), so a fixed p is used when building the reference circuit.
"""
import numpy as np
from qec_decoder.data_gen import build_circuit

_REF_P = 0.005


def detector_order(d: int) -> np.ndarray:
    """Permutation of detector indices sorted by (round t, y, x)."""
    coords = build_circuit(d, _REF_P).get_detector_coordinates()
    idx = sorted(coords, key=lambda i: (coords[i][2], coords[i][1], coords[i][0]))
    return np.asarray(idx, dtype=np.int64)
