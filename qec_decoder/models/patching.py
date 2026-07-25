import numpy as np

PATCH_MAX = 16


def n_patches(n_detectors: int, patch_qubits: int = 12) -> int:
    return int(np.ceil(n_detectors / patch_qubits))


def make_patches(dets: np.ndarray, patch_qubits: int = 12) -> np.ndarray:
    if patch_qubits > PATCH_MAX:
        raise ValueError(f"patch_qubits={patch_qubits} exceeds cap {PATCH_MAX}")
    dets = np.asarray(dets).astype(np.float32)
    shots, D = dets.shape
    k = n_patches(D, patch_qubits)
    padded = np.zeros((shots, k * patch_qubits), dtype=np.float32)
    padded[:, :D] = dets
    return padded.reshape(shots, k, patch_qubits)
