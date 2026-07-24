import argparse
import os
import numpy as np
import stim
from qec_decoder.seed import SEED


def build_circuit(d: int, p: float, rounds: int | None = None) -> stim.Circuit:
    rounds = d if rounds is None else rounds
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=rounds,
        after_clifford_depolarization=p,
        after_reset_flip_probability=p,
        before_measure_flip_probability=p,
        before_round_data_depolarization=p,
    )


def generate(d: int, p: float, shots: int, seed: int = SEED):
    circuit = build_circuit(d, p)
    sampler = circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    return dets.astype(bool), obs.astype(bool)


def save_npz(out_dir: str, d: int, p: float, dets, obs) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"d{d}_p{p}.npz")
    np.savez_compressed(path, detection_events=dets, observable_flips=obs,
                        d=d, p=p)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=200000)
    ap.add_argument("--out", type=str, default="data/")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    dets, obs = generate(a.d, a.p, a.shots, a.seed)
    path = save_npz(a.out, a.d, a.p, dets, obs)
    print(f"saved {path}  dets={dets.shape} obs={obs.shape}")


if __name__ == "__main__":
    main()
