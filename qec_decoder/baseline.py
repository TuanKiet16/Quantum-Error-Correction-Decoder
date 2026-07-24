import argparse
import time
import numpy as np
import pymatching
from qec_decoder import data_gen, metrics
from qec_decoder.seed import SEED


def build_matching(d: int, p: float) -> pymatching.Matching:
    circuit = data_gen.build_circuit(d, p)
    dem = circuit.detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def decode_batch(matching, dets: np.ndarray) -> np.ndarray:
    preds = matching.decode_batch(dets)
    return np.asarray(preds).astype(bool)


def evaluate(d: int, p: float, shots: int, seed: int = SEED) -> dict:
    matching = build_matching(d, p)
    dets, obs = data_gen.generate(d, p, shots, seed)
    preds = decode_batch(matching, dets)
    if preds.ndim == 1:
        preds = preds[:, None]
    ler = metrics.logical_error_rate(preds, obs)
    # per-shot latency over a small sample
    sample = dets[: min(200, len(dets))]
    times = []
    for row in sample:
        t0 = time.perf_counter()
        matching.decode(row)
        times.append(time.perf_counter() - t0)
    return {"logical_error_rate": ler,
            "latency_ms": metrics.latency_stats(times)["mean_ms"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--shots", type=int, default=20000)
    a = ap.parse_args()
    out = evaluate(a.d, a.p, a.shots)
    print(out)


if __name__ == "__main__":
    main()
