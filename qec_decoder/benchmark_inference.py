import argparse
import time
import numpy as np
from qec_decoder import data_gen, inference, metrics


def benchmark(ckpt_path: str, d: int, p: float, repeats: int = 50) -> dict:
    model, meta = inference.load_model(ckpt_path)
    dets, _ = data_gen.generate(d, p, repeats, seed=1)
    times = []
    for row in dets:
        t0 = time.perf_counter()
        inference.predict_single(model, row)
        times.append(time.perf_counter() - t0)
    stats = metrics.latency_stats(times)
    is_qcnn = meta["name"].startswith("qcnn")
    stats["warn"] = bool(is_qcnn and stats["mean_ms"] > 500.0)
    if stats["warn"]:
        print("QCNN inference is slow, consider caching results for the demo")
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--p", type=float, default=0.01)
    ap.add_argument("--repeats", type=int, default=50)
    a = ap.parse_args()
    print(benchmark(a.ckpt, a.d, a.p, a.repeats))


if __name__ == "__main__":
    main()
