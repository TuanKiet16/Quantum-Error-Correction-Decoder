import argparse
import json
import os
import numpy as np
import stim
from qec_decoder import data_gen, inference


def manual_error_scenarios(d: int) -> list:
    n_data = d * d
    rng = np.random.default_rng(d)
    scenarios = []
    for _ in range(10):
        q = int(rng.integers(0, n_data))
        typ = "X" if rng.random() < 0.5 else "Z"
        scenarios.append([[q, typ]])
    return scenarios


def detection_from_errors(d: int, errors: list) -> list:
    base = data_gen.build_circuit(d, 0.0)
    inject = stim.Circuit()
    for q, typ in errors:
        inject.append(f"{typ}_ERROR", [q], 1.0)   # probability-1 error
    circuit = inject + base
    sampler = circuit.compile_detector_sampler(seed=1)
    dets, _ = sampler.sample(1, separate_observables=True)
    return [int(b) for b in dets[0]]


def _det_key(det: list) -> str:
    return "".join(str(b) for b in det)


def build_cache(ckpt_paths: dict, ds: list, out: str = "results/demo_cache.json"):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cache = {}
    for decoder, ckpt in ckpt_paths.items():
        model, _ = inference.load_model(ckpt)
        for d in ds:
            for errors in manual_error_scenarios(d):
                det = detection_from_errors(d, errors)
                pred = inference.predict_single(model, np.array(det, np.float32))
                cache[f"{decoder}:{d}:{_det_key(det)}"] = pred
    json.dump(cache, open(out, "w"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True,
                    help="decoder=path pairs, e.g. cnn=checkpoints/cnn_d3.pt")
    ap.add_argument("--ds", type=int, nargs="+", default=[3])
    ap.add_argument("--out", default="results/demo_cache.json")
    a = ap.parse_args()
    ckpts = dict(kv.split("=", 1) for kv in a.ckpt)
    print(build_cache(ckpts, a.ds, a.out))


if __name__ == "__main__":
    main()
