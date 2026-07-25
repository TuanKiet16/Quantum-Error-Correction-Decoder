import argparse
import json
import os
import numpy as np
from qec_decoder import baseline, metrics
from qec_decoder.seed import SEED


def sweep_mwpm(ds, ps, shots, seed=SEED) -> dict:
    points = []
    eps_at_ref = {}
    ref_p = ps[len(ps) // 2]
    for d in ds:
        for p in ps:
            ev = baseline.evaluate(d, p, shots, seed)
            eps = metrics.epsilon_per_cycle(ev["logical_error_rate"], rounds=d)
            points.append({"d": d, "p": float(p),
                           "logical_error_rate": ev["logical_error_rate"],
                           "epsilon_d": eps, "latency_ms": ev["latency_ms"]})
            if abs(p - ref_p) < 1e-12:
                eps_at_ref[d] = eps
    lam = {}
    if len(eps_at_ref) >= 2:
        lam["overall"] = metrics.suppression_factor(eps_at_ref)
    return {"points": points, "lambda": lam, "ref_p": float(ref_p)}


def write_threshold(result: dict, out: str = "results/threshold.json") -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=int, nargs="+", default=[3, 5, 7, 9])
    ap.add_argument("--ps", type=float, nargs="+",
                    default=[1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-2, 1.5e-2, 2e-2])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--out", default="results/threshold.json")
    a = ap.parse_args()
    from qec_decoder.runlog import Run
    with Run("precompute", vars(a)) as run:
        res = sweep_mwpm(a.ds, a.ps, a.shots)
        out = write_threshold(res, a.out)
        run.record({"out": out, "n_points": len(res["points"]),
                    "lambda": res["lambda"]})
    print(out)


if __name__ == "__main__":
    main()
