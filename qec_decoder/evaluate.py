"""Accuracy evaluation for the neural decoders, compared against MWPM.

Scores a trained QCNN/CNN checkpoint by its logical error rate on fresh Stim
samples, then sweeps all available decoders on one shared test set per (d, p) so
the comparison is apples-to-apples. Inference is chunked under `no_grad`, so peak
memory stays flat regardless of shot count or code distance — safe to run on a
laptop CPU.

The test seed is deliberately far from training's (`SEED + i`); see TEST_SEED.
"""
import argparse
import json
import os
import numpy as np
import torch

from qec_decoder import baseline, data_gen, inference, metrics
from qec_decoder.seed import SEED
from qec_decoder.train import micro_batch_size

# Training samples with seed = SEED + p_index. Offset the test draw far away so
# the evaluation set is disjoint from anything the model trained on.
TEST_SEED = SEED + 10_000

NEURAL_NAMES = ("qcnn_cong", "qcnn_hybrid", "cnn")


def predict_batch(model, dets: np.ndarray, qchunk: int = 2048,
                  device: str = "cpu") -> np.ndarray:
    """Logical-flip predictions for a batch, chunked to bound memory.

    QCNN-Cong issues `circuits_per_sample` circuits per row, so the chunk is
    sized the same way training's micro-batch is — peak work per forward is
    capped even at large d. On CUDA (with QEC_QML_DEVICE=default.qubit) each
    chunk's circuits run as one batched contraction, far faster than the
    per-circuit CPU path.
    """
    model.eval()
    x = torch.tensor(np.asarray(dets, dtype=np.float32), device="cpu")
    n = x.shape[0]
    chunk = micro_batch_size(model, n, qchunk)
    preds = []
    with torch.no_grad():
        for s in range(0, n, chunk):
            logits = model(x[s:s + chunk].to(device))
            preds.append((logits > 0.0).cpu().numpy())
    return np.concatenate(preds, axis=0).astype(bool)


def _point(decoder, d, p, preds, obs) -> dict:
    if preds.ndim == 1:
        preds = preds[:, None]
    p_L = metrics.logical_error_rate(preds, obs)
    return {
        "decoder": decoder,
        "d": int(d),
        "p": float(p),
        "logical_error_rate": p_L,
        "uncertainty": metrics.binomial_uncertainty(p_L, len(obs)),
        "epsilon_d": metrics.epsilon_per_cycle(p_L, rounds=d),
        "fidelity": metrics.fidelity(p_L),
    }


def evaluate_model(model, d: int, ps, shots: int, seed: int = TEST_SEED,
                   qchunk: int = 2048, device: str = "cpu") -> list:
    """Per-p logical error rate for one already-loaded neural model."""
    model.to(device)
    points = []
    for i, p in enumerate(ps):
        dets, obs = data_gen.generate(d, p, shots, seed + i)
        preds = predict_batch(model, dets, qchunk, device)
        points.append(_point(_model_name(model), d, p, preds, obs))
    return points


def _model_name(model) -> str:
    return {"QCNNCong": "qcnn_cong", "QCNNHybrid": "qcnn_hybrid",
            "CNNDecoder": "cnn"}.get(type(model).__name__, type(model).__name__)


def sweep(ckpt_dir: str, ds, ps, shots: int, seed: int = TEST_SEED,
          qchunk: int = 2048, device: str = "cpu") -> dict:
    """Evaluate every available decoder on one shared test set per (d, p)."""
    from qec_decoder.runlog import get_logger
    log = get_logger()
    points = []
    eps_ref = {}                       # {decoder: {d: epsilon_d}} at ref_p
    ref_p = ps[len(ps) // 2]
    for d in ds:
        # Load whichever neural checkpoints exist for this distance.
        models = {}
        for name in NEURAL_NAMES:
            path = os.path.join(ckpt_dir, f"{name}_d{d}.pt")
            if os.path.exists(path):
                m, _ = inference.load_model(path)
                models[name] = m.to(device)
        log.info(f"eval d={d}: decoders={['mwpm'] + list(models)} "
                 f"shots={shots} ps={list(ps)}")
        matching = baseline.build_matching(d, ps[0])
        for i, p in enumerate(ps):
            dets, obs = data_gen.generate(d, p, shots, seed + i)
            # MWPM on the identical shots
            mwpm_preds = baseline.decode_batch(matching, dets)
            row = _point("mwpm", d, p, mwpm_preds, obs)
            points.append(row)
            eps_ref.setdefault("mwpm", {})
            # Neural decoders on the same shots
            per_decoder = [("mwpm", row)]
            for name, model in models.items():
                preds = predict_batch(model, dets, qchunk, device)
                r = _point(name, d, p, preds, obs)
                points.append(r)
                per_decoder.append((name, r))
            log.info("  d={} p={:.3f}  ".format(d, p) + "  ".join(
                "{}:pL={:.4f}".format(nm, r["logical_error_rate"])
                for nm, r in per_decoder))
            if abs(p - ref_p) < 1e-12:
                for name, r in per_decoder:
                    eps_ref.setdefault(name, {})[d] = r["epsilon_d"]
    lam = {name: metrics.suppression_factor(by_d)
           for name, by_d in eps_ref.items() if len(by_d) >= 2}
    return {"points": points, "lambda": lam, "ref_p": float(ref_p),
            "test_seed": int(seed), "shots": int(shots)}


def write(result: dict, out: str = "results/comparison.json") -> str:
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(result, open(out, "w"), indent=2)
    return out


def _print_table(result: dict) -> None:
    print(f"{'decoder':<12}{'d':>3}{'p':>8}{'p_L':>10}{'±':>10}")
    for r in result["points"]:
        print(f"{r['decoder']:<12}{r['d']:>3}{r['p']:>8.4f}"
              f"{r['logical_error_rate']:>10.4f}{r['uncertainty']:>10.4f}")
    if result["lambda"]:
        print("Lambda (suppression):",
              {k: round(v, 3) for k, v in result["lambda"].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--ds", type=int, nargs="+", default=[3, 5])
    ap.add_argument("--ps", type=float, nargs="+",
                    default=[0.003, 0.005, 0.008, 0.01, 0.015])
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=TEST_SEED)
    ap.add_argument("--qchunk", type=int, default=2048)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda (cuda needs QEC_QML_DEVICE=default.qubit)")
    ap.add_argument("--out", default="results/comparison.json")
    a = ap.parse_args()
    if str(a.device).startswith("cuda"):
        torch.set_default_device(a.device)
    from qec_decoder.runlog import Run
    with Run("evaluate", vars(a), seed=a.seed) as run:
        res = sweep(a.ckpt_dir, a.ds, a.ps, a.shots, a.seed, a.qchunk, a.device)
        out = write(res, a.out)
        run.record({"out": out, "n_points": len(res["points"]),
                    "lambda": res["lambda"]})
    _print_table(res)
    print(out)


if __name__ == "__main__":
    main()
