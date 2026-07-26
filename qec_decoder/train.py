import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
from qec_decoder import data_gen, geometry
from qec_decoder.runlog import get_logger
from qec_decoder.seed import set_seed, SEED
from qec_decoder.models.cnn import CNNDecoder
from qec_decoder.models.qcnn_cong import QCNNCong
from qec_decoder.models.qcnn_hybrid import QCNNHybrid


def build_model(name: str, n_detectors: int, detector_order=None) -> nn.Module:
    if name == "cnn":
        return CNNDecoder(n_detectors, detector_order)
    if name == "qcnn_cong":
        return QCNNCong(n_detectors, detector_order=detector_order)
    if name == "qcnn_hybrid":
        return QCNNHybrid(n_detectors, detector_order=detector_order)
    raise ValueError(f"unknown model {name}")


def make_dataset(d: int, ps, shots_per_p: int, seed: int = SEED):
    Xs, ys = [], []
    for i, p in enumerate(ps):
        dets, obs = data_gen.generate(d, p, shots_per_p, seed + i)
        Xs.append(dets.astype(np.float32))
        ys.append(obs[:, 0].astype(np.float32))   # first logical observable
    return np.concatenate(Xs), np.concatenate(ys)


def micro_batch_size(model, batch_size: int, qchunk: int) -> int:
    """Largest micro-batch whose quantum-circuit count stays under `qchunk`.

    QCNN-Cong issues `circuits_per_sample` (= n_patches) circuits per sample,
    which grows with the code distance and drives GPU memory. Capping the
    circuits per forward keeps peak memory ~constant across d; classical/hybrid
    models (cps=1) are unaffected because the cap never bites.
    """
    cps = int(getattr(model, "circuits_per_sample", 1))
    mb = max(1, qchunk // max(1, cps))
    return min(batch_size, mb)


def train(name, d, ps, shots_per_p, epochs, out_dir="checkpoints", seed=SEED,
          batch_size: int = 256, device: str = "cpu", qchunk: int = 2048,
          lr: float = 1e-2):
    set_seed(seed)
    # default.qubit builds its statevector on the default torch device; point it
    # at the GPU so the sim runs there instead of raising a device mismatch.
    if str(device).startswith("cuda"):
        torch.set_default_device(device)
    X, y = make_dataset(d, ps, shots_per_p, seed)
    n_detectors = X.shape[1]
    order = geometry.detector_order(d)
    model = build_model(name, n_detectors, order).to(device)
    Xt = torch.tensor(X, device="cpu")
    yt = torch.tensor(y, device="cpu").unsqueeze(1)
    n = Xt.shape[0]
    mb = micro_batch_size(model, batch_size, qchunk)
    # Manual seeded minibatching: DataLoader's sampler builds its shuffle tensor
    # on the default torch device, which clashes with a CPU generator once the
    # default device is CUDA. An explicit CPU randperm avoids that.
    gen = torch.Generator(device="cpu").manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
    # Logical flips are rare at low p; weight the positive class by neg/pos so the
    # decoder can't win by always predicting "no error".
    n_pos = float(yt.sum().item())
    n_neg = float(n - n_pos)
    pos_weight = torch.tensor([n_neg / n_pos if n_pos > 0 else 1.0], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model.train()

    log = get_logger()
    cps = int(getattr(model, "circuits_per_sample", 1))
    n_batches = (n + batch_size - 1) // batch_size
    log_every = max(1, n_batches // 5)     # ~5 progress lines per epoch
    log.info(f"train {name} d{d}: N={n} n_det={n_detectors} "
             f"batches/epoch={n_batches} micro_batch={mb} circuits/sample={cps} "
             f"device={device} epochs={epochs} lr={lr} "
             f"pos_weight={pos_weight.item():.2f}")
    t_train = time.perf_counter()
    for ep in range(epochs):
        ep_t = time.perf_counter()
        perm = torch.randperm(n, generator=gen, device="cpu")
        running = torch.zeros((), device=device)   # accumulate on-device, no sync
        for bi, s in enumerate(range(0, n, batch_size)):
            idx = perm[s:s + batch_size]
            bsz = idx.numel()
            opt.zero_grad()
            # Gradient accumulation over micro-batches: each does its own
            # backward and frees its graph, so peak memory tracks `mb`, not the
            # full batch. Losses are weighted by micro-size / batch-size so the
            # accumulated gradient equals the full-batch mean.
            for ms in range(0, bsz, mb):
                mi = idx[ms:ms + mb]
                Xb = Xt[mi].to(device)
                yb = yt[mi].to(device)
                logits = model(Xb)
                loss = loss_fn(logits, yb) * (mi.numel() / bsz)
                loss.backward()
                running += loss.detach()
            opt.step()
            if (bi + 1) % log_every == 0 or bi + 1 == n_batches:
                el = time.perf_counter() - ep_t
                done = min(s + batch_size, n)
                rate = done * cps / el if el > 0 else 0.0
                log.info(f"  {name} d{d} ep {ep + 1}/{epochs} "
                         f"batch {bi + 1}/{n_batches} loss={running.item() / (bi + 1):.4f} "
                         f"{rate:.0f} circ/s")
        sched.step()
        ep_dur = time.perf_counter() - ep_t
        eta = ep_dur * (epochs - ep - 1)
        log.info(f"{name} d{d} epoch {ep + 1}/{epochs} done "
                 f"loss={running.item() / n_batches:.4f} {ep_dur:.1f}s "
                 f"eta {eta / 60:.1f}min")
    log.info(f"train {name} d{d} finished in {(time.perf_counter() - t_train) / 60:.1f}min")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_d{d}.pt")
    torch.save({"state_dict": model.cpu().state_dict(), "name": name, "d": d,
                "n_detectors": n_detectors}, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["qcnn_cong", "qcnn_hybrid", "cnn"])
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--ps", type=float, nargs="+", required=True)
    ap.add_argument("--shots", type=int, default=5000)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="checkpoints")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda (cuda needs QEC_QML_DEVICE=default.qubit)")
    ap.add_argument("--qchunk", type=int, default=2048,
                    help="max quantum circuits per forward; caps GPU memory at "
                         "large d via gradient accumulation")
    ap.add_argument("--lr", type=float, default=1e-2)
    a = ap.parse_args()
    from qec_decoder.runlog import Run
    with Run("train", vars(a), seed=SEED) as run:
        path = train(a.model, a.d, a.ps, a.shots, a.epochs, a.out,
                     batch_size=a.batch_size, device=a.device, qchunk=a.qchunk,
                     lr=a.lr)
        run.record({"checkpoint": path})
    print(f"saved {path}")


if __name__ == "__main__":
    main()
