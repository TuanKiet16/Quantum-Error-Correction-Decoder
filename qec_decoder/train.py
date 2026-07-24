import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from qec_decoder import data_gen
from qec_decoder.seed import set_seed, SEED
from qec_decoder.models.cnn import CNNDecoder
from qec_decoder.models.qcnn_cong import QCNNCong
from qec_decoder.models.qcnn_hybrid import QCNNHybrid


def build_model(name: str, n_detectors: int) -> nn.Module:
    if name == "cnn":
        return CNNDecoder(n_detectors)
    if name == "qcnn_cong":
        return QCNNCong(n_detectors)
    if name == "qcnn_hybrid":
        return QCNNHybrid(n_detectors)
    raise ValueError(f"unknown model {name}")


def make_dataset(d: int, ps, shots_per_p: int, seed: int = SEED):
    Xs, ys = [], []
    for i, p in enumerate(ps):
        dets, obs = data_gen.generate(d, p, shots_per_p, seed + i)
        Xs.append(dets.astype(np.float32))
        ys.append(obs[:, 0].astype(np.float32))   # first logical observable
    return np.concatenate(Xs), np.concatenate(ys)


def train(name, d, ps, shots_per_p, epochs, out_dir="checkpoints", seed=SEED):
    set_seed(seed)
    X, y = make_dataset(d, ps, shots_per_p, seed)
    n_detectors = X.shape[1]
    model = build_model(name, n_detectors)
    Xt = torch.tensor(X)
    yt = torch.tensor(y).unsqueeze(1)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}_d{d}.pt")
    torch.save({"state_dict": model.state_dict(), "name": name, "d": d,
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
    a = ap.parse_args()
    path = train(a.model, a.d, a.ps, a.shots, a.epochs, a.out)
    print(f"saved {path}")


if __name__ == "__main__":
    main()
