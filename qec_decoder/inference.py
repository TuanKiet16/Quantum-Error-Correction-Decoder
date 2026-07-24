import numpy as np
import torch
from qec_decoder import train


def load_model(ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = train.build_model(ckpt["name"], ckpt["n_detectors"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def predict_single(model, det_row: np.ndarray) -> int:
    x = torch.tensor(np.asarray(det_row, dtype=np.float32)).unsqueeze(0)
    with torch.no_grad():
        logit = model(x)
    return int(logit.item() > 0.0)
