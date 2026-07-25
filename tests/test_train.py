import torch
from qec_decoder import train


def test_build_model_detector_count():
    m = train.build_model("cnn", n_detectors=12)
    assert m(torch.rand(2, 12)).shape == (2, 1)


def test_make_dataset_mixes_p():
    X, y = train.make_dataset(d=3, ps=[0.005, 0.02], shots_per_p=50, seed=1)
    assert X.shape[0] == 100 and y.shape[0] == 100


def test_train_smoke_saves_checkpoint(tmp_path):
    # tiny 1-epoch CNN run to keep it fast; not a training-quality assertion
    path = train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    assert path.endswith("cnn_d3.pt")
    ckpt = torch.load(path, map_location="cpu")
    assert "state_dict" in ckpt and ckpt["n_detectors"] > 0


class _Model:
    def __init__(self, cps):
        self.circuits_per_sample = cps


def test_micro_batch_caps_quantum_circuits():
    # cps grows with d for Cong; micro-batch must shrink so cps*mb <= qchunk
    m = _Model(cps=28)  # ~ d=7
    mb = train.micro_batch_size(m, batch_size=256, qchunk=2048)
    assert mb == 2048 // 28
    assert mb * m.circuits_per_sample <= 2048


def test_micro_batch_never_accumulates_for_classical():
    # cps=1 (cnn/hybrid): cap never bites, micro-batch == full batch
    assert train.micro_batch_size(_Model(cps=1), batch_size=256, qchunk=2048) == 256


def test_micro_batch_at_least_one():
    # pathological cps larger than qchunk still yields a runnable micro-batch
    assert train.micro_batch_size(_Model(cps=5000), batch_size=256, qchunk=2048) == 1


def test_grad_accumulation_matches_full_batch_gradient():
    # Accumulated micro-batch gradients must equal the full-batch mean gradient.
    import torch.nn as nn
    torch.manual_seed(0)
    net = nn.Linear(4, 1)
    x, y = torch.randn(6, 4), torch.randint(0, 2, (6, 1)).float()
    loss_fn = nn.BCEWithLogitsLoss()

    net.zero_grad()
    loss_fn(net(x), y).backward()
    full = torch.cat([p.grad.flatten() for p in net.parameters()]).clone()

    net.zero_grad()
    for s in range(0, 6, 2):                       # micro-batches of 2
        xb, yb = x[s:s + 2], y[s:s + 2]
        (loss_fn(net(xb), yb) * (xb.shape[0] / 6)).backward()
    accum = torch.cat([p.grad.flatten() for p in net.parameters()])
    assert torch.allclose(full, accum, atol=1e-6)
