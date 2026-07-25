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
