import numpy as np
from qec_decoder import train, inference, benchmark_inference


def test_load_and_predict_single(tmp_path):
    path = train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    model, meta = inference.load_model(path)
    det = np.zeros(meta["n_detectors"], dtype=np.float32)
    out = inference.predict_single(model, det)
    assert out in (0, 1)


def test_benchmark_reports_latency(tmp_path):
    path = train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    res = benchmark_inference.benchmark(path, d=3, p=0.01, repeats=5)
    assert res["mean_ms"] > 0 and "warn" in res
