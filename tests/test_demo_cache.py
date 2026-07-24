import json
from qec_decoder import train, precompute_demo_cases as pdc


def test_detection_from_errors_deterministic():
    a = pdc.detection_from_errors(3, [[0, "X"]])
    b = pdc.detection_from_errors(3, [[0, "X"]])
    assert a == b and all(v in (0, 1) for v in a)


def test_build_cache_writes_json(tmp_path):
    ckpt = train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    out = pdc.build_cache({"cnn": ckpt}, ds=[3],
                          out=str(tmp_path / "demo_cache.json"))
    data = json.load(open(out))
    assert any(k.startswith("cnn:3:") for k in data)
