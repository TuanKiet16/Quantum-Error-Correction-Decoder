import json
from qec_decoder import train, precompute_demo_cases as pdc


def test_detection_from_errors_deterministic():
    a = pdc.detection_from_errors(3, [[0, "X"]])
    b = pdc.detection_from_errors(3, [[0, "X"]])
    assert a == b and all(v in (0, 1) for v in a)


def test_scenarios_produce_distinct_nonzero_detections():
    from qec_decoder.precompute_demo_cases import manual_error_scenarios, detection_from_errors
    for d in (3, 5):
        pats = [tuple(detection_from_errors(d, e)) for e in manual_error_scenarios(d)]
        nonzero = [p for p in pats if any(p)]
        assert len(nonzero) >= len(pats) // 2, f"d={d}: too many all-zero detections"
        assert len(set(pats)) > 1, f"d={d}: detection patterns not distinct"


def test_build_cache_writes_json(tmp_path):
    ckpt = train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    out = pdc.build_cache({"cnn": ckpt}, ds=[3],
                          out=str(tmp_path / "demo_cache.json"))
    data = json.load(open(out))
    assert any(k.startswith("cnn:3:") for k in data)
