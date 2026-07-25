import numpy as np
import torch
from qec_decoder import evaluate, data_gen
from qec_decoder.seed import SEED
from qec_decoder.models.cnn import CNNDecoder
from qec_decoder.models.qcnn_cong import QCNNCong


def test_test_seed_disjoint_from_training():
    # Training draws with SEED + i; the test seed must not collide, else the
    # evaluation set overlaps training data.
    assert evaluate.TEST_SEED != SEED
    train_dets, _ = data_gen.generate(3, 0.01, 128, SEED)
    test_dets, _ = data_gen.generate(3, 0.01, 128, evaluate.TEST_SEED)
    assert not np.array_equal(train_dets, test_dets)


def test_predict_batch_chunking_is_invariant():
    # A per-patch model (Cong) chunked vs in one shot must agree exactly.
    dets, _ = data_gen.generate(5, 0.01, 40, seed=1)
    model = QCNNCong(dets.shape[1])
    whole = evaluate.predict_batch(model, dets, qchunk=10_000)   # one forward
    chunked = evaluate.predict_batch(model, dets, qchunk=64)     # many chunks
    assert np.array_equal(whole, chunked)
    assert whole.shape[0] == 40


def test_predict_batch_thresholds_at_zero():
    dets, _ = data_gen.generate(3, 0.01, 16, seed=2)
    model = CNNDecoder(dets.shape[1])
    preds = evaluate.predict_batch(model, dets)
    assert preds.dtype == bool and preds.shape[0] == 16


def test_evaluate_model_reports_valid_pL():
    model = CNNDecoder(24)
    pts = evaluate.evaluate_model(model, d=3, ps=[0.005, 0.01], shots=200)
    assert len(pts) == 2
    for r in pts:
        assert 0.0 <= r["logical_error_rate"] <= 1.0
        assert r["uncertainty"] >= 0.0
        assert r["decoder"] == "cnn" and r["d"] == 3


def test_sweep_without_checkpoints_still_scores_mwpm(tmp_path):
    # Empty checkpoint dir: MWPM is always evaluated, neural decoders skipped.
    res = evaluate.sweep(str(tmp_path), ds=[3], ps=[0.005, 0.01], shots=300)
    decoders = {p["decoder"] for p in res["points"]}
    assert decoders == {"mwpm"}
    assert len(res["points"]) == 2
    for r in res["points"]:
        assert 0.0 <= r["logical_error_rate"] <= 1.0
