import numpy as np
import torch
from qec_decoder import geometry, train, inference
from qec_decoder.data_gen import build_circuit
from qec_decoder.models.qcnn_cong import QCNNCong


def test_detector_order_is_a_permutation():
    order = geometry.detector_order(3)
    assert sorted(order.tolist()) == list(range(build_circuit(3, 0.005).num_detectors))


def test_detector_order_sorted_by_t_y_x():
    coords = build_circuit(5, 0.005).get_detector_coordinates()
    order = geometry.detector_order(5)
    keys = [(coords[i][2], coords[i][1], coords[i][0]) for i in order]
    assert keys == sorted(keys)


def test_default_order_is_identity():
    m = QCNNCong(24)   # no detector_order -> identity buffer
    assert torch.equal(m.det_perm, torch.arange(24))


def test_geometry_buffer_round_trips_through_checkpoint(tmp_path):
    # Trained with the geometry order; a fresh identity-built model must recover
    # that order from the checkpoint on load (so inference needs no reorder).
    order = geometry.detector_order(3)
    path = train.train("qcnn_cong", d=3, ps=[0.01], shots_per_p=64, epochs=1,
                       out_dir=str(tmp_path), seed=1)
    m, _ = inference.load_model(path)
    assert np.array_equal(m.det_perm.cpu().numpy(), order)
