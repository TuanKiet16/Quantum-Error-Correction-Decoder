import torch
from qec_decoder.models.cnn import CNNDecoder, count_params


def test_cnn_forward_and_param_budget():
    model = CNNDecoder(n_detectors=12)
    x = torch.rand(4, 12)
    y = model(x)
    assert y.shape == (4, 1)
    assert count_params(model) < 200_000


def test_qcnn_cong_forward():
    import torch
    from qec_decoder.models.qcnn_cong import QCNNCong
    model = QCNNCong(n_detectors=12, patch_qubits=12)
    x = torch.rand(2, 12)
    y = model(x)
    assert y.shape == (2, 1)
