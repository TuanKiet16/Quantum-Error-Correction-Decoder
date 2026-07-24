from fastapi.testclient import TestClient
from qec_decoder.api.server import app

client = TestClient(app)


def test_inject_random_returns_layout():
    r = client.post("/inject", json={"d": 3, "mode": "random", "p": 0.01})
    assert r.status_code == 200
    body = r.json()
    assert "detection_events" in body and "layout" in body


def test_decode_mwpm():
    inj = client.post("/inject", json={"d": 3, "mode": "random", "p": 0.01}).json()
    r = client.post("/decode", json={"d": 3,
                    "detection_events": inj["detection_events"],
                    "decoder": "mwpm"})
    assert r.status_code == 200
    assert "latency_ms" in r.json() and "success" in r.json()


def test_decode_cnn_with_checkpoint(tmp_path, monkeypatch):
    import qec_decoder.api.server as srv
    from qec_decoder import train
    train.train("cnn", d=3, ps=[0.01], shots_per_p=64, epochs=1, out_dir=str(tmp_path), seed=1)
    monkeypatch.setattr(srv, "CKPT_DIR", str(tmp_path))
    inj = client.post("/inject", json={"d": 3, "mode": "random", "p": 0.01}).json()
    r = client.post("/decode", json={"d": 3, "detection_events": inj["detection_events"], "decoder": "cnn"})
    assert r.status_code == 200 and "correction" in r.json()
