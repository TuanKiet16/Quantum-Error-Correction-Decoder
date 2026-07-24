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
