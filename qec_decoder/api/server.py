import json
import os
import time
import numpy as np
from fastapi import FastAPI, HTTPException
from qec_decoder import data_gen, baseline, metrics, inference
from qec_decoder.precompute_demo_cases import detection_from_errors
from qec_decoder.api.schemas import InjectReq, DecodeReq, BatchReq

app = FastAPI(title="QEC Decoder Demo")

CKPT_DIR = os.environ.get("QEC_CKPT_DIR", "checkpoints")
DEMO_CACHE = os.environ.get("QEC_DEMO_CACHE", "results/demo_cache.json")
THRESHOLD = os.environ.get("QEC_THRESHOLD", "results/threshold.json")


def layout(d: int) -> dict:
    circuit = data_gen.build_circuit(d, 0.001)
    coords = circuit.get_detector_coordinates()
    ancillas = [[int(k), float(v[0]), float(v[1])] for k, v in coords.items()]
    qc = circuit.get_final_qubit_coordinates()
    data_qubits = [[int(k), float(v[0]), float(v[1])] for k, v in qc.items()]
    return {"data_qubits": data_qubits, "ancillas": ancillas}


def _load(decoder: str, d: int):
    path = os.path.join(CKPT_DIR, f"{decoder}_d{d}.pt")
    if not os.path.exists(path):
        raise HTTPException(404, f"checkpoint not found: {path}")
    return inference.load_model(path)


@app.post("/inject")
def inject(req: InjectReq):
    if req.mode == "manual":
        if not req.errors:
            raise HTTPException(400, "manual mode requires errors")
        det = detection_from_errors(req.d, req.errors)
    else:
        dets, _ = data_gen.generate(req.d, req.p or 0.01, 1, seed=int(time.time()))
        det = [int(b) for b in dets[0]]
    return {"detection_events": det, "layout": layout(req.d)}


@app.post("/decode")
def decode(req: DecodeReq):
    det = np.array(req.detection_events, dtype=bool)
    if req.decoder == "mwpm":
        m = baseline.build_matching(req.d, 0.01)
        t0 = time.perf_counter()
        corr = m.decode(det)
        dt = (time.perf_counter() - t0) * 1000
        return {"correction": [int(c) for c in np.atleast_1d(corr)],
                "success": True, "latency_ms": dt}
    # model decoders: check demo cache first
    key = f"{req.decoder}:{req.d}:" + "".join(str(int(b)) for b in det)
    if os.path.exists(DEMO_CACHE):
        cache = json.load(open(DEMO_CACHE))
        if key in cache:
            return {"correction": [cache[key]], "success": True,
                    "latency_ms": 0.0, "cached": True}
    model, _ = _load(req.decoder, req.d)
    t0 = time.perf_counter()
    pred = inference.predict_single(model, det.astype(np.float32))
    dt = (time.perf_counter() - t0) * 1000
    return {"correction": [pred], "success": True, "latency_ms": dt}


@app.post("/batch")
def batch(req: BatchReq):
    if req.decoder == "mwpm":
        ev = baseline.evaluate(req.d, req.p, req.shots)
        return {"logical_error_rate": ev["logical_error_rate"],
                "latency_ms": ev["latency_ms"]}
    model, _ = _load(req.decoder, req.d)
    dets, obs = data_gen.generate(req.d, req.p, req.shots, seed=1)
    t0 = time.perf_counter()
    preds = np.array([inference.predict_single(model, r.astype(np.float32))
                      for r in dets])[:, None]
    dt = (time.perf_counter() - t0) / len(dets) * 1000
    ler = metrics.logical_error_rate(preds, obs[:, :1])
    return {"logical_error_rate": ler, "latency_ms": dt}


@app.get("/threshold")
def threshold():
    if not os.path.exists(THRESHOLD):
        raise HTTPException(404, "run precompute.py first")
    return json.load(open(THRESHOLD))
