# QEC Decoder Demo — QCNN vs MWPM

Compares two QCNN decoders (Cong pure-quantum, Liu hybrid QCCNN) and a classical
CNN against an MWPM baseline for the rotated surface code.

## Install
```bash
pip install -e ".[dev]"
```

## End-to-end run order
```bash
# Part 1 — data
python -m qec_decoder.data_gen --d 3 --p 0.005 --shots 200000 --out data/

# Part 2 — MWPM baseline (number to beat)
python -m qec_decoder.baseline --d 3 --p 0.005 --shots 20000

# Part 3 — train (HPC; see hpc/HPC_SETUP.md). Locally you can smoke-train the CNN:
python -m qec_decoder.train --model cnn --d 3 --ps 0.003 0.005 0.01 --shots 5000 --epochs 20

# Part 3b — benchmark on the demo machine (CPU) + optional cache
python -m qec_decoder.benchmark_inference --ckpt checkpoints/cnn_d3.pt --d 3

# Part 5 — threshold precompute
python -m qec_decoder.precompute --ds 3 5 7 9 --out results/threshold.json

# Part 4 — API
uvicorn qec_decoder.api.server:app --reload
```

## Metrics
Reported via `qec_decoder/metrics.py`, following the Google below-threshold
paper (logical error per cycle ε_d, suppression factor Λ = ε_d/ε_{d+2}, binomial
uncertainty) and Q_Design fidelity-vs-error curves.

## Tests
```bash
pytest              # fast unit + forward-pass smoke
pytest --runslow    # includes long/training tests (not run by default)
```

## HPC training + deployment
See `hpc/HPC_SETUP.md`. Copy only `checkpoints/` + `results/` to the demo machine.

## Part 6 (optional)
`qec_decoder/google_data.py` loads the Google d3/d5/d7 dataset once the 5.7 GB
zip is downloaded to `data/`. It stops with instructions if the file is absent.
