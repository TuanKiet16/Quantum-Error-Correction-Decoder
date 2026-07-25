# HPC Training + Demo Deployment

## 1. Train on HPC
```bash
# copy the repo (without data/) to HPC, then:
sbatch hpc/train_hpc.sh qcnn_cong 3 0.003 0.005 0.008 0.01 0.015
sbatch hpc/train_hpc.sh qcnn_hybrid 3 0.003 0.005 0.008 0.01 0.015
sbatch hpc/train_hpc.sh cnn 3 0.003 0.005 0.008 0.01 0.015
```
Repeat with distance 5 / 7 / 9 as GPU budget allows. Training is NOT run on the
demo machine.

## 2. Copy to the demo machine
Copy ONLY these back (not raw `data/`):
- `checkpoints/` — trained models
- `results/` — `threshold.json`, `demo_cache.json`

## 3. Self-check the demo machine before going on stage
```bash
python -m qec_decoder.benchmark_inference --ckpt checkpoints/qcnn_cong_d3.pt --d 3
```
If it prints "QCNN inference is slow, consider caching results for the demo",
build the cache:
```bash
python -m qec_decoder.precompute_demo_cases \
    --ckpt qcnn_cong=checkpoints/qcnn_cong_d3.pt cnn=checkpoints/cnn_d3.pt --ds 3
```

## Optional: Google 105Q dataset (Part 6)

Real Google hardware detection records (d3/d5/d7). ~5.7 GB. Only needed to
validate decoders against experimental data; the rest of the pipeline runs
without it. Fetch on HPC scratch, not locally:

```bash
bash hpc/fetch_google_data.sh data
# resumable; verifies md5 21fa6ad35b395d838ebcdbc92e364a12
```

Lands at `data/google_105Q_surface_code_d3_d5_d7.zip`, where
`qec_decoder.google_data.GOOGLE_ZIP` expects it. Parsing (`load()`) is still a
stub — read the README inside the zip first, then map its .dem + detection
records into `(detection_events[N,D], observable_flips[N,L])`.
