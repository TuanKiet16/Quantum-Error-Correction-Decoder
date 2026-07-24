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
