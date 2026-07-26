#!/usr/bin/env bash
# One-shot train + evaluate + plot on a Quandela Notebook box
# (NVIDIA L4 24GB, ~90-core AMD EPYC, ~330GB RAM, persistent /workspace).
#
# Strategy: QCNN-Cong runs one circuit PER PATCH and dominates cost, so it takes
# the GPU (sequential, large --qchunk). QCNN-Hybrid (1 circuit/sample) and the
# classical CNN are cheap, so they run on the many CPU cores in parallel. Neither
# the GPU nor the CPUs sit idle. Re-running skips finished checkpoints.
#
# Tune via env, e.g.:  DISTANCES="3 5" SHOTS=50000 bash run_quandela.sh
set -euo pipefail

REPO_URL=https://github.com/TuanKiet16/Quantum-Error-Correction-Decoder.git
WORK=${WORK:-/workspace}
DISTANCES=(${DISTANCES:-3 5 7})
PS="${PS:-0.003 0.005 0.008 0.01 0.015}"
SHOTS=${SHOTS:-20000}
EPOCHS=${EPOCHS:-60}
BATCH=${BATCH:-1024}
GPU_QCHUNK=${GPU_QCHUNK:-6144}     # L4 24GB fits ~6k circuits/forward w/ backprop
EVAL_QCHUNK=${EVAL_QCHUNK:-16384}  # eval has no backprop -> push higher
EVAL_SHOTS=${EVAL_SHOTS:-50000}
CPU_JOB_THREADS=${CPU_JOB_THREADS:-10}   # per parallel CPU job (90 cores / ~6 jobs)

# ---- setup ----
cd "$WORK"
[ -d Quantum-Error-Correction-Decoder ] || git clone "$REPO_URL"
cd Quantum-Error-Correction-Decoder
git pull --ff-only || true
python -c "import torch" 2>/dev/null || pip install -q torch
pip install -q -e . matplotlib
mkdir -p logs checkpoints results figures

train() {   # model  distance  device  qchunk  qml_device
  local model=$1 d=$2 device=$3 qchunk=$4 qmldev=$5
  local ckpt="checkpoints/${model}_d${d}.pt"
  if [ -f "$ckpt" ]; then echo "skip (exists): $ckpt"; return 0; fi
  echo ">>> train $model d$d on $device"
  QEC_QML_DEVICE=$qmldev OMP_NUM_THREADS=$CPU_JOB_THREADS \
  python -m qec_decoder.train --model "$model" --d "$d" --ps $PS \
      --shots "$SHOTS" --epochs "$EPOCHS" --batch-size "$BATCH" \
      --device "$device" --qchunk "$qchunk" \
      > "logs/${model}_d${d}.log" 2>&1
  echo "<<< done $model d$d"
}

# ---- GPU stream: QCNN-Cong, one distance at a time (full GPU each) ----
( for d in "${DISTANCES[@]}"; do train qcnn_cong "$d" cuda "$GPU_QCHUNK" default.qubit; done ) &
gpu_pid=$!

# ---- CPU stream: QCNN-Hybrid + CNN, all distances in parallel ----
for d in "${DISTANCES[@]}"; do
  train qcnn_hybrid "$d" cpu 4096 lightning.qubit &
  train cnn         "$d" cpu 4096 lightning.qubit &
done

wait "$gpu_pid"   # cong finished
wait              # hybrid + cnn finished
echo "=== all training done; checkpoints: ==="; ls -la checkpoints

# ---- evaluate all decoders vs MWPM on a shared fresh test set (GPU) ----
QEC_QML_DEVICE=default.qubit python -m qec_decoder.evaluate \
    --ckpt-dir checkpoints --ds "${DISTANCES[@]}" --ps $PS \
    --shots "$EVAL_SHOTS" --device cuda --qchunk "$EVAL_QCHUNK" \
    --out results/comparison.json

python -m qec_decoder.plot_comparison --out figures/decoder_comparison.pdf
echo "=== DONE ==="
echo "results/comparison.json  +  figures/decoder_comparison.pdf"
python -c "import json;d=json.load(open('results/comparison.json'));print('Lambda:',d['lambda'])"
