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

# Unbuffered Python so per-epoch progress streams live into the redirected
# per-model logs instead of sitting in a block buffer.
export PYTHONUNBUFFERED=1

REPO_URL=https://github.com/TuanKiet16/Quantum-Error-Correction-Decoder.git
WORK=${WORK:-/workspace}
DISTANCES=(${DISTANCES:-3 5})
CONG_MAX_D=${CONG_MAX_D:-5}         # skip pure-Cong past this d (its per-patch K blows up)
PS="${PS:-0.003 0.005 0.008 0.01 0.015}"
SHOTS=${SHOTS:-15000}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-4096}               # big batch amortizes default.qubit kernel launches on the L4
GPU_QCHUNK=${GPU_QCHUNK:-6144}     # L4 24GB fits ~6k circuits/forward w/ backprop
EVAL_QCHUNK=${EVAL_QCHUNK:-16384}  # eval has no backprop -> push higher
EVAL_SHOTS=${EVAL_SHOTS:-50000}
CPU_JOB_THREADS=${CPU_JOB_THREADS:-10}   # per parallel CPU job (90 cores / ~6 jobs)

# ---- setup ----
cd "$WORK"
[ -d Quantum-Error-Correction-Decoder ] || git clone "$REPO_URL"
cd Quantum-Error-Correction-Decoder
git pull --ff-only || true
# Torch must match the box's CUDA 12.4 driver. Plain `pip install torch` pulls a
# newer-CUDA wheel that fails with "NVIDIA driver too old" on this L4, so install
# (or reinstall) the cu124 build whenever CUDA isn't actually usable.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "installing torch cu124 (matches the CUDA 12.4 driver)..."
  pip uninstall -y torch >/dev/null 2>&1 || true
  pip install -q torch --index-url https://download.pytorch.org/whl/cu124
fi
pip install -q -e . matplotlib
mkdir -p logs checkpoints results figures

train() {   # model  distance  device  qchunk  qml_device
  local model=$1 d=$2 device=$3 qchunk=$4 qmldev=$5
  local ckpt="checkpoints/${model}_d${d}.pt" log="logs/${model}_d${d}.log"
  if [ -f "$ckpt" ]; then echo "skip (exists): $ckpt"; return 0; fi
  echo ">>> train $model d$d on $device"
  if [ "$device" = cuda ]; then
    # GPU (Cong) stream: show progress live in the terminal AND save the log.
    QEC_QML_DEVICE=$qmldev \
    python -m qec_decoder.train --model "$model" --d "$d" --ps $PS \
        --shots "$SHOTS" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --device "$device" --qchunk "$qchunk" 2>&1 | tee "$log"
  else
    # Parallel CPU jobs: quiet to their own logs so they don't interleave.
    QEC_QML_DEVICE=$qmldev OMP_NUM_THREADS=$CPU_JOB_THREADS \
    python -m qec_decoder.train --model "$model" --d "$d" --ps $PS \
        --shots "$SHOTS" --epochs "$EPOCHS" --batch-size "$BATCH" \
        --device "$device" --qchunk "$qchunk" > "$log" 2>&1
  fi
  echo "<<< done $model d$d"
}

# ---- GPU stream: both QCNNs on default.qubit (it broadcasts the batch; lightning
#      on CPU would Python-loop each sample and be far slower). Hybrid first (1
#      circuit/sample, cheap), then Cong up to CONG_MAX_D (per-patch K explodes). ----
(
  for d in "${DISTANCES[@]}"; do train qcnn_hybrid "$d" cuda "$GPU_QCHUNK" default.qubit; done
  for d in "${DISTANCES[@]}"; do
    if [ "$d" -le "$CONG_MAX_D" ]; then
      train qcnn_cong "$d" cuda "$GPU_QCHUNK" default.qubit
    else
      echo "skip qcnn_cong d$d (> CONG_MAX_D=$CONG_MAX_D; too many patches/sample)"
    fi
  done
) &
gpu_pid=$!

# ---- CPU stream: classical CNN (no quantum sim) in parallel across the cores ----
for d in "${DISTANCES[@]}"; do
  train cnn "$d" cpu 4096 lightning.qubit &
done

wait "$gpu_pid"   # QCNNs finished
wait              # cnn finished
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
