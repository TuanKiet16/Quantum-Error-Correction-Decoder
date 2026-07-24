#!/usr/bin/env bash
#SBATCH --job-name=qec-train
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=qec-train-%j.log
set -euo pipefail

MODEL="${1:-qcnn_cong}"
DIST="${2:-3}"
shift 2 || true
PS="${*:-0.003 0.005 0.008 0.01 0.015}"

python -m venv .venv-hpc
source .venv-hpc/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

python -m qec_decoder.train --model "$MODEL" --d "$DIST" \
    --ps $PS --shots 20000 --epochs 100 --out checkpoints
echo "checkpoint written to checkpoints/${MODEL}_d${DIST}.pt"
