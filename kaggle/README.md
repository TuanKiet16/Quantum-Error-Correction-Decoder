# Kaggle GPU training

Run the two QCNN decoders on a Kaggle GPU instead of the CPU `lightning.qubit`
path. Upload `qec_qcnn_kaggle.ipynb` as a new Kaggle notebook (or "New Notebook
-> File -> Import").

## Settings (right sidebar)
- **Accelerator: GPU** (T4 or P100)
- **Internet: On** (to clone the public repo and pip install)

## What it does
1. Clones the `feat/qec-qcnn-decoder` branch.
2. Installs deps (Torch is preinstalled on the Kaggle GPU image).
3. Sets `QEC_QML_DEVICE=default.qubit` — PennyLane's Torch-native statevector,
   which runs on CUDA and backprops the whole minibatch at once. Chosen over
   `lightning.gpu` because at 10–12 qubits the batch-parallel `default.qubit` is
   faster here and needs no cuQuantum install.
4. Trains with `--device cuda` and bundles `checkpoints/` + `results/` into
   `qec_qcnn_output.zip` (grab it from the notebook's **Output** tab).

## Why this is safe for the rest of the repo
The backend is env-driven (`qec_decoder/models/qdevice.py`); default stays CPU
`lightning.qubit`, so tests and the HPC path are unchanged. GPU is opt-in via
`QEC_QML_DEVICE=default.qubit` + `--device cuda`.

## Tuning
Edit the `MODELS / DISTANCES / PS / SHOTS / EPOCHS` block in cell 6. Kaggle GPU
sessions cap around 9h — start at `d=3`, grow to `d=5` once timing is known.
