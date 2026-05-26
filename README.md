# FedULearn-IoT
This repository contains the code for the development of a federated unified model for device identification and intrusion detection using packet flow data.

# Multimode Federated Learning for IoT Intrusion Detection

A federated learning framework for multi-output IoT network intrusion detection, supporting simultaneous classification of **traffic type** (attack category) and **device type** across distributed clients. The pipeline benchmarks multiple deep learning and traditional ML models with optional differential privacy via Opacus.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [How to Run](#how-to-run)
- [Models](#models)
- [Outputs](#outputs)
- [Known Issues & Tips](#known-issues--tips)

---

## Overview

This notebook implements a federated learning pipeline in which multiple clients each train a local model on a partition of IoT network traffic data. A central server aggregates client model parameters using **Federated Averaging (FedAvg)** across configurable communication rounds. The system simultaneously predicts:

- **Traffic Type** — attack category (e.g. SYN Flood, Port Scan, Vul Scan, SlowLoris)
- **Device Label** — IoT device type (e.g. Nestcam, SamsungTV, Smartplug)

---

## Features

- Multi-output classification (traffic type + device type simultaneously)
- Federated learning with FedAvg aggregation
- Differential privacy via [Opacus](https://opacus.ai/) (configurable)
- Seven model architectures across two frameworks: PyTorch and TensorFlow/Keras
- Traditional ML baselines: XGBoost, Random Forest, Decision Tree
- Balanced class weighting, optional SMOTE oversampling
- Per-round evaluation with comprehensive metrics
- Automatic saving of models, confusion matrices, classification reports, and privacy budget logs

---

## Requirements

### Python Version
Python 3.9+

### Conda Environment (Recommended)

```bash
conda create -n multimode python=3.9
conda activate multimode
```

### Package Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow==2.19.0
pip install opacus
pip install scikit-learn xgboost imbalanced-learn
pip install pandas numpy matplotlib seaborn tqdm tabulate
pip install scienceplots psutil joblib
```

> **GPU note:** A CUDA-capable GPU is strongly recommended. The notebook auto-detects `cuda` and falls back to `cpu`. Training on CPU with this dataset will be very slow.

---

## Dataset

The notebook is designed for the **NIM LAB IoT Dataset 2025**, a multi-label network traffic dataset containing IoT device traffic annotated with attack categories.

Expected CSV columns include network-level features (TCP flags, IP fields, payload statistics, protocol ratios, etc.) plus two label columns:

- `Label` — IoT device type
- `Traffic Type` — attack/traffic category

### Setting the Dataset Path

In the final cell of the notebook, update the `csv_file` path to point to your local copy:

```python
csv_file = r"path/to/your/NIMLABIoT_processed.csv"
```

The notebook expects the dataset to already be in its processed/combined CSV form. It handles deduplication, rare-class removal, feature selection, and scaling internally.

---

## Project Structure

```
Multimode.ipynb          # Main notebook — all code lives here
classification_reports/  # Per-model classification reports (auto-created)
confusion_matrices/      # Confusion matrix CSVs and plots (auto-created)
models/                  # Saved model files (.pth for PyTorch, .h5 for TF) (auto-created)
metrics/                 # Per-round and ensemble metrics CSVs (auto-created)
metrics_final.csv        # Consolidated metrics across all models (auto-created)
privacy_budget.csv       # Per-client epsilon log (auto-created if DP is enabled)
```

---

## Configuration

All hyperparameters are controlled through the `Parameters` dataclass at the top of the notebook (Cell 1):

```python
@dataclass
class Parameters:
    lr: float = 0.01                    # Learning rate (SGD)
    device: torch.device = ...          # Auto-detects CUDA or CPU
    sample_rate: float = 0.2            # Opacus Poisson sampling rate
    noise_multiplier: float = 1.5       # DP noise multiplier
    max_grad_norm: float = 0.5          # Per-sample gradient clipping norm (Opacus)
    delta: float = 1e-4                 # Target delta for (ε, δ)-DP
    epochs: int = 100                   # Total epochs (split across rounds)
    num_clients: int = 5                # Number of federated clients
    batch_size: int = 128               # Mini-batch size
    communication_rounds: int = 10      # Number of FL aggregation rounds
    conv_filters: List[int] = [32, 64]  # CNN filter sizes
    kernel_size: int = 3                # CNN kernel size
    dropout_rate: float = 0.3           # Dropout probability
    use_smote: bool = False             # Enable SMOTE oversampling
    use_dp: bool = True                 # Enable differential privacy
    model_type: str = 'dl'             # 'dl', 'ml', or 'both'
    multi_output: bool = True           # True = joint task; False = separate models per task
```

Modify `args = Parameters(conv_filters=[32, 64])` to customise a run. For example, to run only ML models without differential privacy:

```python
args = Parameters(conv_filters=[32, 64], model_type='ml', use_dp=False)
```

---

## How to Run

1. **Install dependencies** (see [Requirements](#requirements)).

2. **Set your dataset path** in the final cell:
   ```python
   csv_file = r"path/to/NIMLABIoT_processed.csv"
   ```

3. **Open and run the notebook sequentially** — all cells must be executed in order from top to bottom:
   - Cell 1: Imports, configuration, data sampling utilities
   - Cell 2+: Model class definitions (CNN, LSTM, BiLSTM, TF1DCNN, ML models, clients, servers)
   - Final cell: Data loading, client partitioning, and full training loop

4. **Re-running after an interrupted run**: Always **restart the kernel** before re-running. Failing to do so can cause GPU memory exhaustion and cuDNN errors. As a precaution, add the following at the top of the training cell:
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

---

## Models

### Deep Learning (PyTorch) — `model_type='dl'`

| Model | Architecture | Notes |
|---|---|---|
| `cnn` | 1D CNN with two conv blocks + dual FC heads | GroupNorm; Opacus DP support |
| `lstm` | 2-layer LSTM + dual FC heads | |
| `bilstm` | 2-layer Bidirectional LSTM + dual FC heads | |

### Deep Learning (TensorFlow/Keras) — `type='tf'`

| Model | Architecture | Notes |
|---|---|---|
| `tf1cnn` | 1D CNN with two conv blocks + dual output heads | Trained locally per client; no Opacus DP |

### Machine Learning (scikit-learn) — `model_type='ml'`

| Model | Notes |
|---|---|
| `xgboost` | XGBoost with multi-output wrapper |
| `rf` | Random Forest with multi-output wrapper |
| `dt` | Decision Tree with multi-output wrapper |

In `multi_output=True` mode each model predicts both traffic type and device type simultaneously. In `multi_output=False` mode, separate models are trained per task (e.g. `cnn_traffic`, `cnn_device`).

---

## Outputs

After training completes, the following files are written to disk:

| Output | Description |
|---|---|
| `metrics_final.csv` | All per-round and final metrics for every model |
| `privacy_budget.csv` | Per-client, per-round privacy epsilon values (DP only) |
| `models/<name>_final.pth` | Saved PyTorch model weights |
| `models/<name>_final.h5` | Saved TensorFlow model |
| `models/<name>_<client>_final.joblib` | Saved ML model per client |
| `confusion_matrices/` | Confusion matrix PNGs and CSVs |
| `classification_reports/` | Per-class precision/recall/F1 reports |

### Metrics Logged Per Round

Traffic and device classification metrics are reported separately, including: Accuracy, Precision, Recall, F1, AUC-ROC, Cohen's Kappa, Matthews Correlation Coefficient, Balanced Accuracy, Hamming Loss, TNR, TPR, FPR, FNR, and Specificity. Resource metrics (memory MB, CPU%, energy proxy) are also logged per training round.

---

## Known Issues & Tips

**`RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`**

This is the most common error when re-running without a kernel restart. Root causes in order of likelihood:

1. GPU memory not freed from a previous run — always restart the kernel before re-running.
2. `PrivacyEngine` wrapping the model a second time on top of an already-wrapped instance — add an `hasattr` guard before calling `make_private`.
3. Redundant `clip_grad_norm_` call conflicting with Opacus's internal per-sample clipping — disable it when `use_dp=True`.
4. `GroupNorm` instability with variable Opacus batch sizes — consider switching to `LayerNorm` for more robust DP training.

**Secure RNG warning from Opacus**

The warning `"Secure RNG turned off"` is expected during experimentation. Enable `secure_mode=True` in `PrivacyEngine` only for production/final runs, as it significantly slows training.

**TF1DCNN final aggregation error**

A known issue causes `set_weights` to fail during the final TF1DCNN aggregation step when the weight list lengths mismatch. Per-round results are still saved correctly; only the final aggregation is affected.

**Large dataset memory usage**

The dataset (~566K samples after sampling) is memory-intensive. If you encounter OOM errors, reduce the sampling fraction in `sample_50_percent_by_label()` or lower `batch_size` in `Parameters`.

---

## Citation / Acknowledgements

Dataset: NIM LAB IoT Dataset 2025  
Privacy framework: [Opacus](https://github.com/pytorch/opacus)  
Federated averaging: McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data*, AISTATS 2017
