# FedULearnIoT

**FedULearnIoT** is a federated learning framework for **simultaneous IoT intrusion detection and device identification**. A single model predicts traffic type (attack category) and device type from the same network-traffic features, trained across distributed clients under configurable non-IID partitioning, with local differential privacy.

Implements four federated aggregation strategies (FedAvg, FedProx, SCAFFOLD, FedNova) and three published device-identification schemes (HFedDI, HAFedL, ConFedDI) as directly comparable baselines, plus classical ML references.

Runs identically from the command line and from a notebook.

> The project is **FedULearn**; the Python package and console command are `fedulearn` (lowercase, per PEP 8).

---

## Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Quick start](#quick-start)
- [Command-line reference](#command-line-reference)
- [Notebook usage](#notebook-usage)
- [Partitioning scenarios](#partitioning-scenarios)
- [Models and strategies](#models-and-strategies)
- [Differential privacy](#differential-privacy)
- [Full configuration reference](#full-configuration-reference)
- [Outputs](#outputs)
- [Project layout](#project-layout)
- [Tests](#tests)
- [Reproducibility and fidelity notes](#reproducibility-and-fidelity-notes)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)
- [License](#license)

---

## Features

- Dual-head classification: traffic type and device type from one shared encoder
- Four aggregation strategies, composable with any PyTorch backbone
- Three device-identification baselines implemented from their source papers
- Six client-partitioning schemes, including Dirichlet non-IID
- Per-client differential privacy via Opacus, with Rényi DP accounting
- Incremental result writing
- Single CLI and notebook entry point over identical code

---

## Requirements

| | |
|---|---|
| Python | 3.9 or newer |
| GPU | CUDA-capable strongly recommended; CPU works but is slow |
| RAM | 16 GB+ for the full dataset |
| OS | Linux, macOS, Windows |

TensorFlow is **optional** — needed only for the `fedavg_tf1cnn` baseline.

---

## Installation

PyTorch is not installable from PyPI in its CUDA build, so install it first and separately.

### 1. Create an environment

```bash
conda create -n fedulearn python=3.10 -y
conda activate fedulearn
```

### 2. Install PyTorch

Pick the wheel matching your CUDA driver (check with `nvidia-smi`):

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the remaining dependencies

```bash
pip install -r requirements.txt
```

### 4. Install the package

```bash
pip install -e .
```

The `-e` (editable) flag means source edits take effect without reinstalling.

### 5. Verify

```bash
fedulearn list-models
```

This prints the available baselines, partitioning schemes and their citations. If the command is not found, use `python -m fedulearn list-models` — equivalent, and independent of whether the console script landed on your `PATH`.

### 6. Optional: TensorFlow baseline

```bash
pip install -r requirements-tf.txt
```

### Installing for a Jupyter kernel

The most common installation problem is `pip` and the notebook kernel pointing at different environments. Install from inside a notebook cell so it targets the kernel's own interpreter:

```python
%pip install -e .
```

Confirm which environment a kernel is using:

```python
import sys; print(sys.executable)
```

---

## Dataset

Designed for the **NIM LAB IoT Dataset 2025**, but any CSV with the same schema works.

### Expected format

A single preprocessed CSV containing packet-header and flow-statistic columns plus two label columns:

| Column | Meaning |
|---|---|
| `Label` | IoT device type (e.g. `Nestcam`, `SamsungTV`, `Smartplug`) |
| `Traffic Type` | attack or traffic category (e.g. `SYN Flood`, `Port Scan`, `SlowLoris`, `Vul Scan`) |

Feature columns include TCP flags, IP fields, payload statistics, protocol ratios and inter-arrival times. The exact whitelist is `cols_to_use` in `fedulearn/data/preprocess.py`; columns outside it are ignored, so extra columns are harmless.

### What the pipeline does automatically

NaN filling, deduplication, rare-class removal, label encoding, stratified train/test splitting, feature scaling, and partitioning across clients. No manual preprocessing is required beyond producing the combined CSV.

### Pointing at your data

```bash
fedulearn run --data /path/to/NIMLABIoT_processed.csv
```

Windows paths work as-is:

```bat
fedulearn run --data C:\data\NIMLABIoT_processed.csv
```

---

## Quick start

### Always smoke-test first

```bash
fedulearn run --data data/NIMLABIoT_processed.csv --smoke-test
```

This runs 1 round × 2 epochs × 2 clients with DP off — minutes rather than hours — and confirms every baseline completes end to end on your hardware. A full sweep allocates `num_clients` models per baseline, so verifying at small scale first avoids discovering a GPU memory problem an hour in.

### Then the real run

```bash
fedulearn run --data data/NIMLABIoT_processed.csv \
    --config configs/default.yaml \
    --output-dir outputs/run01
```

### Common variations

```bash
# Two specific baselines, no differential privacy
fedulearn run --data d.csv --only fedprox_cnn scaffold_cnn --no-dp

# Dirichlet non-IID partitioning
fedulearn run --data d.csv --partition dirichlet --dirichlet-alpha 0.3

# Every partitioning scheme in sequence
fedulearn run --data d.csv \
    --partitions iid device_skew attack_skew quantity_skew dirichlet

# Include the original DL models and classical ML references
fedulearn run --data d.csv --include-original --include-ml

# Recurrent backbones
fedulearn run --data d.csv --only fedprox_lstm scaffold_bilstm

# Separate per-task models instead of the joint objective
fedulearn run --data d.csv --single-task

# Force CPU
fedulearn run --data d.csv --device cpu
```

---

## Command-line reference

```
fedulearn list-models      # every runnable model, scheme and citation
fedulearn run --help       # all flags
fedulearn run [options]
```

`python -m fedulearn ...` is equivalent throughout.

**Resolution order:** dataclass defaults → `--config` YAML → explicit flags. A flag always wins.

### Core

| Flag | Default | Description |
|---|---|---|
| `--data CSV` | *required* | Path to the preprocessed dataset |
| `--config YAML` | — | YAML file of `Parameters` overrides |
| `--output-dir DIR` | `outputs` | Where models, metrics and figures are written |

### Federation

| Flag | Default | Description |
|---|---|---|
| `--num-clients N` | `5` | Number of federated clients |
| `--rounds N` | `10` | Communication (aggregation) rounds |
| `--epochs N` | `100` | Total epochs, divided across rounds |
| `--batch-size N` | `128` | Mini-batch size |
| `--lr F` | `0.01` | Learning rate |
| `--sample-rate F` | `0.2` | Test-split fraction |

Local epochs per round are `max(1, epochs // rounds)`.

### Differential privacy

| Flag | Default | Description |
|---|---|---|
| `--dp` / `--no-dp` | on | Enable or disable Opacus DP |
| `--noise-multiplier F` | `1.5` | DP noise scale |
| `--max-grad-norm F` | `0.5` | Per-sample gradient clipping norm |
| `--delta F` | `1e-4` | Target δ for (ε, δ)-DP |

### Architecture

| Flag | Default | Description |
|---|---|---|
| `--conv-filters N [N ...]` | `32 64` | CNN channel widths |
| `--kernel-size N` | `3` | Convolution kernel size |
| `--dropout-rate F` | `0.3` | Dropout probability |

### Partitioning

| Flag | Default | Description |
|---|---|---|
| `--partition SCHEME` | `iid` | One of `iid`, `device_skew`, `attack_skew`, `label_skew`, `quantity_skew`, `dirichlet` |
| `--partitions S [S ...]` | — | Run several schemes in sequence |
| `--dirichlet-alpha F` | `0.5` | Dirichlet concentration; smaller is more skewed |
| `--dirichlet-target T` | `device` | `device`, `traffic` or `both` |
| `--shards-per-client N` | `2` | Shard-skew severity; 1 is most extreme |
| `--partition-seed N` | `42` | Partition RNG seed, separate from the training seed |

### Experiment selection

| Flag | Default | Description |
|---|---|---|
| `--only NAME [NAME ...]` | — | Restrict to these baselines; implies baselines are enabled |
| `--include-original` | off | Also run `cnn`, `lstm`, `bilstm`, `tf1cnn` |
| `--include-ml` | off | Also run `xgboost`, `rf`, `dt` |
| `--no-baselines` | — | Skip the federated baselines |
| `--single-task` | off | Train separate traffic and device models |

### Misc

| Flag | Default | Description |
|---|---|---|
| `--seed N` | `42` | RNG seed for Python, NumPy and PyTorch |
| `--device {cuda,cpu}` | auto | Force a device |
| `--smoke-test` | off | 1 round, 2 epochs, 2 clients, DP off |

---

## Notebook usage

`notebooks/fedulearn_demo.ipynb` is a working walkthrough. The core pattern:

```python
from fedulearn import Parameters, set_config
from fedulearn.runner import prepare_data, run_experiment

cfg = Parameters(num_clients=5, communication_rounds=10, epochs=100,
                 use_dp=True, partition="dirichlet", dirichlet_alpha=0.3,
                 output_dir="outputs/notebook")
set_config(cfg)

# Load and partition once, then reuse across experiments.
data = prepare_data("data/NIMLABIoT_processed.csv", cfg)

results = run_experiment("data/NIMLABIoT_processed.csv", data=data)
final = results[results["round"] == "final"]
final[["model", "strategy", "partition", "traffic_f1", "device_f1"]]
```

`run_experiment` returns a `pandas.DataFrame` with one row per model per round, plus a `final` row per model.

### Understanding the flow

**The partition is fixed when the data is loaded, not when a model is trained.** `prepare_data` splits the training set and returns a `data` object that already contains that split; `run_experiment` trains on whatever split it is handed. This gives two mutually exclusive usage paths:

| | Path A — one scenario | Path B — sweep |
|---|---|---|
| Use when | iterating, or reporting a single setting | producing the IID-vs-non-IID table |
| You call | `prepare_data`, then `run_experiment` | `run_scenarios` only |
| Partition set by | `cfg.partition` before `prepare_data` | `partitions=[...]` argument |
| Data loaded | once, reused | once per scheme, internally |

Changing `cfg.partition` after `prepare_data` and reusing the same `data` trains on the **old** split — call `prepare_data` again to repartition. The package prints a warning when it detects this mismatch.

Path B:

```python
from fedulearn.runner import run_scenarios

df = run_scenarios("data/NIMLABIoT_processed.csv", cfg,
                   partitions=["iid", "device_skew", "dirichlet"])
```

Do not pass `data=` to `run_scenarios` — that would pin every scheme to a single split.

### Running a single model

```python
from fedulearn.registry import build_baseline_configs
from fedulearn.runner import train_one_model, prepare_output_dirs

paths = prepare_output_dirs(cfg)
mc = build_baseline_configs(task="both", only=["scaffold_cnn"])[0]
metric_rows, privacy_rows = train_one_model(mc, data=data, paths=paths, cfg=cfg)
```

---

## Partitioning scenarios

| Scheme | What varies | Signature |
|---|---|---|
| `iid` | nothing (default) | Gini ≈ 0, TV ≈ 0 |
| `device_skew` | which **device** classes each client sees | high TV device |
| `attack_skew` | which **attack** classes each client sees | high TV traffic |
| `label_skew` | the joint (device, attack) label | both heads move |
| `quantity_skew` | client **sizes** only; class balance stays global | high Gini, TV ≈ 0 |
| `dirichlet` | per-class Dirichlet allocation | tunable via `--dirichlet-alpha` |

Terminology follows Li et al. (arXiv:2102.02079): *label distribution skew* is different clients observing different class sets; *quantity skew* is different clients holding different amounts. `quantity_skew` isolates the second from the first, making the two separable in an ablation.

`dirichlet` implements the standard non-IID benchmark of Hsu et al. (arXiv:1909.06335): for each class, draw client proportions from `Dir(α)`. Small α concentrates each class on few clients; α ≥ 100 is effectively IID.

### Quantifying the skew

Multi-scenario runs write `partition_stats.csv` with two measures per scheme:

- `quantity_gini` — Gini coefficient of client sizes; 0 is equal, → 1 is lopsided
- `mean_tv_traffic` / `mean_tv_device` — mean total-variation distance between each client's label distribution and the global one, per head; 0 is IID, 1 is fully disjoint
---

## Models and strategies

### Aggregation strategies

The four strategies act on parameter tensors, so they compose with any PyTorch backbone. The registry is the cross product, giving names like `fedprox_cnn`:

| Strategy | Method | Reference |
|---|---|---|
| `fedavg` | Weighted parameter averaging | [McMahan et al](https://arxiv.org/abs/1602.05629)  |
| `fedprox` | Proximal term `μ/2·‖w − wᵗ‖²` anchoring local updates | [Li et al](https://arxiv.org/abs/1812.06127) |
| `scaffold` | Control variates correcting client drift | [Karimireddy et al](https://arxiv.org/abs/1910.06378) |
| `fednova` | Normalised averaging over unequal local step counts | [Wang et al](https://arxiv.org/abs/2007.07481) |

Available backbones: `_cnn`, `_lstm`, `_bilstm`. All twelve combinations are valid.

### Device-identification baselines

| Name | Method | Reference |
|---|---|---|
| `hfeddi` | Horizontal FL device identification; 128/64/32 DNN, GroupNorm, weighted cross-entropy, unweighted aggregation | Sumitra & Shenoy,  [doi:10.1016/j.jnca.2023.103616](https://doi.org/10.1016/j.jnca.2023.103616) |
| `hafedl` | Hessian-aware adaptive perturbation against gradient-leakage attacks; per-client σ via modified RDP | Sumitra, Sharma & Shenoy,  [doi:10.1109/ACCESS.2024.3454074](https://doi.org/10.1109/ACCESS.2024.3454074) |
| `confeddi` | Lightweight 1D AlexNet on packet headers, weighted cross-entropy, momentum SGD | Chen, Xiong, Wang & Chen, [doi:10.1002/cpe.70237](https://doi.org/10.1002/cpe.70237) |

### TensorFlow and classical ML

- `fedavg_tf1cnn` — FedAvg only, no Opacus path, always trains without DP. Requires `requirements-tf.txt`.
- `--include-ml` adds `xgboost`, `rf`, `dt`.

### Default queue

Seven runs: the four aggregation strategies on the CNN backbone, plus the three device-identification methods. LSTM/BiLSTM variants and the TensorFlow baseline are opt-in by name, since the full grid is sixteen.

`BASELINE_REFERENCES` in `fedulearn/registry.py` holds the citation for every baseline, and a test asserts none can exist without one.

---

## Differential privacy

Enabled by default (`use_dp: true`), implemented with Opacus and Rényi DP accounting. Per-client, per-round ε is written to `privacy_budget.csv`.

---

## Full configuration reference

Every field of `Parameters` in `fedulearn/config.py`. `configs/default.yaml` mirrors these with comments; `configs/smoke_test.yaml` is a fast variant.

### Optimisation

| Field | Default | Description |
|---|---|---|
| `lr` | `0.01` | Learning rate |
| `epochs` | `100` | Total epochs, divided across rounds |
| `batch_size` | `128` | Mini-batch size |

### Federation

| Field | Default | Description |
|---|---|---|
| `num_clients` | `5-10` | Federated clients |
| `communication_rounds` | `10` | Aggregation rounds |
| `sample_rate` | `0.2` | Test-split fraction |

### Differential privacy

| Field | Default | Description |
|---|---|---|
| `use_dp` | `True` | Enable Opacus DP |
| `noise_multiplier` | `1.5` | Noise scale |
| `max_grad_norm` | `0.5` | Per-sample clipping norm |
| `delta` | `1e-4` | Target δ |

### Architecture

| Field | Default | Description |
|---|---|---|
| `conv_filters` | `[32, 64]` | CNN channel widths |
| `kernel_size` | `3` | Convolution kernel |
| `dropout_rate` | `0.3` | Dropout probability |

### Partitioning

| Field | Default | Description |
|---|---|---|
| `partition` | `iid` | Active scheme |
| `partitions` | `None` | List of schemes to run in sequence |
| `dirichlet_alpha` | `0.5` | Dirichlet concentration |
| `dirichlet_target` | `device` | Head the Dirichlet scheme skews |
| `shards_per_client` | `2` | Shard-skew severity |
| `partition_seed` | `42` | Partition RNG seed |

### Experiment shape

| Field | Default | Description |
|---|---|---|
| `model_type` | `dl` | `dl`, `ml` or `both` |
| `multi_output` | `True` | Joint objective vs. separate per-task models |
| `use_smote` | `False` | SMOTE oversampling |

### FedProx / SCAFFOLD

| Field | Default | Description |
|---|---|---|
| `fedprox_mu` | `0.01` | Proximal coefficient μ |
| `scaffold_lr_g` | `1.0` | Server learning rate η_g |

### HFedDI

Values taken from the paper (§3.3, Table 3).

| Field | Default | Description |
|---|---|---|
| `hfeddi_hidden` | `[128, 64, 32]` | Hidden layer widths |
| `hfeddi_group_size` | `8` | GroupNorm groups per hidden layer |
| `hfeddi_momentum` | `0.9` | SGD momentum |
| `hfeddi_dropout` | `0.0` | Paper specifies none |

### HAFedL

| Field | Default | Description |
|---|---|---|
| `hafedl_hessian_threshold` | `0.4` | Weights above are protected from noise |
| `hafedl_clip_C` | `0.4` | Clipping threshold C |
| `hafedl_epsilon` | `5.0` | Target ε |
| `hafedl_delta` | `1e-5` | Target δ — paper states 0.7; see notes |
| `hafedl_sampling_rate` | `0.04` | Sampling rate S for device identification |
| `hafedl_curvature` | `fisher` | `fisher` or `hutchinson` |
| `hafedl_threshold_mode` | `value` | `value` (paper) or `quantile`; see notes |
| `hafedl_ema` | `0.9` | Curvature EMA factor |
| `hafedl_optimizer` | `adam` | Optimizer |

### ConFedDI

| Field | Default | Description |
|---|---|---|
| `confeddi_momentum` | `0.9` | SGD momentum |

### Runtime

| Field | Default | Description |
|---|---|---|
| `device` | auto | `cuda` if available, else `cpu` |
| `seed` | `42` | RNG seed |
| `output_dir` | `outputs` | Run output root |

### YAML example

```yaml
num_clients: 10
communication_rounds: 20
epochs: 200
batch_size: 128
lr: 0.005

use_dp: true
noise_multiplier: 1.5
delta: 1.0e-5

partition: dirichlet
dirichlet_alpha: 0.3
dirichlet_target: device

seed: 42
output_dir: outputs/run01
```

Unknown keys are rejected with an explicit error, so typos fail loudly rather than being silently ignored.

---

## Outputs

Everything is written under `output_dir`, so separate runs never overwrite one another:

```
outputs/run01/
├── config.json                  # exact configuration used
├── metrics_final.csv            # every model, every round, plus a final row
├── privacy_budget.csv           # per-client, per-round epsilon (DP only)
├── partition_stats.csv          # skew statistics (multi-scenario runs)
├── metrics_all_scenarios.csv    # combined table (multi-scenario runs)
├── models/                      # .pth / .h5 / .joblib
├── metrics/                     # per-model ensemble metrics
├── classification_reports/      # per-class precision, recall, F1
└── confusion_matrices/          # PNG figures and CSV matrices
```

Multi-scenario runs additionally nest one subdirectory per scheme.

### Metrics

Reported separately for the traffic and device heads: accuracy, precision, recall, F1, AUC-ROC, Cohen's κ, Matthews correlation, balanced accuracy, Hamming loss, TPR, TNR, FPR, FNR, specificity. Per-round wall-clock, memory, CPU and an energy proxy are also recorded.

`metrics_final.csv` carries `model`, `strategy`, `partition`, `task` and `round` columns, so comparing baselines is a `groupby`.

---

## Project layout

```
fedulearn/
├── config.py               Parameters, active-config accessor, seeding, GPU cleanup
├── registry.py             which models exist; client/server construction
├── runner.py               experiment orchestration and scenario sweeps
├── cli.py                  argparse entry point
├── data/
│   ├── sampling.py         class-balanced sub-sampling
│   ├── preprocess.py       CSV loading, cleaning, encoding, scaling
│   ├── partitioning.py     six client-partitioning schemes
│   └── loader.py           client partitioning, DataLoader construction
├── models/
│   ├── torch_models.py     MultiOutputCNN, LSTMModel, BiLSTM
│   ├── baselines.py        HFedDIModel, ConFedDIModel, HFedDI weighted loss
│   ├── tf_models.py        TF1DCNN (optional TensorFlow)
│   └── ml_models.py        MLModel, MLEnsembleModel
├── federated/
│   ├── clients.py          DLClient, TF1Client, MLClient
│   ├── servers.py          DLServer, TF1Server, MLServer
│   ├── strategies.py       FedDLClient, FedDLServer
│   ├── privacy.py          HAFedL per-client RDP budget
│   └── dp_compat.py        Opacus compatibility for recurrent models
├── evaluation/
│   ├── metrics.py          compute_metrics
│   └── evaluate.py         evaluate_dl / evaluate_tf1 / evaluate_ml
└── persistence/
    ├── io.py               model, report and figure saving
    └── incremental.py      crash-safe incremental CSV writing
```

`FedDLClient` inherits `train()` from `DLClient` and injects strategy behaviour by wrapping `optimizer.step()`. Loss weighting, DP accounting and resource metrics are therefore identical across strategies, which is what makes the comparison fair.

---

## Tests

```bash
pip install pytest
pytest -q                    # everything
pytest -q -m "not slow"      # unit tests only
```

---

## Troubleshooting

### `ModuleNotFoundError: No module named 'fedulearn'`

The package is not installed in the environment being used. Most often `pip` and the Jupyter kernel point at different environments.

```python
import sys; print(sys.executable)   # which interpreter is running
```

Then either install into the kernel's environment from a cell:

```python
%pip install -e ..     # from notebooks/; use -e . from the project root
```

or, from a terminal, activate the environment first and run `pip install -e .` from the directory **containing `pyproject.toml`**. Unzipping sometimes creates a doubly-nested folder; confirm with `ls pyproject.toml` (Windows: `dir pyproject.toml`).

If several copies of the project exist, check which one is actually imported:

```python
import fedulearn; print(fedulearn.__file__)
```

### `RuntimeError: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR`

Almost always exhausted or fragmented GPU memory from a previous run, not a code fault. Each baseline allocates `num_clients` models on the device, so a seven-baseline sweep is seven times the pressure of one.

- Restart the kernel (notebook) or start a fresh process (CLI). `run_experiment` calls `free_gpu()` between models, but that cannot reclaim a leaked CUDA context.
- Lower `--batch-size` or `--num-clients`.
- Verify with `--smoke-test` before a full sweep.
- Under DP, confirm `PrivacyEngine` is not wrapping an already-wrapped model — re-running a definitions cell without restarting is the usual cause.

### `ModuleNotFoundError: No module named 'tensorflow'`

Expected. TensorFlow is optional and needed only for `fedavg_tf1cnn`. Install `requirements-tf.txt` or omit that baseline.

### `Opacus: Secure RNG turned off`

Expected during experimentation. Enable `secure_mode=True` in `PrivacyEngine` for final runs; it slows training substantially.

### `ValueError: Unknown configuration keys`

A YAML key does not match a `Parameters` field. The error lists the offenders. This is deliberate, so typos fail loudly.

### A client has no samples under extreme skew

The partitioner guarantees a minimum per client and redistributes from the largest donor. If it still occurs, raise `--shards-per-client` or `--dirichlet-alpha`, or lower `--num-clients`.

---

## Citation
If you use this code for your work, please cite the paper and if you want to discuss further, please contact me via [Email](triumphantmindstech@gmail.com)

```latex
@article{Ogobuchi2026fedulearn,
  title={Federated Unified Learning for IoT Device Fingerprinting and Intrusion Detection - An Optimized Privacy-Preserving Framework with Dynamic Feature Adaptation},
  author={Ogobuchi Daniel Okey, Sajjad Dadkhah, Rongxing Lu, Demóstenes Zegarra Rodríguez, João Henrique Kleinschmidt},
  journal={Internet of Things},
  year={2026},
  publisher={Under Review},
}
```
---
- **Datasets Used:** [NIM LAB IoT Dataset 2025-1](https://ieee-dataport.org/documents/dalhousie-nims-lab-iot-attack-dataset-2025-1) and [CICIoMT2024](https://www.unb.ca/cic/datasets/iomt-dataset-2024.html)

## Acknowledgement
We acknowledge that during the development of the project, Claude Sonet 4.6 was used to assist in the code refinement, error fixing and documentation. The authors take full responsibility of the code.
## License

MIT.
