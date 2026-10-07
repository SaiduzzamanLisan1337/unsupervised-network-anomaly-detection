# Unsupervised Network Anomaly Detection with Autoencoders

**GitHub repository:** `autoencoder-network-anomaly-detection`

> Train a denoising Autoencoder mainly on normal network traffic, use reconstruction error as the anomaly score, and compare it with **Isolation Forest** under the same feature representation and normal-only training protocol.


## What this repository does

The project implements a reproducible, research-oriented anomaly-detection pipeline for tabular network-flow data.

### Primary detector — Denoising Autoencoder

The model learns to reconstruct benign traffic:

$$
A(x)=\frac{1}{d}\sum_{j=1}^{d}(x_j-\hat{x}_j)^2
$$

Higher reconstruction error indicates that a flow is less consistent with the learned benign traffic representation.

### Baseline — Isolation Forest

Isolation Forest is trained on the **same transformed benign-only training data** and produces an independent unsupervised anomaly score.

### Default threshold

The operational threshold is the configured quantile of scores measured on **benign validation traffic only**. The default is the 99.5th percentile.

This preserves the central research property: attack labels are not needed to train the detector or calibrate the default threshold.

## Architecture

```text
Network flows
     │
     ▼
Data validation
+ missing-value handling
+ leakage-aware feature exclusion
     │
     ▼
Fit preprocessing on benign training only
(imputation + RobustScaler + OneHotEncoder)
     │
     ├──────────────────────────┐
     ▼                          ▼
Benign training           Benign validation
     │                          │
     ▼                          ▼
Denoising Autoencoder      Threshold calibration
     │                          │
     └──────────────┬───────────┘
                    ▼
            Held-out test set
                    │
             ┌──────┴──────┐
             ▼             ▼
       Autoencoder   Isolation Forest
       score/error     anomaly score
             │             │
             └──────┬──────┘
                    ▼
              Comparative metrics
```

## Why it is different from supervised IDS classification

A supervised IDS typically learns:

```text
features → known attack class
```

This repository instead learns:

```text
normal traffic → benign representation → anomaly score
```

That makes it a useful baseline for research on:

- unknown or emerging attacks;
- open-world anomaly detection;
- distribution shift and concept drift;
- continual/adaptive learning;
- human-in-the-loop security monitoring.

## Repository layout

```text
.
├── config/
│   ├── default.yaml
│   └── unsw_nb15.yaml
├── data/
│   ├── raw/                 # ignored; place UNSW-NB15 here
│   ├── processed/           # ignored; generated CSVs
│   └── sample/              # bundled synthetic smoke test
├── docs/
│   └── EXPERIMENT_PROTOCOL.md
├── scripts/
│   ├── check_unsw.py
│   ├── prepare_unsw.py
│   ├── predict.py
│   └── train.py
├── src/anomaly_detection/
│   ├── cli.py
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   ├── models.py
│   ├── pipeline.py
│   ├── thresholds.py
│   └── ...
├── tests/
├── artifacts/               # ignored generated experiments
├── reports/                 # ignored generated reports
└── README.md
```

## Quick start — works without UNSW-NB15

Python 3.10–3.13 is covered by CI. Python 3.12 is a conservative local choice for research environments.

### Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
```

Then run the bundled sample:

```bash
python scripts/train.py
```

The default command trains the Autoencoder and Isolation Forest on `data/sample/traffic_sample.csv` and writes results to:

```text
artifacts/sample_experiment/
```

You can also use:

```bash
./run.sh
```

or:

```bash
make sample
```

### Why `scripts/train.py` has no required `train` subcommand

The repository also provides the lower-level package CLI:

```bash
python -m anomaly_detection train --config config/default.yaml --data data/sample/traffic_sample.csv --output artifacts/sample_experiment
```

But the PyCharm-friendly wrapper intentionally accepts direct training options:

```bash
python scripts/train.py
python scripts/train.py --config config/default.yaml --data data/sample/traffic_sample.csv --output artifacts/my_experiment
```

This avoids the common mistake of launching `train.py` with no command and receiving `the following arguments are required: command`.

## Prediction

After training:

```bash
python scripts/predict.py
```

The default writes:

```text
reports/predictions.csv
```

For a custom experiment:

```bash
python scripts/predict.py \
  --model-dir artifacts/my_experiment \
  --input data/sample/traffic_sample.csv \
  --output reports/my_predictions.csv
```

The prediction file contains the Autoencoder score and anomaly decision. When an Isolation Forest model is present, its score and decision are also included.

## What training produces

Each experiment directory contains:

```text
artifacts/<experiment>/
├── autoencoder.pt
├── isolation_forest.joblib
├── preprocessor.joblib
├── config_used.yaml
├── feature_metadata.json
├── training_history.csv
├── test_predictions.csv
├── feature_contributions.csv
├── metrics.json
├── experiment_summary.json
└── plots/
    ├── training_curve.png
    ├── score_distributions.png
    └── roc_pr_curves.png
```

`feature_contributions.csv` aggregates reconstruction error back to the original source feature, making high-scoring flows easier to inspect.

## UNSW-NB15

The main benchmark configuration is `config/unsw_nb15.yaml`.

UNSW Research states that UNSW-NB15 contains 2,540,044 records distributed across four CSV files (`UNSW-NB15_1.csv` through `UNSW-NB15_4.csv`) and describes 49 features in `UNSW-NB15_features.csv`.

### 1. Obtain the dataset

Use the official UNSW-NB15 source page:

https://research.unsw.edu.au/projects/unsw-nb15-dataset

The dataset page also provides the ground-truth/event resources and notes the academic-use citation requirements.

### 2. Place the required files in `data/raw/`

```text
data/raw/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv
├── UNSW-NB15_features.csv
└── UNSW-NB15_LIST_EVENTS.csv   # optional for this pipeline
```

Do **not** commit those large data files to GitHub. The repository `.gitignore` excludes them.

### 3. Verify the dataset layout

```bash
python scripts/check_unsw.py
```

The script gives an explicit list of missing files instead of failing later inside pandas.

### 4. Prepare the combined CSV

```bash
python scripts/prepare_unsw.py
```

For a quick 100,000-row smoke test:

```bash
python scripts/prepare_unsw.py \
  --files UNSW-NB15_1.csv \
  --max-rows 100000 \
  --output data/processed/UNSW_NB15_smoke.csv
```

### 5. Train on UNSW-NB15

```bash
python scripts/train.py \
  --config config/unsw_nb15.yaml \
  --data data/processed/UNSW_NB15.csv \
  --output artifacts/unsw_nb15
```

Or run the complete preparation + training workflow:

```bash
./run_unsw.sh
```

or:

```bash
make train-unsw
```

### Leakage-aware exclusions

The UNSW configuration excludes:

- `srcip`
- `dstip`
- `attack_cat`

`attack_cat` is a target/attack-family field and must not become an Autoencoder input. Source/destination IPs are treated as endpoint identifiers rather than generalizable behavioural features.

The `label` column is used only for benchmark splitting/filtering and final evaluation. The Autoencoder and Isolation Forest are trained only on benign rows.

## Evaluation

The pipeline reports:

**Ranking:**

- ROC-AUC
- PR-AUC

**At the unsupervised validation-calibrated threshold:**

- Precision
- Recall
- F1
- Specificity
- False-positive rate
- Balanced accuracy
- Confusion matrix

PR-AUC should receive particular attention when the attack class is imbalanced.

## Research protocol

See [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) for the controlled-comparison protocol, leakage rules, reproducibility requirements, and recommended ablations.

A strong next stage is to compare:

```text
static anomaly detection
        ↓
attack-type classification
        ↓
human review / feedback
        ↓
adaptive threshold or continual update
```

That provides a clean bridge from this baseline into a more substantial continual or human-in-the-loop intrusion-detection research project.

## Smoke-test results

These values come from the bundled **synthetic smoke-test dataset** only. They demonstrate that the rebuilt software runs end-to-end; they are **not UNSW-NB15 benchmark results** and should not be presented as published performance.

| Detector | ROC-AUC | PR-AUC | Precision | Recall | F1 | Specificity |
|---|---:|---:|---:|---:|---:|---:|
| Denoising Autoencoder | 0.9857 | 0.9640 | 0.9827 | 0.5152 | 0.6759 | 0.9966 |
| Isolation Forest | 0.8049 | 0.6170 | 0.8387 | 0.0788 | 0.1440 | 0.9943 |

The default sample configuration uses a 50-epoch maximum with early stopping and a 99.5th-percentile threshold calibrated from benign validation traffic only.

Reproduce with:

```bash
python -m pip install -e ".[test]"
python scripts/train.py
python scripts/predict.py
```

The experiment trains both detectors on benign samples only. Test labels are used only for final evaluation metrics.

## Testing

```bash
python -m pytest -q
```

The GitHub Actions workflow runs the test suite across Python 3.10–3.13.

## GitHub push checklist

Before the first push:

```bash
git init
git add .
git status
```

Confirm that these are **not** staged:

```text
data/raw/*.csv
data/processed/*.csv
artifacts/*
reports/*
.venv/
```

Then:

```bash
git commit -m "Build unsupervised network anomaly detection baseline"
git branch -M main
git remote add origin https://github.com/<YOUR-USERNAME>/autoencoder-network-anomaly-detection.git
git push -u origin main
```

## Citation

The repository includes `CITATION.cff`. For UNSW-NB15 itself, follow the citation guidance on the official UNSW dataset page.

## License

MIT. See [`LICENSE`](LICENSE).
