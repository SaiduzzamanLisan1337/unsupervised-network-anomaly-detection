# Changelog

## 1.2.0 — 2026-10-07

- Rebuilt the repository for fresh-clone reproducibility.
- Added a direct `scripts/train.py` wrapper with sensible defaults, so it no longer requires a `train` subcommand.
- Added a direct `scripts/predict.py` wrapper.
- Added `python -m anomaly_detection` and an `anomaly-detect` console entry point.
- Added UNSW-NB15 preflight validation with actionable missing-file messages.
- Added batched inference to reduce memory pressure on larger datasets.
- Honoured the configured missing-value strategy.
- Removed dead label-handling code from preprocessing.
- Added CI for Python 3.10–3.13.
- Added a safer Makefile workflow and explicit sample-data documentation.
- Hardened the configuration validation and model-loading checks.

## 1.1.0 — 2026-10-07

- Added first-class UNSW-NB15 preparation support.
- Added leakage-aware UNSW configuration.
- Added chunked raw-data preparation.
- Added benign-validation quantile thresholding.
- Added reconstruction-based feature attribution.

## 1.0.0 — 2026-10-06

- Initial unsupervised Autoencoder + Isolation Forest research baseline.
