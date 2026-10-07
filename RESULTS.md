# Sample experiment results

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
