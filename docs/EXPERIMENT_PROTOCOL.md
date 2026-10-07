# Experiment protocol

## Research question

Can a denoising autoencoder trained only on benign network traffic detect anomalous flows more effectively than a classical Isolation Forest baseline under the same feature representation and false-alarm target?

## Primary hypothesis

A representation-learning model trained to reconstruct the benign manifold will separate anomalous traffic from normal traffic better than an isolation-based baseline, especially when anomalies differ through combinations of correlated flow features rather than isolated marginal outliers.

## Protocol

Both detectors use the same:

- raw input columns after leakage-aware exclusions;
- imputation, scaling, and one-hot encoding;
- benign-only training partition;
- benign validation partition;
- held-out test partition;
- validation-quantile threshold target.

Attack labels are not passed to either detector during training and are not used by the default threshold-calibration procedure.

For benchmark datasets with labels, labels are used only to create the train/validation/test benchmark split, isolate benign training/validation rows, and calculate final evaluation metrics.

## Metrics

Ranking metrics:

- ROC-AUC
- PR-AUC

Operational threshold metrics:

- precision
- recall / true-positive rate
- F1
- specificity
- false-positive rate
- balanced accuracy

PR-AUC should receive particular attention when attacks are rare because ROC-AUC alone can hide poor positive predictive value.

## Reproducibility requirements

For every reported experiment, preserve:

1. the exact YAML configuration;
2. the dataset version/source and preprocessing decision;
3. random seed;
4. train/validation/test row counts;
5. model architecture;
6. threshold quantile;
7. evaluation metrics;
8. plots and prediction outputs.

The training command automatically writes these artifacts under the selected experiment directory.

## Recommended ablations

1. Plain autoencoder vs denoising autoencoder.
2. Latent dimensions: 4, 8, 16, 32.
3. Threshold quantiles: 0.990, 0.995, 0.999.
4. Autoencoder vs Isolation Forest.
5. Random split vs time-ordered split.
6. Known vs unseen attack-family evaluation.
7. Static vs adaptive thresholding under distribution drift.
8. Single-dataset vs cross-dataset transfer.

## Research extension

A strong next stage is to place anomaly detection before a downstream attack-type classifier and route uncertain/high-risk flows to human review. This creates a natural bridge from static anomaly detection to adaptive, continual, and human-in-the-loop intrusion detection.
