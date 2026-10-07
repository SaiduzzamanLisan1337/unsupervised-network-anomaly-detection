from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_recall_curve, roc_curve

from .config import Config, load_config
from .data import load_csv, prepare_data, save_preprocessor, split_data
from .metrics import binary_metrics, save_metrics
from .models import DenoisingAutoencoder, get_device, load_autoencoder, save_autoencoder, train_autoencoder
from .seed import set_seed
from .thresholds import quantile_threshold
from .utils import ensure_dir, write_json, write_yaml


def _save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_training(history, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history.train_loss, label="train")
    ax.plot(history.val_loss, label="benign validation")
    ax.set_title("Autoencoder reconstruction loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.legend()
    _save_plot(fig, out)


def _plot_distributions(ae_scores: np.ndarray, if_scores: np.ndarray, y_true: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    benign = y_true == 0
    attack = y_true == 1
    axes[0].hist(ae_scores[benign], bins=50, alpha=0.7, density=True, label="benign")
    axes[0].hist(ae_scores[attack], bins=50, alpha=0.7, density=True, label="attack")
    axes[0].set_title("Autoencoder anomaly scores")
    axes[0].set_xlabel("Reconstruction error")
    axes[0].set_ylabel("Density")
    axes[0].legend()
    axes[1].hist(if_scores[benign], bins=50, alpha=0.7, density=True, label="benign")
    axes[1].hist(if_scores[attack], bins=50, alpha=0.7, density=True, label="attack")
    axes[1].set_title("Isolation Forest anomaly scores")
    axes[1].set_xlabel("-IsolationForest score_samples")
    axes[1].set_ylabel("Density")
    axes[1].legend()
    _save_plot(fig, out)


def _plot_roc_pr(ae_scores: np.ndarray, if_scores: np.ndarray, y_true: np.ndarray, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fpr, tpr, _ = roc_curve(y_true, ae_scores)
    axes[0].plot(fpr, tpr, label="Autoencoder")
    fpr, tpr, _ = roc_curve(y_true, if_scores)
    axes[0].plot(fpr, tpr, label="Isolation Forest")
    axes[0].plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    axes[0].set_title("ROC curve")
    axes[0].set_xlabel("False-positive rate")
    axes[0].set_ylabel("True-positive rate")
    axes[0].legend()

    precision, recall, _ = precision_recall_curve(y_true, ae_scores)
    axes[1].plot(recall, precision, label="Autoencoder")
    precision, recall, _ = precision_recall_curve(y_true, if_scores)
    axes[1].plot(recall, precision, label="Isolation Forest")
    axes[1].axhline(float(np.mean(y_true)), linestyle="--", linewidth=1, label="attack prevalence")
    axes[1].set_title("Precision-recall curve")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend()
    _save_plot(fig, out)


def _aggregate_contributions(contrib: np.ndarray, source_map: list[str]) -> pd.DataFrame:
    unique_sources = list(dict.fromkeys(source_map))
    result = np.zeros((len(contrib), len(unique_sources)), dtype=np.float32)
    groups = {source: [i for i, item in enumerate(source_map) if item == source] for source in unique_sources}
    for column, source in enumerate(unique_sources):
        result[:, column] = contrib[:, groups[source]].sum(axis=1)
    return pd.DataFrame(result, columns=unique_sources)


def _predict_ae(
    model: DenoisingAutoencoder,
    x: np.ndarray,
    device: torch.device,
    *,
    batch_size: int,
    return_contributions: bool = False,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    scores: list[np.ndarray] = []
    reconstructions: list[np.ndarray] = []
    contributions: list[np.ndarray] = []
    model.eval()

    for start in range(0, len(x), batch_size):
        batch_np = x[start : start + batch_size]
        batch = torch.from_numpy(batch_np).to(device)
        recon, contrib = model.reconstruction_contributions(batch)
        scores.append(contrib.mean(dim=1).cpu().numpy())
        if return_contributions:
            reconstructions.append(recon.cpu().numpy())
            contributions.append(contrib.cpu().numpy())

    score_array = np.concatenate(scores) if scores else np.empty(0, dtype=np.float32)
    recon_array = np.concatenate(reconstructions) if return_contributions else None
    contrib_array = np.concatenate(contributions) if return_contributions else None
    return score_array, recon_array, contrib_array


def _binary_labels(frame: pd.DataFrame, label_column: str, benign_label: str) -> np.ndarray | None:
    if label_column not in frame.columns:
        return None
    return (frame[label_column].astype(str).to_numpy() != str(benign_label)).astype(np.int8)


def _label_counts(frame: pd.DataFrame, label_column: str) -> dict[str, int]:
    if label_column not in frame.columns:
        return {}
    return {str(k): int(v) for k, v in frame[label_column].astype(str).value_counts().to_dict().items()}


def train_experiment(config_path: str, data_path: str, output_dir: str) -> dict:
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    out = ensure_dir(output_dir)
    plots = ensure_dir(out / "plots")

    print(f"Loading dataset: {Path(data_path).resolve()}")
    df = load_csv(data_path)
    splits, features = split_data(
        df,
        label_column=cfg.label_column,
        benign_label=cfg.benign_label,
        id_columns=cfg.id_columns,
        drop_columns=cfg.drop_columns,
        feature_columns=cfg.feature_columns,
        test_size=cfg.test_size,
        validation_size=cfg.validation_size,
        seed=cfg.seed,
    )
    print(
        f"Split: train_normal={len(splits.train_normal):,}, "
        f"val_normal={len(splits.val_normal):,}, test={len(splits.test):,}"
    )
    prepared = prepare_data(
        splits,
        features,
        missing_value_strategy=cfg.missing_value_strategy,
    )
    print(f"Features: {len(features)} raw → {prepared.x_train.shape[1]} transformed")

    write_yaml(_config_to_dict(cfg), out / "config_used.yaml")
    save_preprocessor(prepared.preprocessor, out / "preprocessor.joblib")
    write_json(
        {
            "feature_columns": features,
            "transformed_feature_names": prepared.feature_names,
            "source_feature_map": prepared.source_feature_map,
            "input_dim": int(prepared.x_train.shape[1]),
        },
        out / "feature_metadata.json",
    )

    device = get_device(cfg.runtime.device)
    print(f"Device: {device}")
    model = DenoisingAutoencoder(
        input_dim=prepared.x_train.shape[1],
        hidden_dims=cfg.model.hidden_dims,
        latent_dim=cfg.model.latent_dim,
        dropout=cfg.model.dropout,
    )
    history = train_autoencoder(
        model,
        prepared.x_train,
        prepared.x_val_normal,
        batch_size=cfg.model.batch_size,
        max_epochs=cfg.model.max_epochs,
        patience=cfg.model.patience,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        noise_std=cfg.model.noise_std,
        gradient_clip_norm=cfg.model.gradient_clip_norm,
        device=device,
        num_workers=cfg.runtime.num_workers,
    )
    save_autoencoder(
        model,
        out / "autoencoder.pt",
        {
            "input_dim": int(prepared.x_train.shape[1]),
            "hidden_dims": cfg.model.hidden_dims,
            "latent_dim": cfg.model.latent_dim,
            "dropout": cfg.model.dropout,
        },
    )
    pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.train_loss) + 1),
            "train_loss": history.train_loss,
            "val_loss": history.val_loss,
        }
    ).to_csv(out / "training_history.csv", index=False)
    _plot_training(history, plots / "training_curve.png")

    val_ae_scores, _, _ = _predict_ae(
        model,
        prepared.x_val_normal,
        device,
        batch_size=cfg.runtime.inference_batch_size,
    )
    test_ae_scores, _, test_contrib = _predict_ae(
        model,
        prepared.x_test,
        device,
        batch_size=cfg.runtime.inference_batch_size,
        return_contributions=True,
    )
    ae_threshold = quantile_threshold(val_ae_scores, cfg.threshold.quantile)

    iso = IsolationForest(
        n_estimators=cfg.isolation_forest.n_estimators,
        max_samples=cfg.isolation_forest.max_samples,
        contamination=cfg.isolation_forest.contamination,
        random_state=cfg.seed,
        n_jobs=cfg.isolation_forest.n_jobs,
    )
    print("Training Isolation Forest baseline...")
    iso.fit(prepared.x_train)
    val_if_scores = -iso.score_samples(prepared.x_val_normal)
    test_if_scores = -iso.score_samples(prepared.x_test)
    if_threshold = quantile_threshold(val_if_scores, cfg.threshold.quantile)
    joblib.dump(iso, out / "isolation_forest.joblib")

    labels_test = _binary_labels(splits.test, cfg.label_column, cfg.benign_label)
    metrics: dict = {
        "experiment": {
            "question": "Can a denoising autoencoder trained only on benign traffic outperform Isolation Forest under the same representation and false-alarm target?",
            "threshold_policy": f"{cfg.threshold.quantile:.3f} quantile of benign validation scores",
            "attack_labels_used_for_training": False,
        },
        "dataset": {
            "path": str(Path(data_path).resolve()),
            "n_rows": int(len(df)),
            "n_features_before_transform": int(len(features)),
            "n_features_after_transform": int(prepared.x_train.shape[1]),
            "n_train_normal": int(len(splits.train_normal)),
            "n_validation_normal": int(len(splits.val_normal)),
            "n_test": int(len(splits.test)),
            "label_counts_full": _label_counts(df, cfg.label_column),
            "label_counts_test": _label_counts(splits.test, cfg.label_column),
            "labels_available": labels_test is not None,
        },
        "autoencoder": {
            "threshold_quantile": cfg.threshold.quantile,
            "threshold": ae_threshold,
            "validation_benign_fpr": float(np.mean(val_ae_scores >= ae_threshold)),
        },
        "isolation_forest": {
            "threshold_quantile": cfg.threshold.quantile,
            "threshold": if_threshold,
            "validation_benign_fpr": float(np.mean(val_if_scores >= if_threshold)),
        },
    }

    predictions = splits.test.copy().reset_index(drop=True)
    predictions["autoencoder_score"] = test_ae_scores
    predictions["autoencoder_anomaly"] = (test_ae_scores >= ae_threshold).astype(np.int8)
    predictions["isolation_forest_score"] = test_if_scores
    predictions["isolation_forest_anomaly"] = (test_if_scores >= if_threshold).astype(np.int8)
    predictions.to_csv(out / "test_predictions.csv", index=False)

    assert test_contrib is not None
    contribution_df = _aggregate_contributions(test_contrib, prepared.source_feature_map)
    contribution_columns = list(contribution_df.columns)
    contribution_df.insert(0, "autoencoder_score", test_ae_scores)
    contribution_df["ranked_features"] = [
        ", ".join(np.asarray(contribution_columns)[np.argsort(row)[::-1][:5]].tolist())
        for row in contribution_df[contribution_columns].to_numpy()
    ]
    contribution_df.to_csv(out / "feature_contributions.csv", index=False)

    if labels_test is not None and len(np.unique(labels_test)) == 2:
        metrics["evaluation"] = {
            "autoencoder": binary_metrics(labels_test, test_ae_scores, ae_threshold),
            "isolation_forest": binary_metrics(labels_test, test_if_scores, if_threshold),
        }
        _plot_distributions(test_ae_scores, test_if_scores, labels_test, plots / "score_distributions.png")
        _plot_roc_pr(test_ae_scores, test_if_scores, labels_test, plots / "roc_pr_curves.png")

    save_metrics(metrics, out / "metrics.json")
    write_json(
        {
            "seed": cfg.seed,
            "device": str(device),
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": __import__("sklearn").__version__,
        },
        out / "experiment_summary.json",
    )
    return metrics


def predict_dataset(model_dir: str, input_path: str, output_path: str) -> pd.DataFrame:
    model_dir = Path(model_dir)
    required = [
        "feature_metadata.json",
        "config_used.yaml",
        "preprocessor.joblib",
        "autoencoder.pt",
        "metrics.json",
    ]
    missing_files = [name for name in required if not (model_dir / name).exists()]
    if missing_files:
        raise FileNotFoundError(f"Model directory is incomplete; missing: {', '.join(missing_files)}")

    metadata = json.loads((model_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    cfg = load_config(model_dir / "config_used.yaml")
    device = get_device(cfg.runtime.device)
    preprocessor = joblib.load(model_dir / "preprocessor.joblib")
    model = load_autoencoder(model_dir / "autoencoder.pt", device)
    frame = load_csv(input_path)

    missing = [c for c in metadata["feature_columns"] if c not in frame.columns]
    if missing:
        raise ValueError(f"Input is missing required features: {missing}")
    x = preprocessor.transform(frame[metadata["feature_columns"]]).astype(np.float32, copy=False)
    scores, _, _ = _predict_ae(
        model,
        x,
        device,
        batch_size=cfg.runtime.inference_batch_size,
    )
    threshold = float(json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["autoencoder"]["threshold"])

    result = frame.copy()
    result["autoencoder_score"] = scores
    result["anomaly"] = (scores >= threshold).astype(np.int8)

    if (model_dir / "isolation_forest.joblib").exists():
        isolation_forest = joblib.load(model_dir / "isolation_forest.joblib")
        isolation_scores = -isolation_forest.score_samples(x)
        isolation_threshold = float(
            json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))["isolation_forest"]["threshold"]
        )
        result["isolation_forest_score"] = isolation_scores
        result["isolation_forest_anomaly"] = (isolation_scores >= isolation_threshold).astype(np.int8)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    return result


def _config_to_dict(cfg: Config) -> dict:
    return {
        "seed": cfg.seed,
        "label_column": cfg.label_column,
        "benign_label": cfg.benign_label,
        "id_columns": cfg.id_columns,
        "drop_columns": cfg.drop_columns,
        "feature_columns": cfg.feature_columns,
        "missing_value_strategy": cfg.missing_value_strategy,
        "test_size": cfg.test_size,
        "validation_size": cfg.validation_size,
        "model": {
            "latent_dim": cfg.model.latent_dim,
            "hidden_dims": cfg.model.hidden_dims,
            "dropout": cfg.model.dropout,
            "noise_std": cfg.model.noise_std,
            "learning_rate": cfg.model.learning_rate,
            "weight_decay": cfg.model.weight_decay,
            "batch_size": cfg.model.batch_size,
            "max_epochs": cfg.model.max_epochs,
            "patience": cfg.model.patience,
            "gradient_clip_norm": cfg.model.gradient_clip_norm,
        },
        "threshold": {"mode": cfg.threshold.mode, "quantile": cfg.threshold.quantile},
        "isolation_forest": {
            "n_estimators": cfg.isolation_forest.n_estimators,
            "max_samples": cfg.isolation_forest.max_samples,
            "contamination": cfg.isolation_forest.contamination,
            "n_jobs": cfg.isolation_forest.n_jobs,
        },
        "runtime": {
            "device": cfg.runtime.device,
            "num_workers": cfg.runtime.num_workers,
            "inference_batch_size": cfg.runtime.inference_batch_size,
        },
    }
