from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    latent_dim: int = 12
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64, 32])
    dropout: float = 0.15
    noise_std: float = 0.05
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256
    max_epochs: int = 120
    patience: int = 15
    gradient_clip_norm: float = 5.0


@dataclass(frozen=True)
class ThresholdConfig:
    mode: str = "normal_quantile"
    quantile: float = 0.995


@dataclass(frozen=True)
class IFConfig:
    n_estimators: int = 400
    max_samples: str | int | float = "auto"
    contamination: str | float = "auto"
    n_jobs: int = -1


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "auto"
    num_workers: int = 0
    inference_batch_size: int = 8192


@dataclass(frozen=True)
class Config:
    seed: int = 42
    label_column: str = "label"
    benign_label: str = "normal"
    id_columns: list[str] = field(default_factory=lambda: ["flow_id"])
    drop_columns: list[str] = field(default_factory=list)
    feature_columns: list[str] = field(default_factory=list)
    missing_value_strategy: str = "median"
    test_size: float = 0.20
    validation_size: float = 0.20
    model: ModelConfig = field(default_factory=ModelConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    isolation_forest: IFConfig = field(default_factory=IFConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _merge_dataclass(cls: type, data: dict[str, Any]):
    fields = {field.name for field in cls.__dataclass_fields__.values()}
    return cls(**{key: value for key, value in data.items() if key in fields})


def load_config(path: str | Path) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path.resolve()}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = Config(
        seed=int(raw.get("seed", 42)),
        label_column=str(raw.get("label_column", "label")),
        benign_label=str(raw.get("benign_label", "normal")),
        id_columns=list(raw.get("id_columns", ["flow_id"])),
        drop_columns=list(raw.get("drop_columns", [])),
        feature_columns=list(raw.get("feature_columns", [])),
        missing_value_strategy=str(raw.get("missing_value_strategy", "median")),
        test_size=float(raw.get("test_size", 0.20)),
        validation_size=float(raw.get("validation_size", 0.20)),
        model=_merge_dataclass(ModelConfig, raw.get("model", {})),
        threshold=_merge_dataclass(ThresholdConfig, raw.get("threshold", {})),
        isolation_forest=_merge_dataclass(IFConfig, raw.get("isolation_forest", {})),
        runtime=_merge_dataclass(RuntimeConfig, raw.get("runtime", {})),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not 0 < cfg.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    if not 0 < cfg.validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1")
    if cfg.test_size + cfg.validation_size >= 1:
        raise ValueError("test_size + validation_size must be < 1")
    if cfg.model.latent_dim <= 0:
        raise ValueError("model.latent_dim must be positive")
    if any(width <= 0 for width in cfg.model.hidden_dims):
        raise ValueError("model.hidden_dims must contain only positive integers")
    if not 0 <= cfg.model.dropout < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if cfg.model.batch_size <= 0 or cfg.model.max_epochs <= 0 or cfg.model.patience <= 0:
        raise ValueError("batch_size, max_epochs and patience must be positive")
    if not 0 < cfg.threshold.quantile < 1:
        raise ValueError("threshold.quantile must be between 0 and 1")
    if cfg.threshold.mode != "normal_quantile":
        raise ValueError("Only unsupervised threshold mode 'normal_quantile' is supported")
    if cfg.missing_value_strategy not in {"median", "mean", "most_frequent", "constant"}:
        raise ValueError("missing_value_strategy must be median, mean, most_frequent, or constant")
    if cfg.runtime.num_workers < 0 or cfg.runtime.inference_batch_size <= 0:
        raise ValueError("runtime.num_workers must be >= 0 and inference_batch_size must be positive")
