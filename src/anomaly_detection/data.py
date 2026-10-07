from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


@dataclass(frozen=True)
class DatasetSplits:
    train_normal: pd.DataFrame
    val_normal: pd.DataFrame
    test: pd.DataFrame
    has_labels: bool


@dataclass(frozen=True)
class PreparedData:
    preprocessor: ColumnTransformer
    x_train: np.ndarray
    x_val_normal: np.ndarray
    x_test: np.ndarray
    feature_names: list[str]
    source_feature_map: list[str]


def load_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path.resolve()}")
    frame = pd.read_csv(path, low_memory=False)
    if frame.empty:
        raise ValueError(f"CSV is empty: {path.resolve()}")
    return frame.replace([np.inf, -np.inf], np.nan)


def split_data(
    df: pd.DataFrame,
    *,
    label_column: str,
    benign_label: str,
    id_columns: Iterable[str],
    drop_columns: Iterable[str],
    feature_columns: Iterable[str] | None,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[DatasetSplits, list[str]]:
    """Create train/validation/test splits with normal-only model-training data.

    Labels are used only for benchmark splitting/filtering/evaluation. The model and
    default threshold never receive attack labels.
    """
    work = df.copy()
    ids = [c for c in id_columns if c in work.columns]
    drops = [c for c in drop_columns if c in work.columns]

    if label_column in work.columns:
        labels = work[label_column].astype(str)
        y_binary = (labels != str(benign_label)).astype(np.int8)
        stratify = y_binary if y_binary.nunique() > 1 else None
        train_val, test = train_test_split(
            work,
            test_size=test_size,
            random_state=seed,
            stratify=stratify,
        )

        train_val_binary = (train_val[label_column].astype(str) != str(benign_label)).astype(np.int8)
        val_fraction_of_trainval = validation_size / (1.0 - test_size)
        stratify_train_val = train_val_binary if train_val_binary.nunique() > 1 else None
        train, val = train_test_split(
            train_val,
            test_size=val_fraction_of_trainval,
            random_state=seed,
            stratify=stratify_train_val,
        )

        train_normal = train[train[label_column].astype(str) == str(benign_label)].copy()
        val_normal = val[val[label_column].astype(str) == str(benign_label)].copy()
        if train_normal.empty:
            raise ValueError("No benign samples remain in the training split.")
        if val_normal.empty:
            raise ValueError("No benign samples remain in the validation split.")
        splits = DatasetSplits(train_normal, val_normal, test.copy(), True)
    else:
        train, test = train_test_split(work, test_size=test_size, random_state=seed)
        val_fraction_of_trainval = validation_size / (1.0 - test_size)
        train, val_normal = train_test_split(
            train,
            test_size=val_fraction_of_trainval,
            random_state=seed,
        )
        splits = DatasetSplits(train.copy(), val_normal.copy(), test.copy(), False)

    excluded = set(ids) | set(drops) | {label_column}
    requested = list(feature_columns or [])
    if requested:
        missing = [c for c in requested if c not in work.columns]
        if missing:
            raise ValueError(f"Configured feature columns missing from CSV: {missing}")
        features = requested
    else:
        features = [c for c in work.columns if c not in excluded]

    if not features:
        raise ValueError("No feature columns remain after excluding label/id columns.")

    # Remove columns that are completely null in benign training data only.
    usable = [c for c in features if not splits.train_normal[c].isna().all()]
    removed = sorted(set(features) - set(usable))
    if removed:
        print(f"Ignoring all-null training features: {', '.join(removed)}")
    if not usable:
        raise ValueError("All candidate features are null in the benign training split.")

    return splits, usable


def _normalization_strategy(strategy: str, *, numeric: bool) -> str:
    allowed = {"median", "mean", "most_frequent", "constant"}
    if strategy not in allowed:
        raise ValueError(f"Unsupported missing_value_strategy={strategy!r}; choose from {sorted(allowed)}")
    if strategy == "most_frequent" or strategy == "constant":
        return strategy
    return strategy if numeric else "most_frequent"


def make_preprocessor(
    train_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    missing_value_strategy: str = "median",
) -> ColumnTransformer:
    numeric = train_df[feature_columns].select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]

    transformers = []
    if numeric:
        num_imputer = SimpleImputer(strategy=_normalization_strategy(missing_value_strategy, numeric=True))
        numeric_pipe = Pipeline(
            [
                ("imputer", num_imputer),
                ("scaler", RobustScaler()),
            ]
        )
        transformers.append(("num", numeric_pipe, numeric))

    if categorical:
        cat_imputer = SimpleImputer(strategy=_normalization_strategy(missing_value_strategy, numeric=False))
        categorical_pipe = Pipeline(
            [
                ("imputer", cat_imputer),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)),
            ]
        )
        transformers.append(("cat", categorical_pipe, categorical))

    if not transformers:
        raise ValueError("No numeric or categorical features are available for preprocessing.")

    return ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0)


def _feature_names_and_sources(preprocessor: ColumnTransformer) -> tuple[list[str], list[str]]:
    names = [str(x) for x in preprocessor.get_feature_names_out()]
    candidate_sources = sorted(
        [c for _, _, cols in preprocessor.transformers_ if cols not in ("drop", "passthrough") for c in cols],
        key=len,
        reverse=True,
    )
    sources: list[str] = []
    for name in names:
        body = name.split("__", 1)[-1]
        source = next(
            (candidate for candidate in candidate_sources if body == candidate or body.startswith(candidate + "_")),
            body,
        )
        sources.append(source)
    return names, sources


def prepare_data(
    splits: DatasetSplits,
    feature_columns: list[str],
    *,
    missing_value_strategy: str = "median",
) -> PreparedData:
    preprocessor = make_preprocessor(
        splits.train_normal,
        feature_columns,
        missing_value_strategy=missing_value_strategy,
    )
    x_train = preprocessor.fit_transform(splits.train_normal[feature_columns]).astype(np.float32, copy=False)
    x_val = preprocessor.transform(splits.val_normal[feature_columns]).astype(np.float32, copy=False)
    x_test = preprocessor.transform(splits.test[feature_columns]).astype(np.float32, copy=False)

    names, sources = _feature_names_and_sources(preprocessor)
    return PreparedData(preprocessor, x_train, x_val, x_test, names, sources)


def save_preprocessor(preprocessor: ColumnTransformer, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: str | Path) -> ColumnTransformer:
    return joblib.load(path)
