from pathlib import Path

import pandas as pd

from anomaly_detection.data import load_csv, split_data


ROOT = Path(__file__).resolve().parents[1]


def test_sample_dataset_loads():
    df = load_csv(ROOT / "data/sample/traffic_sample.csv")
    assert len(df) > 0
    assert "label" in df.columns


def test_split_keeps_autoencoder_training_benign_only():
    df = pd.DataFrame(
        {
            "flow_id": range(20),
            "bytes": list(range(100, 120)),
            "protocol": ["tcp"] * 18 + ["udp", "udp"],
            "label": ["normal"] * 15 + ["dos"] * 5,
        }
    )
    splits, features = split_data(
        df,
        label_column="label",
        benign_label="normal",
        id_columns=["flow_id"],
        drop_columns=[],
        feature_columns=[],
        test_size=0.2,
        validation_size=0.2,
        seed=42,
    )
    assert set(splits.train_normal["label"]) == {"normal"}
    assert set(splits.val_normal["label"]) == {"normal"}
    assert "bytes" in features and "protocol" in features


def test_sample_smoke_script_defaults_are_documented():
    assert (ROOT / "scripts/train.py").exists()
    assert (ROOT / "scripts/predict.py").exists()
    assert (ROOT / "scripts/check_unsw.py").exists()
