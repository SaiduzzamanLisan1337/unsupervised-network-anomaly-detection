from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import argparse

from anomaly_detection.pipeline import predict_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Autoencoder anomaly inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-dir", default="artifacts/sample_experiment", help="Trained experiment directory")
    parser.add_argument("--input", default="data/sample/traffic_sample.csv", help="Input CSV")
    parser.add_argument("--output", default="reports/predictions.csv", help="Prediction CSV")
    args = parser.parse_args()
    frame = predict_dataset(args.model_dir, args.input, args.output)
    print(f"Wrote {len(frame):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
