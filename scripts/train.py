from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

import argparse

from anomaly_detection.pipeline import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the Autoencoder + Isolation Forest anomaly detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default="config/default.yaml", help="Experiment YAML configuration")
    parser.add_argument("--data", default="data/sample/traffic_sample.csv", help="Training/evaluation CSV")
    parser.add_argument("--output", default="artifacts/sample_experiment", help="Experiment output directory")
    args = parser.parse_args()
    train_experiment(args.config, args.data, args.output)


if __name__ == "__main__":
    main()
