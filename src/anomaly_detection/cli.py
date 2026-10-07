from __future__ import annotations

import argparse
import json

from .pipeline import predict_dataset, train_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Unsupervised network anomaly detection with a denoising Autoencoder and Isolation Forest."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train both detectors and evaluate on a held-out test set")
    train.add_argument("--config", required=True, help="YAML experiment configuration")
    train.add_argument("--data", required=True, help="Input CSV")
    train.add_argument("--output", required=True, help="Directory for models, metrics and plots")

    predict = sub.add_parser("predict", help="Run deployment-style inference using a trained Autoencoder")
    predict.add_argument("--model-dir", required=True, help="Directory produced by the train command")
    predict.add_argument("--input", required=True, help="CSV containing the required model features")
    predict.add_argument("--output", required=True, help="Prediction CSV path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "train":
        metrics = train_experiment(args.config, args.data, args.output)
        print("\n=== Experiment summary ===")
        print(json.dumps(metrics, indent=2))
    else:
        frame = predict_dataset(args.model_dir, args.input, args.output)
        print(f"Wrote {len(frame):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
