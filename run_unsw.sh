#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

python scripts/check_unsw.py
python scripts/prepare_unsw.py \
  --input-dir data/raw \
  --features data/raw/UNSW-NB15_features.csv \
  --output data/processed/UNSW_NB15.csv
python scripts/train.py \
  --config config/unsw_nb15.yaml \
  --data data/processed/UNSW_NB15.csv \
  --output artifacts/unsw_nb15
