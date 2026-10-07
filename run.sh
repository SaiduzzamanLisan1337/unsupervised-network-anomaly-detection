#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"
python scripts/train.py "$@"
