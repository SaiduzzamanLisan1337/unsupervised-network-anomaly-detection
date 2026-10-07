from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


DEFAULT_PARTITIONS = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
]


def feature_names(features_path: Path) -> list[str]:
    if not features_path.exists():
        raise FileNotFoundError(
            f"Feature definition file not found: {features_path.resolve()}\n"
            "Download UNSW-NB15_features.csv from the official UNSW-NB15 dataset page "
            "and place it under data/raw/."
        )
    info = pd.read_csv(features_path, encoding="ISO-8859-1")
    name_col = next((c for c in info.columns if str(c).strip().lower() == "name"), None)
    if name_col is None:
        raise ValueError("UNSW-NB15_features.csv must contain a 'Name' column.")
    names = [str(value).strip().lower() for value in info[name_col].tolist() if str(value).strip()]
    if len(names) != 49:
        raise ValueError(f"Expected 49 UNSW-NB15 column names, found {len(names)}.")
    return names


def _header_matches(path: Path, expected: list[str]) -> bool:
    first = pd.read_csv(path, nrows=0, encoding="ISO-8859-1").columns.tolist()
    normalized = [str(x).strip().lower() for x in first]
    return normalized == expected


def combine(
    input_dir: Path,
    features_path: Path,
    output: Path,
    files: list[str],
    max_rows: int | None,
    chunksize: int,
) -> None:
    names = feature_names(features_path)
    missing = [name for name in files if not (input_dir / name).exists()]
    if missing:
        expected = "\n".join(f"  - {name}" for name in missing)
        raise FileNotFoundError(
            f"Missing UNSW-NB15 input file(s) under {input_dir.resolve()}:\n{expected}\n"
            "See data/raw/README.md for the expected dataset layout."
        )
    if chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    if max_rows is not None and max_rows <= 0:
        raise ValueError("--max-rows must be positive when supplied")

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    total = 0
    first_write = True
    for filename in files:
        path = input_dir / filename
        has_header = _header_matches(path, names)
        reader = pd.read_csv(
            path,
            header=0 if has_header else None,
            names=None if has_header else names,
            encoding="ISO-8859-1",
            low_memory=False,
            chunksize=chunksize,
        )
        for chunk in reader:
            chunk.columns = [str(c).strip().lower() for c in chunk.columns]
            if len(chunk.columns) != len(names):
                raise ValueError(
                    f"{path.name}: expected {len(names)} columns, found {len(chunk.columns)}."
                )
            if max_rows is not None:
                remaining = max_rows - total
                if remaining <= 0:
                    break
                chunk = chunk.iloc[:remaining]

            chunk.to_csv(output, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            total += len(chunk)
            if max_rows is not None and total >= max_rows:
                break
        print(f"Processed {filename}")
        if max_rows is not None and total >= max_rows:
            break

    if total == 0:
        raise ValueError("No rows were written.")
    print(f"Wrote {total:,} rows to {output.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine the four raw UNSW-NB15 CSV partitions into one pipeline-ready CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", default="data/raw")
    parser.add_argument("--features", default="data/raw/UNSW-NB15_features.csv")
    parser.add_argument("--output", default="data/processed/UNSW_NB15.csv")
    parser.add_argument("--files", nargs="+", default=DEFAULT_PARTITIONS)
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for a quick smoke test")
    parser.add_argument("--chunksize", type=int, default=100_000)
    args = parser.parse_args()
    combine(Path(args.input_dir), Path(args.features), Path(args.output), args.files, args.max_rows, args.chunksize)


if __name__ == "__main__":
    main()
