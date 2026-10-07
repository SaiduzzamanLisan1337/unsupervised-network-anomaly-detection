from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)


REQUIRED = [
    "UNSW-NB15_1.csv",
    "UNSW-NB15_2.csv",
    "UNSW-NB15_3.csv",
    "UNSW-NB15_4.csv",
    "UNSW-NB15_features.csv",
]


def main() -> None:
    root = Path("data/raw")
    missing = [name for name in REQUIRED if not (root / name).exists()]
    if missing:
        print("UNSW-NB15 dataset is NOT ready.")
        print(f"Expected directory: {root.resolve()}")
        print("Missing files:")
        for name in missing:
            print(f"  - {name}")
        print("\nSee data/raw/README.md for the official source and directory layout.")
        raise SystemExit(1)

    print("UNSW-NB15 dataset layout looks ready:")
    for name in REQUIRED:
        print(f"  ✓ {name}")


if __name__ == "__main__":
    main()
