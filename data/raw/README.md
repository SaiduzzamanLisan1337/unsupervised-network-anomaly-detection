# UNSW-NB15 raw data

The actual UNSW-NB15 dataset is intentionally **not committed to this repository** because of its size and dataset-specific usage conditions.

Official source:
https://research.unsw.edu.au/projects/unsw-nb15-dataset

Download the four CSV partitions plus the feature-definition file and place them here:

```text
data/raw/
├── UNSW-NB15_1.csv
├── UNSW-NB15_2.csv
├── UNSW-NB15_3.csv
├── UNSW-NB15_4.csv
├── UNSW-NB15_features.csv
└── UNSW-NB15_LIST_EVENTS.csv   # optional reference file
```

The project uses the four partitions and `UNSW-NB15_features.csv`. The event-list file is not required for training.

Check the layout before preparation:

```bash
PYTHONPATH=src python scripts/check_unsw.py
```

Then prepare the combined CSV:

```bash
PYTHONPATH=src python scripts/prepare_unsw.py
```
