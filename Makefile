PYTHON ?= python
export PYTHONPATH := src

.PHONY: install test sample predict check-unsw prepare-unsw train-unsw unsw-smoke clean

install:
	$(PYTHON) -m pip install -e ".[test]"

test:
	$(PYTHON) -m pytest -q

sample:
	$(PYTHON) scripts/train.py

predict:
	$(PYTHON) scripts/predict.py

check-unsw:
	$(PYTHON) scripts/check_unsw.py

prepare-unsw: check-unsw
	$(PYTHON) scripts/prepare_unsw.py

unsw-smoke: check-unsw
	$(PYTHON) scripts/prepare_unsw.py --files UNSW-NB15_1.csv --max-rows 100000 --output data/processed/UNSW_NB15_smoke.csv
	$(PYTHON) scripts/train.py --config config/unsw_nb15.yaml --data data/processed/UNSW_NB15_smoke.csv --output artifacts/unsw_smoke

train-unsw: prepare-unsw
	$(PYTHON) scripts/train.py --config config/unsw_nb15.yaml --data data/processed/UNSW_NB15.csv --output artifacts/unsw_nb15

clean:
	rm -rf artifacts/sample_experiment artifacts/unsw_smoke reports/predictions.csv data/processed/UNSW_NB15_smoke.csv .pytest_cache
