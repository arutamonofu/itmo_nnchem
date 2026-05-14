PYTHON ?= python
CLI := PYTHONPATH=src $(PYTHON) -m perovskite_screening.cli

.PHONY: data smoke final-reduced check-final-reduced final-smoke all summary validate splits test

data:
	$(CLI) prepare-data --config configs/data/matbench_perovskites.yaml
	$(CLI) make-splits --config configs/default.yaml

splits:
	$(CLI) make-splits --config configs/default.yaml

smoke:
	$(CLI) run-suite --config configs/experiments/descriptor_rf.yaml --split-strategy random_iid --budgets B500 --seeds 42
	$(CLI) collect --runs-dir outputs/runs --out outputs/summary/results.csv

final-reduced:
	bash scripts/run_reduced_final_suite.sh

check-final-reduced:
	PYTHONPATH=src python scripts/check_reduced_protocol_ready.py

final-smoke:
	$(CLI) prepare-data --config configs/data/matbench_perovskites.yaml
	$(CLI) make-splits --config configs/default.yaml
	$(CLI) run-suite --config configs/experiments/descriptor_xgb.yaml --split-strategy random_iid --budgets B500 --seeds 42
	$(CLI) run-suite --config configs/experiments/cgcnn.yaml --split-strategy random_iid --budgets B500 --seeds 42
	$(CLI) run-suite --config configs/experiments/matgl.yaml --split-strategy random_iid --budgets B500 --seeds 42
	$(CLI) collect --runs-dir outputs/runs --out outputs/summary/results.csv
	$(CLI) evaluate-tail-metrics --results outputs/summary/results.csv --out outputs/summary/tail_metrics.csv
	$(CLI) validate-results --runs-dir outputs/runs

all: data smoke

summary:
	$(CLI) collect --runs-dir outputs/runs --out outputs/summary/results.csv

validate:
	$(CLI) validate-results --runs-dir outputs/runs

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests/unit
