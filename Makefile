PYTHON ?= python
CLI := PYTHONPATH=src $(PYTHON) -m perovskite_screening.cli

.PHONY: data smoke all summary validate splits test

data:
	$(CLI) prepare-data --config configs/data/matbench_perovskites.yaml
	$(CLI) make-splits --config configs/default.yaml

splits:
	$(CLI) make-splits --config configs/default.yaml

smoke:
	$(CLI) run-suite --config configs/experiments/descriptor_rf.yaml --split-strategy random_iid --budgets B500 --seeds 42
	$(CLI) collect --runs-dir outputs/runs --out outputs/summary/results.csv

all: data smoke

summary:
	$(CLI) collect --runs-dir outputs/runs --out outputs/summary/results.csv

validate:
	$(CLI) validate-results --runs-dir outputs/runs

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q tests/unit
