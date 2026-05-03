.PHONY: data smoke all summary validate

data:
	python scripts/01_load_dataset.py
	python scripts/02_make_splits.py

smoke:
	python scripts/03_smoke_mean_baseline.py
	python scripts/99_collect_results.py

all: data smoke

summary:
	python scripts/99_collect_results.py

validate:
	python scripts/98_validate_results.py
