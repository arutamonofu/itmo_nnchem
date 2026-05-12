# Perovskite Screening

`perovskite_screening` is a package-first experimental pipeline for reproducible ML comparisons on `matbench_perovskites` formation energy prediction.

The active experiment flow is:

```text
config -> data loading -> split strategy -> training budget -> model -> training/inference -> evaluation -> artifacts -> summary
```

## Installation

```bash
conda env create -f environment.yml
conda activate perovskite_screening
pip install -e ".[dev]"
```

Optional model dependencies remain separate:

```bash
pip install -r requirements/xgb.txt
pip install -r requirements/cgcnn.txt
pip install -r requirements/matgl.txt
pip install -r requirements/matgl-dgl.txt
```

## CLI

```bash
perovskite-screening prepare-data --config configs/data/matbench_perovskites.yaml
perovskite-screening make-splits --config configs/default.yaml

perovskite-screening run \
  --config configs/experiments/descriptor_rf.yaml \
  --split-strategy random_iid \
  --budget B500 \
  --seed 42

perovskite-screening run-suite \
  --config configs/experiments/descriptor_rf.yaml \
  --split-strategy random_iid \
  --budgets all \
  --seeds 42

perovskite-screening collect --runs-dir outputs/runs --out outputs/summary/results.csv
perovskite-screening validate-results --runs-dir outputs/runs
```

`make data`, `make smoke`, `make summary`, and `make validate` are thin wrappers around the same CLI.

## Python API

```python
from perovskite_screening.data.dataset import load_dataset
from perovskite_screening.data.splits import build_split
from perovskite_screening.data.budgets import build_budgets
from perovskite_screening.pipeline.run_experiment import run_experiment
```

## Data Contract

The canonical dataset is:

```text
data/processed/dataset.pkl
```

It is a pandas DataFrame with:

```text
sample_id
structure
target
```

Inspection artifacts are also written:

```text
data/processed/targets.csv
data/processed/structures.json.gz
data/processed/dataset_info.json
```

## Split And Budget Contract

Active split strategies:

```text
random_iid
element_set
```

Split files live under `data/splits/<strategy>/` and contain:

```text
sample_id
split
split_strategy
```

For `element_set`, each `element_set_key` belongs to exactly one of train, validation, or test.

Training budgets are absolute labeling budgets:

```text
B500
B1000
B2000
B4000
B8000
Bfull
```

Budgets are deterministic nested subsets of the fixed train split. Validation and test are fixed for every budget. `target_tails` is an evaluation slice, not a primary split strategy.

## Results

Runs write result CSV files to:

```text
outputs/runs/
outputs/runs/predictions/
```

Summaries are collected into:

```text
outputs/summary/results.csv
```

Required result columns are enforced by `perovskite-screening validate-results`.
