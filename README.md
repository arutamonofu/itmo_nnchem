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

### CGCNN Dependencies

CGCNN uses PyTorch and PyTorch Geometric. Install its optional dependency set with:

```bash
pip install -r requirements/cgcnn.txt
```

The requirements file intentionally does not hard-code CUDA wheel indexes. For CUDA environments, first install the PyTorch build matching your driver/CUDA runtime from the official PyTorch selector, then use the PyTorch Geometric install selector if your torch/CUDA pair requires additional CUDA-specific wheels.

Check the import surface used by the CGCNN model and trainer:

```bash
python -c "import torch; import torch_geometric; from torch_geometric.nn import CGConv; print('CGCNN deps OK')"
```

Run a minimal CGCNN smoke experiment after data preparation and splits exist:

```bash
perovskite-screening run \
  --config configs/experiments/cgcnn.yaml \
  --split-strategy random_iid \
  --budget B500 \
  --seed 42
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

Run the reduced final protocol with:

```bash
make final-reduced
```

This prepares data, makes splits, runs XGB, CGCNN, and MatGL on `random_iid`
and `element_set` with budgets `B500`, `B2000`, `B8000`, and `Bfull` at seed
`42`, then collects and validates results. The protocol contains 24 expected
runs:

```text
3 models x 2 split strategies x 4 budgets x 1 seed = 24 runs
```

For a smaller final-protocol smoke test, run:

```bash
make final-smoke
```

This runs XGB, CGCNN, and MatGL only on `random_iid`, `B500`, and seed `42`.
Both targets collect the summary into `outputs/summary/results.csv`. Existing
run outputs are not deleted automatically before collection.

## Experiment Hyperparameters

Final-run hyperparameters are fixed before running the reported budget suite. The
pipeline does not perform per-budget or per-split tuning; each budget and split
uses the same model settings for a given experiment config. Model parameters are
stored in `configs/experiments/*.yaml` under `model.params`: XGB in
`descriptor_xgb.yaml`, scratch-trained CGCNN in `cgcnn.yaml`, and MatGL
fine-tuning in `matgl.yaml`. CGCNN monitors validation MAE after each epoch,
uses validation MAE for scheduler updates and early stopping, and restores the
best validation checkpoint before test inference. MatGL uses frozen fine-tuning
in the final config, with early-stopping patience set at config level.

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
Each result row includes `model_params_json`, a JSON object with the effective
model parameters used for the run.
