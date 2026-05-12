# Data-efficient screening of perovskites

This repository contains a minimal reproducible data and results skeleton for an educational ML project on formation energy prediction with `matbench_perovskites`.

The goal is to make all model comparisons use the same dataset, train / validation / test split, absolute training budgets, metrics, and result format. The test set is fixed and must not be changed. All models must report metrics on the same test set.

## Installation

### Base Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate itmo_nnchem
```

`environment.yml` is the conda entry point. The base Python package list lives in `requirements.txt`, so the base dependency constraints are not duplicated.

The base environment is enough for:

```text
data loading
split generation
mean baseline
descriptor RandomForest baseline
result validation and summary collection
```

The dataset is loaded through `matminer`; the legacy `matbench` package is intentionally not installed because its fixed dependency pins are not compatible with Python 3.11.

### Optional Model Dependencies

Deep learning dependencies are intentionally not included in the base environment. Install model-specific dependencies on top of the activated `itmo_nnchem` environment:

```bash
# CGCNN scaffold and PyTorch experiments
pip install -r requirements/cgcnn.txt

# MatGL scaffold with the default current MatGL backend
pip install -r requirements/matgl.txt

# MatGL models that require the legacy DGL backend
pip install -r requirements/matgl-dgl.txt
```

Use the smallest extra file needed for your experiment:

```text
requirements/cgcnn.txt      PyTorch for scripts/models/train_cgcnn.py
requirements/matgl.txt      MatGL for scripts/models/train_matgl.py
requirements/matgl-dgl.txt  MatGL plus legacy DGL backend dependencies
```

PyTorch and graph-library wheels depend on CPU/CUDA/ROCm setup. If using a GPU, install PyTorch with the official command for the target CUDA/ROCm version first, then install the remaining model-specific requirements.

`make` targets do not install dependencies. They only run project scripts.

## Reproduce the Skeleton

Run these commands from the repository root:

```bash
make data
make smoke
```

Equivalent explicit commands:

```bash
python scripts/01_load_dataset.py
python scripts/02_make_splits.py
python scripts/03_smoke_mean_baseline.py
python scripts/99_collect_results.py
```

Expected key outputs:

```text
data/processed/dataset.pkl
data/processed/targets.csv
data/processed/structures.json.gz
data/processed/dataset_info.json
data/splits/random_iid/train.csv
data/splits/random_iid/val.csv
data/splits/random_iid/test.csv
data/splits/element_set/train.csv
data/splits/element_set/val.csv
data/splits/element_set/test.csv
data/splits/target_tails/target_bins.csv
data/splits/target_tails/tail_thresholds.json
data/splits/split_diagnostics.csv
results/raw/mean_baseline.csv
results/summary.csv
```

You can validate result files before rebuilding the summary:

```bash
make validate
```

## Data Contract

The canonical dataset file is:

```text
data/processed/dataset.pkl
```

It is a pandas DataFrame with columns:

```text
sample_id
structure
target
```

`sample_id` is an integer from `0` to `N-1`, `structure` is a `pymatgen.core.Structure`, and `target` is formation energy in `eV/unit cell`.

For quick inspection, use:

```text
data/processed/targets.csv
data/processed/structures.json.gz
```

## Split Contract

Split generation is configured in `configs/project_config.json`:

```text
random_seed = 42
default_split_strategy = random_iid
train / validation / test = 80% / 10% / 10%
```

Main split files:

```text
data/splits/random_iid/train.csv
data/splits/random_iid/val.csv
data/splits/random_iid/test.csv
data/splits/element_set/train.csv
data/splits/element_set/val.csv
data/splits/element_set/test.csv
```

Each split file contains:

```text
sample_id
split
split_strategy
```

Available split strategies:

```text
random_iid   independent random split of individual samples
element_set  grouped split by sorted unique chemical element set
```

For `element_set`, a material group such as `Ba-O-Ti` appears in exactly one of train, validation, or test. The test split contains unseen element combinations, not necessarily unseen individual elements.

Target-tail labels are saved separately for evaluation:

```text
data/splits/target_tails/target_bins.csv
data/splits/target_tails/tail_thresholds.json
```

`target_bins.csv` contains:

```text
sample_id,target,target_bin_10,target_bin_5
```

`target_tail_eval` is an evaluation slice, not a deployment-realistic train/test split, because it uses the target value to define the slice.

Training sizes are treated as absolute labeling budgets rather than percentages of the benchmark. We start from a small pilot budget of 500 labeled structures and repeatedly double the number of training samples to obtain a log-spaced learning curve. The sequence stops when the next budget would be too close to the full training set; the full train set is then added as the final upper-bound benchmark. This reflects a practical materials-screening scenario where the key constraint is the number of available DFT-labeled structures.

Budget metadata is written next to each split strategy:

```text
data/splits/random_iid/budgets.json
data/splits/element_set/budgets.json
```

For the current `random_iid` train size, expected budgets are:

```text
B500
B1000
B2000
B4000
B8000
Bfull
```

Each budget is a deterministic nested subset of the original train split. Validation and test splits remain fixed for every budget.

## Result Format

Each model run should write one or more rows to a CSV file in:

```text
results/raw/
```

Required columns:

```text
model_name,model_family,budget_name,train_budget_samples,train_fraction_actual,full_train_size,budget_strategy,split_seed,model_seed,split_id,mae,rmse,r2,n_train,n_val,n_test,target_unit,predictions_path,notes
```

Allowed `model_family` values:

```text
descriptor_baseline
cgcnn
matgl
dummy
```

Main budget identifiers:

```text
B500
B1000
B2000
B4000
B8000
Bfull
```

New comparisons should use `budget_name` and sort by `train_budget_samples`.

Prediction files should be placed in `results/predictions/` with columns:

```text
sample_id,split,y_true,y_pred
```

After adding result files, rebuild the summary:

```bash
python scripts/99_collect_results.py
```

The combined table is saved to:

```text
results/summary.csv
```

## Training Scripts

Put executable training scripts in:

```text
scripts/models/
  train_template.py
  train_descriptor_baseline.py
  train_cgcnn.py
  train_matgl.py
```

Model-specific code should stay in one `scripts/models/train_*.py` file. Students do not need to split featurizers, model code, or training logic into `src/`; `src/` is only shared project infrastructure.

Start new experiments by copying:

```text
scripts/models/train_template.py
```

The template accepts `--budget` or `--budgets`, `--seed`, and `--split-strategy`, loads the selected fixed split, writes predictions, and writes one result row per budget.

Example commands:

```bash
python scripts/models/train_template.py --budget B500 --seed 42
python scripts/models/train_descriptor_baseline.py --budget B500 --seed 42
python scripts/models/train_descriptor_baseline.py --budgets B500,B2000,Bfull --seed 42
python scripts/models/train_descriptor_baseline.py --budgets all --seed 42 --split-strategy element_set
python scripts/models/train_cgcnn.py --budget B500 --seed 42 --epochs 2
python scripts/models/train_matgl.py --budget B500 --seed 42 --epochs 2
```

These scripts are starting points. CGCNN and MatGL scripts contain TODO placeholders and fallback behavior, so they still run when optional deep learning dependencies are not installed.
