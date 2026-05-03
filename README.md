# Data-efficient screening of perovskites

This repository contains a minimal reproducible data and results skeleton for an educational ML project on formation energy prediction with `matbench_perovskites`.

The goal is to make all model comparisons use the same dataset, train / validation / test split, low-data train subsets, metrics, and result format. The test set is fixed and must not be changed. All models must report metrics on the same test set.

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
data/splits/train_indices.csv
data/splits/val_indices.csv
data/splits/test_indices.csv
data/splits/train_2_5_indices.csv
data/splits/train_25_indices.csv
data/splits/train_100_indices.csv
data/splits/split_assignments.csv
data/splits/split_info.json
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

The fixed split is configured in `configs/project_config.json`:

```text
random_seed = 42
train / validation / test = 80% / 10% / 10%
split_id = seed_42_80_10_10
```

Main split files:

```text
data/splits/train_indices.csv
data/splits/val_indices.csv
data/splits/test_indices.csv
```

Low-data train subsets:

```text
data/splits/train_2_5_indices.csv
data/splits/train_25_indices.csv
data/splits/train_100_indices.csv
```

The low-data subsets are sampled only from the training pool and are nested:

```text
train_2_5 subset train_25 subset train_100
```

## Result Format

Each model run should write one or more rows to a CSV file in:

```text
results/raw/
```

Required columns:

```text
model,model_family,train_fraction,seed,split_id,mae,rmse,r2,n_train,n_val,n_test,target_unit,predictions_path,notes
```

Allowed `model_family` values:

```text
descriptor_baseline
cgcnn
matgl
dummy
```

Allowed `train_fraction` values:

```text
0.025
0.25
1.0
```

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

The template accepts `--train-fraction` and `--seed`, loads the fixed splits, writes predictions, and writes one result row.

Example commands:

```bash
python scripts/models/train_template.py --train-fraction 0.025 --seed 42
python scripts/models/train_descriptor_baseline.py --train-fraction 0.025 --seed 42
python scripts/models/train_cgcnn.py --train-fraction 0.025 --seed 42 --epochs 2
python scripts/models/train_matgl.py --train-fraction 0.025 --seed 42 --epochs 2
```

These scripts are starting points. CGCNN and MatGL scripts contain TODO placeholders and fallback behavior, so they still run when optional deep learning dependencies are not installed.
