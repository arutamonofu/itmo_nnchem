# Technical Task: Data & Results Skeleton for Perovskite Screening Project

## 1. Project context

We are building a small educational ML project: **Data-efficient screening of perovskites**.

The project compares several approaches for predicting formation energy on the `matbench_perovskites` dataset:

- descriptor-based classical ML baseline: RF / XGBoost;
- CGCNN trained from scratch;
- pretrained / fine-tuned MatGL / M3GNet / MEGNet.

The current milestone is not to build the final model, but to prepare a reliable technical skeleton:

- dataset is loaded;
- train / validation / test splits are fixed;
- low-data subsets are created;
- all participants use the same data splits;
- results from different models can be saved and compared in one table.

The project plan uses `matbench_perovskites`, low-data regimes `2.5%`, `25%`, `100%`, and metrics MAE, RMSE, R². :contentReference[oaicite:0]{index=0}

---

## 2. Main goal

Implement a minimal reproducible data and results pipeline for the project.

The output should allow other team members to do this:

1. Load the same dataset.
2. Use the same train / validation / test split.
3. Use the same low-data train subsets.
4. Train their models independently.
5. Save results in the same format.
6. Collect all model results into one summary table.

Do not over-engineer. This is an educational project with a short deadline.

---

## 3. What should NOT be implemented

Do not implement:

- MLflow;
- DVC;
- databases;
- Docker;
- Hydra;
- complex config systems;
- web app;
- Streamlit dashboard;
- full CGCNN training;
- full MatGL fine-tuning;
- RF / XGBoost baseline training, unless needed only as a trivial smoke test.

The focus is data preparation, split consistency, and result format.

---

## 4. Expected repository structure

Create the following structure:

```text
project/
  README.md
  requirements.txt

  configs/
    project_config.json

  data/
    raw/
    processed/
    splits/

  results/
    raw/
    predictions/
    summary.csv

  scripts/
    01_load_dataset.py
    02_make_splits.py
    03_smoke_mean_baseline.py
    99_collect_results.py

  src/
    data_io.py
    metrics.py
    result_schema.py
    validation.py
````

The exact internal code structure can be adjusted, but the public files and folders above should exist.

---

## 5. Dependencies

Create `requirements.txt`.

Minimal dependencies:

```text
numpy
pandas
scipy
scikit-learn
matminer
pymatgen
tqdm
```

Do not add deep learning dependencies here. CGCNN / MatGL team members can add their own environment requirements later.

---

## 6. Config file

Create `configs/project_config.json`.

Content:

```json
{
  "dataset_name": "matbench_perovskites",
  "random_seed": 42,
  "test_size": 0.10,
  "val_size": 0.10,
  "train_fractions": [0.025, 0.25, 1.0],
  "target_unit": "eV/unit cell",
  "split_id": "seed_42_80_10_10"
}
```

Important:

* `test_size = 0.10`
* `val_size = 0.10`
* remaining data goes to train
* test set must stay fixed for all experiments

---

## 7. Dataset loading

Implement:

```text
scripts/01_load_dataset.py
```

The script should:

1. Load `matbench_perovskites`.
2. Normalize the dataframe to contain at least:

   * `sample_id`
   * `structure`
   * `target`
3. Save the dataset in formats convenient for different team members.

Expected outputs:

```text
data/raw/matbench_perovskites.pkl
data/processed/dataset.pkl
data/processed/targets.csv
data/processed/structures.json.gz
data/processed/dataset_info.json
```

### 7.1 `dataset.pkl`

A pandas DataFrame with columns:

```text
sample_id
structure
target
```

Where:

* `sample_id` is an integer from `0` to `N-1`;
* `structure` is a `pymatgen.core.Structure` object;
* `target` is formation energy.

### 7.2 `targets.csv`

CSV with:

```text
sample_id,target
```

This file is for quick inspection and debugging.

### 7.3 `structures.json.gz`

JSONL gzip file. One line per sample:

```json
{
  "sample_id": 0,
  "structure": { "...": "pymatgen Structure.as_dict()" },
  "target": -1.234
}
```

This is useful because pickle is Python-specific, while JSON is more transparent.

### 7.4 `dataset_info.json`

Should contain:

```json
{
  "dataset_name": "matbench_perovskites",
  "n_samples": 18928,
  "columns": ["sample_id", "structure", "target"],
  "target_name": "target",
  "target_unit": "eV/unit cell"
}
```

The exact `n_samples` should be taken from the loaded dataset, not hardcoded.

---

## 8. Split creation

Implement:

```text
scripts/02_make_splits.py
```

The script should read:

```text
data/processed/dataset.pkl
```

and create fixed splits.

Use:

```text
random_seed = 42
train / val / test = 80% / 10% / 10%
```

The split should be random but reproducible.

Expected outputs:

```text
data/splits/train_indices.csv
data/splits/val_indices.csv
data/splits/test_indices.csv

data/splits/train_2_5_indices.csv
data/splits/train_25_indices.csv
data/splits/train_100_indices.csv

data/splits/split_assignments.csv
data/splits/split_info.json
```

### 8.1 Main split files

Each file should contain one column:

```text
sample_id
```

Example:

```csv
sample_id
12
45
391
```

### 8.2 Low-data subsets

Low-data subsets must be sampled only from the training pool.

Use one shuffled train list and make nested subsets:

```text
train_2_5 ⊂ train_25 ⊂ train_100
```

Interpret fractions as fractions of the full dataset size, but sample only from the train pool:

```python
n_2_5 = round(0.025 * N_total)
n_25 = round(0.25 * N_total)
n_100 = len(train)
```

This gives approximately:

```text
2.5% ≈ 500 samples
25% ≈ 4700 samples
100% = full train split
```

### 8.3 `split_assignments.csv`

Create a single convenience file:

```csv
sample_id,split,in_train_2_5,in_train_25,in_train_100
0,train,false,true,true
1,test,false,false,false
2,val,false,false,false
```

### 8.4 `split_info.json`

Should contain:

```json
{
  "split_id": "seed_42_80_10_10",
  "random_seed": 42,
  "n_total": 18928,
  "n_train": 15142,
  "n_val": 1893,
  "n_test": 1893,
  "n_train_2_5": 473,
  "n_train_25": 4732,
  "n_train_100": 15142
}
```

Exact values should be computed dynamically.

---

## 9. Validation checks

Implement validation utilities in:

```text
src/validation.py
```

The split script should automatically check:

1. No overlap between train, val, test.
2. Every sample belongs to exactly one of train, val, test.
3. `train_2_5` is a subset of `train`.
4. `train_25` is a subset of `train`.
5. `train_100` equals full train.
6. `train_2_5` is a subset of `train_25`.
7. `train_25` is a subset of `train_100`.
8. There are no duplicated `sample_id`.
9. Dataset size in split files equals dataset size.

If any check fails, raise a clear error.

---

## 10. Metrics

Implement:

```text
src/metrics.py
```

Function:

```python
def compute_regression_metrics(y_true, y_pred) -> dict:
    ...
```

Return:

```python
{
    "mae": ...,
    "rmse": ...,
    "r2": ...
}
```

Use scikit-learn metrics.

MAE is the primary metric.

---

## 11. Result format

All team members should save one result row per model run.

Create:

```text
src/result_schema.py
```

Define the required columns:

```text
model
model_family
train_fraction
seed
split_id
mae
rmse
r2
n_train
n_val
n_test
target_unit
predictions_path
notes
```

### 11.1 Meaning of columns

```text
model
```

Specific model name, for example:

```text
rf
xgboost
cgcnn
matgl_m3gnet
mean_baseline
```

```text
model_family
```

One of:

```text
descriptor_baseline
cgcnn
matgl
dummy
```

```text
train_fraction
```

One of:

```text
0.025
0.25
1.0
```

```text
seed
```

Random seed used in the model run.

```text
split_id
```

Use:

```text
seed_42_80_10_10
```

```text
mae, rmse, r2
```

Metrics on the fixed test set.

```text
n_train, n_val, n_test
```

Number of samples used.

```text
target_unit
```

Use:

```text
eV/unit cell
```

```text
predictions_path
```

Relative path to prediction file.

```text
notes
```

Free text, can be empty.

---

## 12. Prediction format

Each model should optionally save predictions to:

```text
results/predictions/
```

Prediction file format:

```csv
sample_id,split,y_true,y_pred
1,test,-1.23,-1.18
2,test,-0.87,-0.91
```

For now, predictions are mainly needed for future parity plots and debugging.

---

## 13. Smoke-test baseline

Implement:

```text
scripts/03_smoke_mean_baseline.py
```

This is not the real baseline. It is only a pipeline smoke test.

The script should:

1. Load `dataset.pkl`.
2. Load splits.
3. For each train fraction:

   * `0.025`
   * `0.25`
   * `1.0`
4. Compute the mean target value on the selected train subset.
5. Predict this constant value on the test set.
6. Compute MAE, RMSE, R².
7. Save predictions.
8. Save result rows.

Expected outputs:

```text
results/raw/mean_baseline.csv
results/predictions/mean_baseline_0_025_seed42.csv
results/predictions/mean_baseline_0_25_seed42.csv
results/predictions/mean_baseline_1_0_seed42.csv
```

This script proves that:

* dataset loading works;
* split loading works;
* metrics work;
* result format works;
* result collection works.

---

## 14. Collecting results

Implement:

```text
scripts/99_collect_results.py
```

The script should:

1. Read all CSV files from:

```text
results/raw/
```

2. Validate that each file contains required columns.
3. Concatenate them.
4. Sort by:

```text
model_family
model
train_fraction
seed
```

5. Save:

```text
results/summary.csv
```

If a result file has missing columns, print a clear error with the file name.

---

## 15. README

Create `README.md` with:

1. Short project description.
2. Installation instructions.
3. Commands to reproduce the skeleton.
4. Data contract for team members.
5. Result format.
6. Example commands.

Minimal README commands:

```bash
pip install -r requirements.txt

python scripts/01_load_dataset.py
python scripts/02_make_splits.py
python scripts/03_smoke_mean_baseline.py
python scripts/99_collect_results.py
```

Also include this note:

```text
The test set is fixed and must not be changed. All models must report metrics on the same test set.
```

---

## 16. Expected final state

After running all scripts, the following files should exist:

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

---

## 17. Acceptance criteria

The task is complete if:

1. The full pipeline runs from a clean repository using the README commands.
2. Dataset is loaded successfully.
3. Splits are reproducible.
4. Low-data subsets are created correctly.
5. Validation checks pass.
6. Smoke mean baseline produces metrics.
7. `results/summary.csv` is created.
8. Other team members can clearly understand which files to use.
9. No unnecessary infrastructure is added.

---

## 18. Implementation style

Keep the code simple.

Requirements:

* use relative paths from repository root;
* do not hardcode absolute local paths;
* write clear error messages;
* prefer simple functions over classes;
* avoid notebooks as the main implementation;
* scripts should be runnable from the project root;
* keep all random operations controlled by `random_seed = 42`.

---

## 19. Suggested implementation order

Implement in this order:

1. Repository structure.
2. `requirements.txt`.
3. `configs/project_config.json`.
4. `scripts/01_load_dataset.py`.
5. `src/data_io.py`.
6. `scripts/02_make_splits.py`.
7. `src/validation.py`.
8. `src/metrics.py`.
9. `src/result_schema.py`.
10. `scripts/03_smoke_mean_baseline.py`.
11. `scripts/99_collect_results.py`.
12. `README.md`.

---

## 20. Main design principle

The goal is not to build a perfect ML platform.

The goal is to make sure that all models in the team are compared fairly:

```text
same dataset
same train / val / test split
same low-data regimes
same metrics
same result format
```
