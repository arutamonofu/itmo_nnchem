# Technical Task: Add model training starter scripts

## 1. Context

The repository already contains a minimal reproducible skeleton for the project **Data-efficient screening of perovskites**.

Current pipeline:

- loads `matbench_perovskites`;
- creates fixed `train / val / test` splits;
- creates low-data train subsets: `2.5%`, `25%`, `100%`;
- defines common metrics: MAE, RMSE, R²;
- collects model results into `results/summary.csv`.

Now we need to add starter training scripts for team members, so each person has a clear template showing:

- how to load the dataset;
- how to load the correct split;
- where to put training code;
- how to save predictions;
- how to save metrics in the common result format.

This task should not implement full production-quality models. The scripts should be simple, readable, and useful as starting points.

---

## 2. Goal

Create the following files:

```text
scripts/models/
  train_template.py
  train_descriptor_baseline.py
  train_cgcnn.py
  train_matgl.py
````

These scripts should be runnable from the repository root.

Example:

```bash
python scripts/models/train_template.py --train-fraction 0.025 --seed 42
python scripts/models/train_descriptor_baseline.py --train-fraction 0.025 --seed 42
python scripts/models/train_cgcnn.py --train-fraction 0.025 --seed 42 --epochs 2
python scripts/models/train_matgl.py --train-fraction 0.025 --seed 42 --epochs 2
```

---

## 3. Important design principle

Keep everything simple.

The purpose of these scripts is to help team members start quickly, not to create final model implementations.

Priorities:

1. Clear structure.
2. Minimal dependencies.
3. Same input / output format for all scripts.
4. Helpful TODO comments.
5. No over-engineering.

---

## 4. Existing repository contracts

Use the existing files:

```text
data/processed/dataset.pkl
data/splits/train_2_5_indices.csv
data/splits/train_25_indices.csv
data/splits/train_100_indices.csv
data/splits/val_indices.csv
data/splits/test_indices.csv
```

The dataset is a pandas DataFrame with columns:

```text
sample_id
structure
target
```

The result files must be written to:

```text
results/raw/
```

Prediction files must be written to:

```text
results/predictions/
```

Required result columns:

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

Prediction columns:

```text
sample_id
split
y_true
y_pred
```

Use:

```text
split_id = seed_42_80_10_10
target_unit = eV/unit cell
```

---

## 5. Shared behavior for all scripts

All four scripts should:

1. Parse CLI arguments.
2. Load `data/processed/dataset.pkl`.
3. Load train / val / test splits.
4. Train or simulate a model.
5. Predict on the fixed test set.
6. Compute MAE, RMSE, R².
7. Save predictions to `results/predictions/`.
8. Save one result row to `results/raw/`.
9. Print a short summary to console.
10. Create output directories automatically if they do not exist.

---

## 6. CLI arguments

All scripts should support:

```text
--train-fraction
--seed
```

Allowed values for `--train-fraction`:

```text
0.025
0.25
1.0
```

Default:

```text
--seed 42
```

Model-specific optional arguments:

```text
--epochs
--batch-size
--lr
--device
```

Use these only where relevant.

---

## 7. Helper logic inside each script

To keep scripts self-contained, each file may include small helper functions directly.

Each script should contain helper functions similar to:

```python
def get_train_split_path(train_fraction: float) -> str:
    ...

def load_ids(path: str) -> list[int]:
    ...

def load_data(train_fraction: float):
    ...

def save_predictions(...):
    ...

def save_result_row(...):
    ...
```

Do not require new complex modules unless very simple.

If the repository already has `src.metrics.compute_regression_metrics`, use it. If not, implement metrics locally using scikit-learn.

---

## 8. `train_template.py`

### Purpose

Generic minimal template for any future model.

This script should be the cleanest and easiest to copy.

### Behavior

Use a dummy mean baseline:

1. Compute mean target value on selected train subset.
2. Predict this constant value on the test set.
3. Save predictions and result row.

### Output files

For `--train-fraction 0.025 --seed 42`:

```text
results/predictions/template_0_025_seed42.csv
results/raw/template.csv
```

### Required values

```text
model = template_mean
model_family = dummy
notes = Template script: mean target baseline. Replace training block with a real model.
```

### Important TODO comments

Add comments showing where team members should replace code:

```python
# TODO: Replace this block with real model training.
# TODO: Replace this block with real model inference.
# TODO: Update model name and model_family.
```

---

## 9. `train_descriptor_baseline.py`

### Purpose

Starter script for Polina: descriptor-based classical ML baseline.

The script should provide a minimal working baseline using simple structure-derived features and scikit-learn.

Do not depend on heavy descriptor generation if it risks breaking the environment. Prefer simple robust features first.

### Minimal feature set

For each `pymatgen.Structure`, compute simple numeric features:

```text
n_sites
volume
density
num_unique_elements
mean_atomic_number
std_atomic_number
```

These features can be computed using pymatgen structure attributes and element atomic numbers.

Example:

```python
composition = structure.composition
atomic_numbers = [el.Z for el in composition.elements]
```

It is acceptable to make simple composition-level features instead of full matminer descriptors.

### Model

Use a simple scikit-learn model:

```text
RandomForestRegressor
```

Suggested defaults:

```text
n_estimators = 100
random_state = seed
n_jobs = -1
```

Optionally support `--model rf`.

Do not add XGBoost unless it is already installed. Avoid making `xgboost` mandatory.

### Behavior

1. Load train / val / test.
2. Featurize structures.
3. Train RandomForestRegressor on train subset.
4. Predict on test.
5. Compute metrics.
6. Save predictions and result row.

### Output files

For `--train-fraction 0.025 --seed 42`:

```text
results/predictions/descriptor_rf_0_025_seed42.csv
results/raw/descriptor_baseline.csv
```

### Required values

```text
model = descriptor_rf
model_family = descriptor_baseline
notes = RandomForest on simple structure/composition features.
```

### Optional validation

The script may compute validation metrics and print them, but the saved result row should contain test metrics.

---

## 10. `train_cgcnn.py`

### Purpose

Starter script for Gleb: CGCNN model.

This should not be a full CGCNN implementation if that would take too long. The goal is to provide a clear scaffold for where CGCNN-specific code should go.

### Acceptable implementation level

Implement one of these two options:

### Option A, preferred if PyTorch is installed

Create a minimal PyTorch smoke-test model that uses simple numeric features from structures, not a real crystal graph yet.

This model should:

1. Featurize each structure with simple numeric features.
2. Train a small MLP for a few epochs.
3. Save predictions and result row.

Clearly mark it as a placeholder:

```text
This is a CGCNN training scaffold. Current implementation uses simple structure features + MLP as a smoke test. Replace featurization and model with real CGCNN graph pipeline.
```

### Option B, if PyTorch is not installed

Fall back to mean baseline and print a warning:

```text
PyTorch is not installed. Running dummy mean baseline inside CGCNN scaffold.
```

Do not fail just because PyTorch is unavailable.

### CLI arguments

Support:

```text
--train-fraction
--seed
--epochs
--batch-size
--lr
--device
```

Defaults:

```text
--epochs 2
--batch-size 64
--lr 1e-3
--device cpu
```

### Output files

For `--train-fraction 0.025 --seed 42`:

```text
results/predictions/cgcnn_scaffold_0_025_seed42.csv
results/raw/cgcnn.csv
```

### Required values

```text
model = cgcnn_scaffold
model_family = cgcnn
notes = CGCNN scaffold. Replace simple feature MLP with real crystal graph pipeline.
```

### Required TODO comments

Add clear placeholders:

```python
# TODO for CGCNN:
# 1. Convert pymatgen Structure objects to crystal graphs.
# 2. Build CGCNN Dataset and DataLoader.
# 3. Replace SimpleMLP with CGCNN model.
# 4. Use validation set for early stopping or model selection.
# 5. Keep final metrics on the fixed test set.
```

---

## 11. `train_matgl.py`

### Purpose

Starter script for Vika: MatGL / M3GNet / MEGNet model.

This should provide a clear scaffold for MatGL fine-tuning or inference, but it should not require MatGL to be installed for the script to run.

### Behavior

Try to import MatGL:

```python
try:
    import matgl
except ImportError:
    matgl = None
```

If MatGL is available:

* do not implement full fine-tuning unless simple;
* print that MatGL is available;
* leave a clearly marked TODO block for loading pretrained M3GNet / MEGNet and fine-tuning.

If MatGL is not available:

* run a dummy mean baseline or simple MLP fallback;
* print a warning that this is only a scaffold.

### CLI arguments

Support:

```text
--train-fraction
--seed
--epochs
--batch-size
--lr
--device
--pretrained-model
```

Defaults:

```text
--epochs 2
--batch-size 32
--lr 1e-4
--device cpu
--pretrained-model M3GNet-MP-2021.2.8-PES
```

The exact pretrained model name may be a placeholder.

### Output files

For `--train-fraction 0.025 --seed 42`:

```text
results/predictions/matgl_scaffold_0_025_seed42.csv
results/raw/matgl.csv
```

### Required values

```text
model = matgl_scaffold
model_family = matgl
notes = MatGL scaffold. Replace fallback baseline with pretrained MatGL inference/fine-tuning.
```

### Required TODO comments

Add clear placeholders:

```python
# TODO for MatGL:
# 1. Load pretrained MatGL / M3GNet / MEGNet model.
# 2. Convert pymatgen Structure objects into MatGL dataset format.
# 3. Run inference or fine-tuning.
# 4. Use validation set for model selection.
# 5. Keep final metrics on the fixed test set.
```

---

## 12. Naming convention for train fractions in filenames

Use safe string names:

```text
0.025 -> 0_025
0.25  -> 0_25
1.0   -> 1_0
```

Example helper:

```python
def fraction_to_name(train_fraction: float) -> str:
    return str(train_fraction).replace(".", "_")
```

---

## 13. Result writing behavior

When saving results to `results/raw/*.csv`:

* If the file does not exist, create it with header.
* If the file exists, append the new row.
* Avoid duplicate rows for the same `model`, `train_fraction`, `seed`.

Preferred behavior:

1. Read existing CSV if it exists.
2. Remove rows with the same `model`, `train_fraction`, `seed`, `split_id`.
3. Append the new row.
4. Save back.

This makes reruns clean.

---

## 14. Prediction writing behavior

When saving predictions:

* overwrite prediction file for the same model / train fraction / seed;
* include only test predictions unless validation predictions are easy to add;
* must include columns:

```text
sample_id
split
y_true
y_pred
```

Use:

```text
split = test
```

---

## 15. Console output

Each script should print something like:

```text
Loaded dataset: N samples
Train fraction: 0.025
Train size: ...
Validation size: ...
Test size: ...
Model: ...
MAE: ...
RMSE: ...
R2: ...
Saved predictions to: ...
Saved result row to: ...
```

This helps team members understand what happened.

---

## 16. Error handling

Add clear errors for common mistakes:

1. Dataset file missing:

```text
data/processed/dataset.pkl not found. Run python scripts/01_load_dataset.py first.
```

2. Split files missing:

```text
Split files not found. Run python scripts/02_make_splits.py first.
```

3. Unsupported train fraction:

```text
Unsupported train_fraction. Use one of: 0.025, 0.25, 1.0.
```

4. Missing required columns in dataset:

```text
Dataset must contain columns: sample_id, structure, target.
```

---

## 17. Dependency constraints

Do not add heavy mandatory dependencies.

Allowed dependencies:

```text
numpy
pandas
scikit-learn
pymatgen
```

Optional dependencies:

```text
torch
matgl
```

The CGCNN and MatGL scripts must still run without optional dependencies by using fallback scaffold behavior.

---

## 18. Update README or TEAM_GUIDE

Add a short section explaining that model starter scripts are available:

````md
## Model starter scripts

Starter scripts are available in:

```text
scripts/models/
````

Examples:

```bash
python scripts/models/train_template.py --train-fraction 0.025 --seed 42
python scripts/models/train_descriptor_baseline.py --train-fraction 0.025 --seed 42
python scripts/models/train_cgcnn.py --train-fraction 0.025 --seed 42 --epochs 2
python scripts/models/train_matgl.py --train-fraction 0.025 --seed 42 --epochs 2
```

These scripts are starting points. CGCNN and MatGL scripts contain placeholders and fallback behavior. Replace the scaffold parts with real model code.

````

---

## 19. Acceptance criteria

The task is complete if:

1. Folder `scripts/models/` exists.
2. The following files exist:

```text
scripts/models/train_template.py
scripts/models/train_descriptor_baseline.py
scripts/models/train_cgcnn.py
scripts/models/train_matgl.py
````

3. Each script runs from repository root.
4. Each script supports `--train-fraction` and `--seed`.
5. Each script loads the common dataset and splits.
6. Each script saves predictions to `results/predictions/`.
7. Each script saves metrics to `results/raw/`.
8. Result files follow the common schema.
9. Running `python scripts/99_collect_results.py` after these scripts includes their results in `results/summary.csv`.
10. CGCNN and MatGL scripts do not fail if `torch` or `matgl` are missing.
11. The code is simple and understandable for students.

---

## 20. Quick manual test

After implementation, run:

```bash
python scripts/01_load_dataset.py
python scripts/02_make_splits.py

python scripts/models/train_template.py --train-fraction 0.025 --seed 42
python scripts/models/train_descriptor_baseline.py --train-fraction 0.025 --seed 42
python scripts/models/train_cgcnn.py --train-fraction 0.025 --seed 42 --epochs 1
python scripts/models/train_matgl.py --train-fraction 0.025 --seed 42 --epochs 1

python scripts/99_collect_results.py
```

Expected outputs:

```text
results/raw/template.csv
results/raw/descriptor_baseline.csv
results/raw/cgcnn.csv
results/raw/matgl.csv
results/summary.csv
```

Prediction files should also appear in:

```text
results/predictions/
```

---

## 21. Final note

The goal is not to produce strong model results in this task.

The goal is to give each participant a working starting point with the correct project conventions:

```text
same dataset
same splits
same train fractions
same metrics
same result format
same output folders
```