# Integration Report

## Date

2026-05-07

## Repository status

- Data files: OK
- Splits: OK
- Result validation: OK
- Summary collection: OK

## Participant files

### Polina - descriptor baseline

Files found:
- `scripts/models/train_descriptor_baseline.py`
- `results/raw/descriptor_baseline.csv`
- `results/analysis/descriptor_diagnostics.csv`

Checks:
- script path: OK
- result CSV: OK
- predictions: partial
- notes: `0.025` RF starter smoke-test was run and produced `results/predictions/descriptor_rf_starter_0_025_seed42.csv`. Other descriptor result rows had missing prediction files, so their `predictions_path` values were cleared instead of pointing to absent files.

### Gleb - CGCNN

Files found:
- `scripts/models/train_cgcnn.py`
- `results/raw/cgcnn.csv`
- `results/predictions/cgcnn_optimized_*.csv`
- `results/predictions/cgcnn_scaffold_*.csv`

Checks:
- script path: OK
- result CSV: OK
- predictions: OK
- optional dependencies: missing `torch_geometric` in the current base environment
- notes: existing CGCNN prediction files use the fixed test split and have matching `y_true`. `cgcnn_scaffold` metrics were inconsistent with its prediction files and were recalculated from predictions. The `cgcnn_scaffold` `1.0` row remains a very poor result, but now honestly reflects the submitted predictions.

### Vika - MatGL

Files found:
- `scripts/models/train_matgl.py`
- `results/raw/matgl.csv`
- `requirements/matgl.txt`
- `requirements/matgl-dgl.txt`

Checks:
- script path: OK
- result CSV: OK
- predictions: missing
- optional dependencies: missing `matgl` / `lightning` in the current base environment
- notes: MatGL result rows had missing prediction files, so their `predictions_path` values were cleared. The script now accepts the smoke-test `--epochs` alias and does not fail during argument parsing when optional dependencies are unavailable.

## Summary table

`results/summary.csv`

Rows collected:
- `descriptor_baseline`: 12
- `cgcnn`: 6
- `matgl`: 3

## Remaining issues

- Most descriptor result rows and all MatGL result rows do not have local prediction files, so their metrics cannot be independently recomputed from predictions.
- CGCNN and MatGL smoke training cannot run in the current base environment because optional model dependencies are not installed.
- `cgcnn_scaffold` `train_fraction=1.0` predictions produce extremely poor metrics: MAE about `50.56`, RMSE about `63.49`, R2 about `-7013.74`.

## Recommended next actions

- Participants should regenerate or provide missing prediction CSV files locally under `results/predictions/`.
- Re-run `python scripts/98_validate_results.py` and `python scripts/99_collect_results.py` after adding any missing predictions.
- Install `requirements/cgcnn.txt` or MatGL requirements only in a model-specific environment before running deep model smoke tests.
