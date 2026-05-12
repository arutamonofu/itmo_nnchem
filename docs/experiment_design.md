# Experiment Design

## 1. Goal

The goal is to run a reduced, realistic final experiment protocol for comparing
three model families on `matbench_perovskites` formation energy prediction under
controlled data splits and training budgets.

This document describes the final protocol, not the earlier large 216-run design.
The final matrix is:

```text
3 models x 2 split strategies x 4 budgets x 1 seed = 24 runs
```

The protocol is intended to be feasible before the deadline while still covering
the main comparison axes:

- descriptor-based supervised baseline;
- crystal graph neural network trained in-project;
- pretrained MatGL transfer-learning baseline;
- IID versus element-disjoint generalization;
- small, medium, large, and full-budget training regimes.

## 2. Final reduced experiment matrix

The final run matrix contains only the required production comparison:

| Axis | Values |
| --- | --- |
| Models | `descriptor_xgb`, `cgcnn`, `matgl` |
| Split strategies | `random_iid`, `element_set` |
| Budgets | `B500`, `B2000`, `B8000`, `Bfull` |
| Seeds | `42` |

Expected number of runs:

```text
3 x 2 x 4 x 1 = 24
```

Excluded from the final matrix:

- Random Forest: excluded because XGB remains the stronger representative
  descriptor-based baseline.
- Mean baseline: excluded because it is not needed for the final model
  comparison.
- `target_extreme`: excluded because it is a diagnostic stress-test split, not an
  obligatory final split before the deadline.
- Multiple seeds: excluded because the current goal is a realistic final run;
  seed stability remains future work.

## 3. Models

### `descriptor_xgb`

`descriptor_xgb` is the final descriptor-based baseline. It uses fixed
hyperparameters from `configs/experiments/descriptor_xgb.yaml` and represents the
classical feature-engineering approach in the reduced protocol.

### `cgcnn`

`cgcnn` is the in-project crystal graph neural network baseline. It uses the
configuration in `configs/experiments/cgcnn.yaml` and tests whether a graph model
trained on the available budget improves over the descriptor baseline.

### `matgl`

`matgl` is the pretrained transfer-learning baseline. The final protocol uses the
configuration in `configs/experiments/matgl.yaml`; see
`docs/matgl_protocol.md` for the model-specific MatGL protocol.

## 4. Split strategies

### `random_iid`

`random_iid` estimates standard IID generalization when train, validation, and
test samples are drawn from the same overall distribution.

### `element_set`

`element_set` estimates extrapolation to compositions with held-out element sets.
This split is harder and is the primary reduced-protocol check for chemical
generalization beyond random IID performance.

## 5. Training budgets

The final budgets are:

| Budget | Purpose |
| --- | --- |
| `B500` | Small-data regime and smoke-scale learning signal |
| `B2000` | Medium-small regime |
| `B8000` | Large but still bounded regime |
| `Bfull` | Full available training set for each split |

These budgets are intentionally sparse. They provide a practical final comparison
without attempting to reconstruct a dense learning curve.

## 6. Hyperparameter protocol

Hyperparameters are fixed before running the final matrix. The pipeline does not
perform per-budget or per-split hyperparameter tuning during the reported final
suite.

Final model configurations:

- `descriptor_xgb`: `configs/experiments/descriptor_xgb.yaml`
- `cgcnn`: `configs/experiments/cgcnn.yaml`
- `matgl`: `configs/experiments/matgl.yaml`

The same configured model settings are used across `random_iid` and
`element_set`, across all four budgets, and at seed `42`.

Model-specific training behavior:

- XGB hyperparameters are read from `model.params` in
  `configs/experiments/descriptor_xgb.yaml`; `random_state` is derived from the
  run seed.
- CGCNN hyperparameters are read from `configs/experiments/cgcnn.yaml`. CGCNN
  evaluates validation MAE after every epoch, steps the scheduler on validation
  MAE, saves the best validation checkpoint, restores it before test inference,
  and uses `early_stopping_patience` when configured.
- MatGL hyperparameters are read from `configs/experiments/matgl.yaml`. The final
  protocol uses frozen fine-tuning and config-level `early_stopping_patience`,
  which overrides the strategy default.

## 7. Execution commands

Prepare data and splits:

```bash
perovskite-screening prepare-data --config configs/data/matbench_perovskites.yaml
perovskite-screening make-splits --config configs/default.yaml
```

Optionally check that reduced-protocol inputs are ready:

```bash
PYTHONPATH=src python scripts/check_reduced_protocol_ready.py
```

Run the final reduced suite:

```bash
for SPLIT in random_iid element_set; do
  perovskite-screening run-suite \
    --config configs/experiments/descriptor_xgb.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42

  perovskite-screening run-suite \
    --config configs/experiments/cgcnn.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42

  perovskite-screening run-suite \
    --config configs/experiments/matgl.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42
done
```

Collect and validate results:

```bash
perovskite-screening collect \
  --runs-dir outputs/runs \
  --out outputs/summary/results.csv

perovskite-screening validate-results \
  --runs-dir outputs/runs
```

Equivalent project wrapper:

```bash
make final-reduced
```

## 8. Result schema

Each run must produce a result row that can be collected into:

```text
outputs/summary/results.csv
```

The result schema must preserve the fields needed to identify and compare each
run:

- model or experiment config;
- split strategy;
- budget;
- seed;
- validation metrics;
- test metrics;
- paths to predictions and run artifacts where available.
- `model_params_json`, a valid JSON object with the effective model parameters
  used for the run.

The collected rows must pass:

```bash
perovskite-screening validate-results --runs-dir outputs/runs
```

## 9. Analysis plan

Primary comparisons:

- Compare `descriptor_xgb`, `cgcnn`, and `matgl` within each split and budget.
- Compare `random_iid` versus `element_set` for the same model and budget to
  quantify the generalization gap.
- Compare `B500`, `B2000`, `B8000`, and `Bfull` within each model and split to
  estimate budget sensitivity.

Recommended summary tables:

- best model per split and budget;
- metric delta from `random_iid` to `element_set`;
- metric delta from `B500` to `Bfull`;
- final full-budget ranking on `random_iid` and `element_set`.

When train and inference time are logged, include compute-aware comparisons such
as metric improvement per training hour or metric versus inference latency.

## 10. Acceptance criteria

The final protocol is complete when:

- exactly 24 expected runs are present for the matrix
  `3 models x 2 splits x 4 budgets x 1 seed`;
- all runs use seed `42`;
- all rows are collected into `outputs/summary/results.csv`;
- `perovskite-screening validate-results --runs-dir outputs/runs` passes;
- every result row includes enough metadata to recover model, split, budget, and
  seed;
- the final analysis can compare model quality on both `random_iid` and
  `element_set`.

## 11. Limitations and future extensions

This reduced protocol is intentionally pragmatic and has known limitations:

- One seed does not allow seed stability to be estimated. Multi-seed evaluation
  should be future work.
- The reduced budgets do not provide a full learning curve. They only sample four
  practical training regimes.
- `target_extreme` is not included in the final run. It remains useful as a
  diagnostic stress-test split after the final comparison is complete.
- Compute trade-off can be quantified only if train and inference time are logged
  consistently for every run.

Possible future extensions:

- add multiple seeds for the final matrix;
- add `target_extreme` as a diagnostic-only post-deadline split;
- add denser budgets for a full learning-curve study;
- report compute-normalized metrics once timing fields are consistently logged.
