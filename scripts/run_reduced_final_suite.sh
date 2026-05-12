#!/usr/bin/env bash
set -euo pipefail

echo "==> preparing data"
perovskite-screening prepare-data --config configs/data/matbench_perovskites.yaml

echo "==> making splits"
perovskite-screening make-splits --config configs/default.yaml

echo "==> checking reduced final protocol inputs"
PYTHONPATH=src python scripts/check_reduced_protocol_ready.py

for SPLIT in random_iid element_set; do
  echo "==> running XGB (${SPLIT})"
  perovskite-screening run-suite \
    --config configs/experiments/descriptor_xgb.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42

  echo "==> running CGCNN (${SPLIT})"
  perovskite-screening run-suite \
    --config configs/experiments/cgcnn.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42

  echo "==> running MatGL (${SPLIT})"
  perovskite-screening run-suite \
    --config configs/experiments/matgl.yaml \
    --split-strategy "$SPLIT" \
    --budgets B500,B2000,B8000,Bfull \
    --seeds 42
done

echo "==> collecting results"
perovskite-screening collect \
  --runs-dir outputs/runs \
  --out outputs/summary/results.csv

echo "==> validating results"
perovskite-screening validate-results \
  --runs-dir outputs/runs
