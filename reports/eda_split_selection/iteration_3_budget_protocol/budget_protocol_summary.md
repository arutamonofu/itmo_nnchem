# Budget-based learning curve protocol

## 1. Motivation

Мы используем absolute annotation budgets, потому что practical materials screening обычно ограничен числом структур, для которых можно получить DFT labels, а не долей от benchmark dataset. Бюджеты 500 -> 1000 -> 2000 -> 4000 -> 8000 -> full train задают логарифмическую learning-curve сетку.

## 2. What changed compared with previous EDA

В iteration 2 low-data subsets были выражены как 2.5% / 25% от train split. Теперь основной experimental protocol заменен на fixed budgets. Sampling остается nested random-object внутри train части соответствующего split-а, seed = 42, target не используется для sampling.

## 3. Budget subset diagnostics

Все fixed budgets помещаются в full train для выбранных split strategies; детали в `budget_definitions.csv`. Для `group_element_set` full train size = `15104`, budget 500 покрывает `56` элементов, `486` element-set groups и `492` reduced-formula groups. Budget 8000 покрывает `5287` element-set groups.

Для `group_element_set` exact train/test overlap по `element_set` равен `0` для всех budgets, как и ожидается. В budget 500 low/high tail counts для `group_element_set`: low 10% = `37`, high 10% = `61`; в full train: low 10% = `1554`, high 10% = `1543`.

## 4. Stability across seeds

Для budget 500 разброс target mean по seeds: random_iid std = `0.0380`, group_element_set std = `0.0397`. Разброс nearest Jaccard mean по seeds: random_iid std = `0.0041`, group_element_set std = `0.0025`.

Budget 500 сильнее зависит от случайного выбора subset-а, чем крупные budgets; для финальных моделей желательно запускать budget subsets по нескольким seeds, особенно для 500 и 1000 labels. Если compute budget ограничен, seed 42 можно оставить как основной candidate protocol, но uncertainty по subset seed нужно явно отметить.

## 5. Baseline difficulty under fixed budgets

Baseline sanity check сохранен в `budget_baseline_difficulty.csv`: DummyRegressor(mean) и RF на starter descriptors, без tuning. RF MAE для budget 500: random_iid = `0.526`, group_element_set = `0.516`. RF MAE для full train: random_iid = `0.424`, group_element_set = `0.421`.

Это sanity check, не финальный benchmark. Он нужен, чтобы увидеть, что fixed budgets дают ожидаемую learning-curve форму и что split protocol технически готов к моделям.

## 6. Final recommendation

- Использовать absolute train budgets: 500, 1000, 2000, 4000, 8000, full train.
- Для каждого budget использовать nested subsets, чтобы learning curve была монотонной по данным.
- Основной split protocol: `random_iid` как IID baseline и `group_element_set` как main composition-aware non-IID split.
- `group_reduced_formula` оставить optional: он убирает exact formula leakage, но допускает overlap по `element_set`.
- Tail metrics считать как evaluation slices.
- Не называть `group_element_set` unseen-elements split: это unseen element-combinations split.

## 7. Files produced

- `baseline_predictions/group_element_set_dummy_mean_budget_1000.csv`
- `baseline_predictions/group_element_set_dummy_mean_budget_2000.csv`
- `baseline_predictions/group_element_set_dummy_mean_budget_4000.csv`
- `baseline_predictions/group_element_set_dummy_mean_budget_500.csv`
- `baseline_predictions/group_element_set_dummy_mean_budget_8000.csv`
- `baseline_predictions/group_element_set_dummy_mean_budget_full.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_1000.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_2000.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_4000.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_500.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_8000.csv`
- `baseline_predictions/group_element_set_rf_starter_descriptors_budget_full.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_1000.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_2000.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_4000.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_500.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_8000.csv`
- `baseline_predictions/group_reduced_formula_dummy_mean_budget_full.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_1000.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_2000.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_4000.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_500.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_8000.csv`
- `baseline_predictions/group_reduced_formula_rf_starter_descriptors_budget_full.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_1000.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_2000.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_4000.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_500.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_8000.csv`
- `baseline_predictions/random_iid_dummy_mean_budget_full.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_1000.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_2000.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_4000.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_500.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_8000.csv`
- `baseline_predictions/random_iid_rf_starter_descriptors_budget_full.csv`
- `budget_baseline_difficulty.csv`
- `budget_definitions.csv`
- `budget_stability_aggregated.csv`
- `budget_stability_raw.csv`
- `budget_subset_diagnostics.csv`
- `budget_subset_files_inventory.csv`
- `candidate_splits/group_element_set/train_budget_1000.csv`
- `candidate_splits/group_element_set/train_budget_2000.csv`
- `candidate_splits/group_element_set/train_budget_4000.csv`
- `candidate_splits/group_element_set/train_budget_500.csv`
- `candidate_splits/group_element_set/train_budget_8000.csv`
- `candidate_splits/group_element_set/train_full.csv`
- `candidate_splits/group_reduced_formula/train_budget_1000.csv`
- `candidate_splits/group_reduced_formula/train_budget_2000.csv`
- `candidate_splits/group_reduced_formula/train_budget_4000.csv`
- `candidate_splits/group_reduced_formula/train_budget_500.csv`
- `candidate_splits/group_reduced_formula/train_budget_8000.csv`
- `candidate_splits/group_reduced_formula/train_full.csv`
- `candidate_splits/random_iid/train_budget_1000.csv`
- `candidate_splits/random_iid/train_budget_2000.csv`
- `candidate_splits/random_iid/train_budget_4000.csv`
- `candidate_splits/random_iid/train_budget_500.csv`
- `candidate_splits/random_iid/train_budget_8000.csv`
- `candidate_splits/random_iid/train_full.csv`
- `fig_budget_baseline_mae.png`
- `fig_budget_coverage.png`
- `fig_budget_nearest_jaccard.png`
- `fig_budget_target_distribution.png`
- `learning_curve_protocol_table.csv`
