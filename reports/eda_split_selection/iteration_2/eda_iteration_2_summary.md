# EDA split selection — iteration 2

## 1. What was checked

Проверены stability по 10 seeds, nested low-data subsets для `group_element_set`, nearest-train similarity, tail distribution и быстрый baseline difficulty check. Результаты первой итерации не перезаписывались; новые файлы сохранены в `reports/eda_split_selection/iteration_2/`.

## 2. Split stability across seeds

Для `group_element_set` test size по seeds был в диапазоне `1872`-`1945` объектов, test target mean был в диапазоне `1.431`-`1.503`. Подробные per-seed значения: `split_stability_raw.csv`, агрегаты mean/std/min/max: `split_stability_aggregated.csv`.

Предварительно `group_element_set` выглядит достаточно стабильным по размеру. Target distribution немного меняется от seed к seed, поэтому stratified group split по target bins можно рассмотреть как refinement, но базовый group split уже пригоден как simple non-IID check, если фиксировать seed и явно репортить target stats.

## 3. Low-data subset coverage

Для `group_element_set` созданы nested 2.5%, 25%, 100% train subsets. Random 2.5% subset содержит `378` объектов и `56` элементов; group-aware 2.5% subset содержит `377` объектов и `56` элементов. Exact test overlap по `element_set` остается `0` для group-aware 2.5%.

Для финальных экспериментов random object subsets проще и дают более ровный target sampling. Group-aware nested subsets лучше согласованы с идеей composition-aware training, но на 2.5% могут давать более бедное chemical coverage и неточный размер из-за дискретных групп. Практичный вариант: использовать random nested subsets как основной low-data protocol внутри выбранного split, а group-aware subsets оставить как sensitivity check.

## 4. Approximate similarity to train

Nearest element-set Jaccard mean: random_iid = `0.960`, group_reduced_formula = `0.814`, group_element_set = `0.772`. Fraction with Jaccard = 1.0: random_iid = `0.829`, group_reduced_formula = `0.198`, group_element_set = `0.000`.

Nearest composition-vector cosine distance mean: random_iid = `0.0225`, group_reduced_formula = `0.0985`, group_element_set = `0.1001`.

По EDA `group_element_set` делает test дальше от train по exact element-set overlap и approximate similarity, чем random. При этом test часто остается близким по shared elements, поэтому корректнее называть его `composition-aware / unseen element-combination split`, а не сильным OOD split по unseen elements.

## 5. Tail distribution

В `group_element_set` test: low 10% = `217` объектов (`0.113`), high 10% = `186` объектов (`0.097`). Это достаточно для tail metrics на этих split-ах. По сохраненной таблице нужно репортить tail fractions рядом с основными метриками, потому что group split может немного менять сложность test через target composition.

## 6. Quick baseline difficulty

Быстрый baseline сохранен в `baseline_split_difficulty.csv`: DummyRegressor(mean) и RF на starter descriptors из существующего pipeline, без hyperparameter tuning. RF 100% train MAE: random_iid = `0.421`, group_reduced_formula = `0.420`, group_element_set = `0.420`; delta group_element_set - random_iid = `-0.001`.

В этой sanity check таблице starter-descriptor RF не показывает заметного MAE-штрафа для `group_element_set`. Поэтому baseline сам по себе не доказывает повышенную сложность split-а; основной аргумент за `group_element_set` здесь — отсутствие exact `element_set` overlap и более низкая nearest-train similarity. Tail MAE (`mae_low_10`, `mae_middle_80`, `mae_high_10`) нужно использовать как diagnostic slices, а не как отдельный split.

## 7. Updated split recommendation

| Split | Status | What it tests | Why / risks |
|---|---|---|---|
| `random_iid` | keep as IID baseline | IID interpolation under current random split | Useful baseline; can include exact chemical leakage, so not enough alone. |
| `group_element_set` | recommended main non-IID split | Generalization to unseen element combinations | Stricter than formula split on element-set overlap; not the same as unseen elements or full OOD. |
| `group_reduced_formula` | optional / softer group split | Removal of exact formula leakage | It removes exact formula leakage, but can leave `element_set` overlap. |
| `target_tail_eval` | use as evaluation slice, not main split | Property extrapolation / stress analysis | Uses target to define slices, so not deployment-realistic split. |
| `site_aware` | not recommended without manual validation | Site-role generalization | Fast proxy was not robust enough in iteration 1. |
| `descriptor_space_ood` | optional future extension | Descriptor-space distance / clustering | Useful extension after explicitly saving descriptor matrix and choosing protocol. |

## 8. Files produced

- `baseline_predictions_group_element_set.csv`
- `baseline_predictions_group_reduced_formula.csv`
- `baseline_predictions_random_iid.csv`
- `baseline_split_difficulty.csv`
- `candidate_splits_group_element_set/train_100.csv`
- `candidate_splits_group_element_set/train_25_groupaware.csv`
- `candidate_splits_group_element_set/train_25_random.csv`
- `candidate_splits_group_element_set/train_2_5_groupaware.csv`
- `candidate_splits_group_element_set/train_2_5_random.csv`
- `descriptor_space_summary.csv`
- `fig_nearest_composition_distance_distribution.png`
- `fig_nearest_jaccard_distribution.png`
- `fig_pca_group_element_set.png`
- `fig_pca_random_iid.png`
- `fig_pca_target.png`
- `fig_pca_target_bins.png`
- `fig_tail_distribution_by_split.png`
- `low_data_subset_diagnostics.csv`
- `nearest_train_composition_distance_per_sample.csv`
- `nearest_train_composition_distance_summary.csv`
- `nearest_train_jaccard_per_sample.csv`
- `nearest_train_jaccard_summary.csv`
- `split_stability_aggregated.csv`
- `split_stability_raw.csv`
- `split_tail_distribution.csv`
