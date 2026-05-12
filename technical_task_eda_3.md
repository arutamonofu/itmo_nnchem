Продолжаем EDA / experimental protocol alignment после второй итерации.

Важное изменение постановки:
Мы больше не рассматриваем train sizes как проценты от конкретного benchmark split-а. Теперь low-data / learning-curve режимы задаются как абсолютные annotation budgets:

- 500 labeled structures
- 1000 labeled structures
- 2000 labeled structures
- 4000 labeled structures
- 8000 labeled structures
- full train set

Смысл:
Это имитирует practical materials screening scenario, где ограничение — не доля от уже существующего датасета, а абсолютное число структур, для которых удалось получить DFT labels.

Бюджеты выбраны как последовательное удвоение. Это логарифмическая learning-curve сетка, которая должна показать:
- как быстро модели улучшаются при росте количества labels;
- на каком бюджете graph-based approaches начинают выигрывать;
- где дополнительная разметка перестаёт давать существенный прирост качества.

Цель этой итерации:
Не переделывать весь EDA, а привести split / subset diagnostics к новой budget-based постановке.

Новые результаты сохраняй в:
`reports/eda_split_selection/iteration_3_budget_protocol/`

Не перезаписывай результаты iteration_1 и iteration_2.

Создай итоговый отчёт:
`reports/eda_split_selection/iteration_3_budget_protocol/budget_protocol_summary.md`

## 1. Зафиксировать новые бюджеты

Используй budgets:

```python
BUDGETS = [500, 1000, 2000, 4000, 8000, "full"]
````

Для каждого split strategy:

* `random_iid`
* `group_element_set`
* optional: `group_reduced_formula`

определи full train size.

Для `"full"` используй весь train соответствующего split-а:

* для `random_iid` — текущий full train;
* для `group_element_set` — full train из candidate split;
* для `group_reduced_formula` — full train из candidate split, если используешь.

Проверь, что все fixed budgets меньше или равны full train size. Если какой-то budget больше full train, пропусти его и добавь warning.

Сохрани:

* `budget_definitions.csv`

Поля:

* split_strategy
* budget_label
* requested_budget
* actual_n_train
* full_train_size
* is_full_train
* warning

## 2. Создать nested budget subsets

Для каждого split strategy создай nested train subsets:

* 500 входит в 1000;
* 1000 входит в 2000;
* 2000 входит в 4000;
* 4000 входит в 8000;
* 8000 входит в full train.

Основной protocol:

* random object sampling внутри train части соответствующего split-а;
* fixed seed = 42 для основного candidate protocol;
* не использовать target для sampling, чтобы не делать искусственно stratified выборку.

Сохрани candidate files:

Для `random_iid`:

* `candidate_splits/random_iid/train_budget_500.csv`
* `candidate_splits/random_iid/train_budget_1000.csv`
* `candidate_splits/random_iid/train_budget_2000.csv`
* `candidate_splits/random_iid/train_budget_4000.csv`
* `candidate_splits/random_iid/train_budget_8000.csv`
* `candidate_splits/random_iid/train_full.csv`

Для `group_element_set`:

* `candidate_splits/group_element_set/train_budget_500.csv`
* `candidate_splits/group_element_set/train_budget_1000.csv`
* `candidate_splits/group_element_set/train_budget_2000.csv`
* `candidate_splits/group_element_set/train_budget_4000.csv`
* `candidate_splits/group_element_set/train_budget_8000.csv`
* `candidate_splits/group_element_set/train_full.csv`

Опционально так же для `group_reduced_formula`.

Сохрани:

* `budget_subset_files_inventory.csv`

## 3. Budget subset diagnostics

Для каждого split strategy и каждого budget посчитай:

Basic:

* n_train
* n_val
* n_test
* budget_label
* is_full_train

Composition coverage:

* n_reduced_formula_groups in train subset
* n_element_set_groups in train subset
* n_elements_covered
* elements_absent_from_budget_but_present_in_full_train
* fraction_of_full_train_reduced_formula_groups_covered
* fraction_of_full_train_element_set_groups_covered

Target distribution:

* target mean/std/min/max
* q05/q10/q25/q50/q75/q90/q95
* n_low_10
* fraction_low_10
* n_middle_80
* fraction_middle_80
* n_high_10
* fraction_high_10
* n_low_5
* n_high_5

Train-test relation:

* train/test exact overlap by reduced_formula
* train/test exact overlap by element_set
* test_objects_seen_in_train_fraction by reduced_formula
* test_objects_seen_in_train_fraction by element_set
* nearest test-to-train element-set Jaccard mean/q10/q50/q90
* fraction of test objects with nearest Jaccard = 1.0
* fraction of test objects with nearest Jaccard >= 0.75
* fraction of test objects with nearest Jaccard < 0.5

Сохрани:

* `budget_subset_diagnostics.csv`

Важно:
Для `group_element_set` exact test overlap by element_set должен оставаться 0 для всех budgets, потому что test groups не пересекаются с full train groups. Проверь это явно.

## 4. Budget stability across subset seeds

Для основного split-а `group_element_set` и для `random_iid` проверь устойчивость budget subsets к seed.

Используй seeds:
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

Для каждого seed создай nested random-object budget subsets:

* 500
* 1000
* 2000
* 4000
* 8000

Для каждого seed/budget/split_strategy посчитай:

* target mean/std/q10/q50/q90;
* n_elements_covered;
* n_reduced_formula_groups;
* n_element_set_groups;
* n_low_10;
* n_high_10;
* nearest test-to-train Jaccard mean;
* nearest test-to-train Jaccard q10;
* nearest test-to-train Jaccard q50;
* nearest test-to-train Jaccard q90.

Сохрани:

* `budget_stability_raw.csv`
* `budget_stability_aggregated.csv`

В aggregated посчитай mean/std/min/max по seeds.

Цель:
Понять, насколько сильно бюджет 500 зависит от случайного выбора subset-а. Если разброс большой, зафиксировать рекомендацию запускать финальные эксперименты по нескольким seeds для budget subsets.

## 5. Обновить quick baseline difficulty под новые budgets

Если это быстро и не ломает pipeline, обнови baseline sanity check для новых budgets.

Models:

* DummyRegressor(mean)
* RF starter descriptors через существующий featurize/helper
* optional: XGBoost starter/expanded, только если это быстро и уже стабильно работает

Split strategies:

* `random_iid`
* `group_element_set`
* optional: `group_reduced_formula`

Budgets:

* 500
* 1000
* 2000
* 4000
* 8000
* full

Metrics:

* MAE
* RMSE
* R²
* MAE low 10%
* MAE middle 80%
* MAE high 10%

Сохрани:

* `budget_baseline_difficulty.csv`
* predictions optional:

  * `baseline_predictions/{split_strategy}_{model}_budget_{budget}.csv`

Важно:
Это не финальный model benchmark, а sanity check. Не трать много времени на tuning. Одинаковые параметры для всех budgets и splits.

## 6. Подготовить learning-curve ready table

Создай таблицу, которую потом можно будет использовать для графика learning curves.

Сохрани:

* `learning_curve_protocol_table.csv`

Поля:

* split_strategy
* budget_label
* n_train
* n_val
* n_test
* budget_type: fixed_budget/full_train
* model_family_placeholder
* metric_placeholder
* notes

Пока model_family_placeholder можно оставить пустым или заполнить значениями:

* descriptor
* cgcnn
* matgl

Цель:
Чтобы финальные результаты моделей можно было прямо join-ить с этой таблицей.

## 7. Графики

Построй и сохрани:

1. `fig_budget_target_distribution.png`

   * target distribution по budgets для `group_element_set`.

2. `fig_budget_coverage.png`

   * x-axis: budget
   * y-axis: n_element_set_groups / n_reduced_formula_groups / n_elements_covered.

3. `fig_budget_nearest_jaccard.png`

   * x-axis: budget
   * y-axis: nearest test-to-train Jaccard mean/q10/q90.

4. Если baseline запущен:

   * `fig_budget_baseline_mae.png`
   * MAE vs budget для random_iid и group_element_set.

Используй понятные подписи. Не перегружай графики.

## 8. Обновить формулировки в отчёте

Создай markdown:

`budget_protocol_summary.md`

Структура:

# Budget-based learning curve protocol

## 1. Motivation

Объясни кратко:
Мы используем absolute annotation budgets, потому что практический сценарий materials screening ограничен числом DFT labels, а не долей от benchmark dataset. Бюджеты 500 → 1000 → 2000 → 4000 → 8000 → full train формируют логарифмическую learning-curve сетку.

## 2. What changed compared with previous EDA

Кратко:
В предыдущей итерации 2.5% / 25% считались как доли от train split. Теперь они заменены на fixed budgets.

## 3. Budget subset diagnostics

Факты:

* coverage по элементам;
* coverage по groups;
* target distribution;
* tail counts.

## 4. Stability across seeds

Вывод:

* насколько стабильны budget subsets;
* достаточно ли одного seed;
* нужно ли запускать финальные модели на нескольких subset seeds.

## 5. Baseline difficulty under fixed budgets

Если baseline запущен:

* кратко сравнить random_iid и group_element_set;
* не делать финальный ranking моделей.

Если baseline не запущен:

* честно написать почему.

## 6. Final recommendation

Ожидаемая рекомендация:

* Использовать absolute train budgets: 500, 1000, 2000, 4000, 8000, full train.
* Для каждого budget использовать nested subsets.
* Основной split protocol:

  * `random_iid` as IID baseline;
  * `group_element_set` as main composition-aware non-IID split.
* `group_reduced_formula` optional.
* Tail metrics считать как evaluation slices.
* Не называть `group_element_set` unseen-elements split; это unseen element-combinations split.

## 7. Files produced

Список файлов.

## 9. Критерии качества

Работа выполнена, если:

* есть fixed budget subsets для random_iid и group_element_set;
* budgets nested;
* есть diagnostics по coverage / target / tails / nearest similarity;
* есть stability across seeds;
* есть markdown summary;
* старые iteration_1 и iteration_2 results не перезаписаны;
* больше не используются формулировки “2.5% / 25% train” как основной experimental protocol;
* в отчёте явно объяснено, почему fixed annotation budgets лучше соответствуют practical materials screening.

Главный принцип:
Не расширять EDA бесконечно. Эта итерация нужна только для выравнивания EDA и experimental design с новой budget-based постановкой.