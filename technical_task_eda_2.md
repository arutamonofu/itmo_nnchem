Продолжаем EDA для выбора split-стратегии.

Первая итерация уже выполнена и лежит в:
`reports/eda_split_selection/`

Не нужно заново делать базовую инвентаризацию датасета, target distribution и первичные candidate splits, если эти файлы уже существуют. Используй результаты первой итерации как вход.

Цель второй итерации:
Проверить, насколько устойчив и содержателен предварительный вывод:

- random split оставить как IID baseline;
- `group_element_set` рассмотреть как основной дополнительный non-IID split;
- `group_reduced_formula` оставить как более мягкий/альтернативный group split;
- target tails использовать как evaluation/stress analysis, а не как основной split;
- site-aware split не использовать без ручной проверки.

Главная задача второй итерации — не “ещё EDA ради EDA”, а проверка последствий найденных фактов:
1. устойчивость split-стратегий к seed;
2. пригодность low-data subsets внутри group split;
3. насколько test удалён от train не только по exact overlap, но и по approximate similarity;
4. как распределяются target tails по split-ам;
5. насколько group split реально сложнее для простого baseline.

Все новые результаты сохраняй в:
`reports/eda_split_selection/iteration_2/`

Не перезаписывай результаты первой итерации. Можно обновить итоговый markdown-отчёт, но старый `eda_summary.md` не удаляй. Создай новый:
`reports/eda_split_selection/iteration_2/eda_iteration_2_summary.md`

## 1. Split stability across seeds

Сгенерируй несколько split-ов для стратегий:

- `group_reduced_formula`;
- `group_element_set`.

Используй seeds:
`[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]`

Для каждого seed и каждой стратегии создай 80/10/10 split.

Для каждого split посчитай:

- train/val/test sizes;
- fraction train/val/test;
- target mean/std/min/max;
- target q05/q10/q50/q90/q95;
- skewness по train/val/test;
- число reduced_formula groups в train/val/test;
- число element_set groups в train/val/test;
- train/test overlap по reduced_formula;
- train/test overlap по element_set;
- test_objects_seen_in_train_fraction по reduced_formula;
- test_objects_seen_in_train_fraction по element_set;
- число элементов в train;
- число элементов в test;
- элементы, которые есть в test, но отсутствуют в train;
- low 10%, middle 80%, high 10% counts по train/val/test.

Сохрани:

- `split_stability_raw.csv`
- `split_stability_aggregated.csv`

В aggregated-таблице для ключевых метрик посчитай mean/std/min/max по seeds.

Отдельно сделай краткий вывод:
- стабилен ли `group_element_set`;
- бывают ли seeds, где target distribution заметно уезжает;
- нужен ли stratified group split по target bins или обычного group split достаточно.

## 2. Low-data subset diagnostics for group_element_set

Для выбранного candidate split `group_element_set` создай nested train subsets:

- 2.5% train;
- 25% train;
- 100% train.

Важно:
- 2.5% должен быть подмножеством 25%;
- 25% должен быть подмножеством 100%;
- test и validation не должны меняться.

Сделай две версии subset sampling:

A. Простая случайная выборка из train объектов.

B. Group-aware выборка:
- выбирать не отдельные объекты, а группы `element_set`;
- затем при необходимости приблизить размер к нужному числу объектов;
- сохранить предупреждение, если точный размер невозможен.

Для каждого subset посчитай:

- n_samples;
- fraction of full train;
- n_reduced_formula_groups;
- n_element_set_groups;
- n_elements_covered;
- список элементов, отсутствующих в subset, но присутствующих в full train;
- target mean/std/min/max/q10/q50/q90;
- low 10%, middle 80%, high 10% counts;
- fraction low/high tails относительно полного датасета;
- coverage относительно test:
  - сколько test reduced_formula groups имеют exact overlap с subset;
  - сколько test element_set groups имеют exact overlap с subset;
  - ожидаемо для group_element_set должно быть 0 по element_set, но проверь;
  - Jaccard similarity до ближайшего subset element_set, если возможно.

Сохрани:

- `low_data_subset_diagnostics.csv`
- candidate subset files:
  - `candidate_splits_group_element_set/train_2_5_random.csv`
  - `candidate_splits_group_element_set/train_25_random.csv`
  - `candidate_splits_group_element_set/train_100.csv`
  - `candidate_splits_group_element_set/train_2_5_groupaware.csv`
  - `candidate_splits_group_element_set/train_25_groupaware.csv`

В выводе ответь:
- какая стратегия subset sampling лучше для проекта;
- не слишком ли бедный 2.5% subset по химическому coverage;
- стоит ли для финальных экспериментов использовать random subsets или group-aware nested subsets.

## 3. Approximate similarity / nearest-train analysis

Exact overlap уже проверен в первой итерации. Теперь нужно оценить approximate similarity.

Для split-ов:

- current `random_iid`;
- candidate `group_reduced_formula`;
- candidate `group_element_set`;

для каждого test object посчитай similarity до train.

Минимальный обязательный анализ:

### 3.1 Element-set Jaccard similarity

Для каждого test element_set найди максимальную Jaccard similarity с любым train element_set.

Jaccard:
intersection(elements_test, elements_train) / union(elements_test, elements_train)

Сохрани для каждого test object:

- sample_id;
- split_strategy;
- test_element_set;
- nearest_train_element_set;
- max_jaccard_similarity;
- n_shared_elements;
- n_test_elements;
- n_train_elements_nearest;
- target.

Сохрани:
- `nearest_train_jaccard_per_sample.csv`
- `nearest_train_jaccard_summary.csv`

Summary:
- mean/std/min/max;
- q05/q10/q25/q50/q75/q90/q95;
- fraction with Jaccard = 1.0;
- fraction with Jaccard >= 0.75;
- fraction with Jaccard >= 0.5;
- fraction with Jaccard < 0.5.

Построй:
- `fig_nearest_jaccard_distribution.png`
- желательно один plot с overlay/hist для random_iid, group_reduced_formula, group_element_set.

### 3.2 Composition-vector distance

Если быстро реализуемо:
- Построй composition vector по элементам всего датасета.
- Для каждого test object найди ближайший train object по cosine distance и/or Euclidean distance.
- Сохрани:
  - `nearest_train_composition_distance_per_sample.csv`
  - `nearest_train_composition_distance_summary.csv`
  - `fig_nearest_composition_distance_distribution.png`

Не трать слишком много времени на оптимизацию. Если полный pairwise слишком тяжёлый, используй sklearn NearestNeighbors.

В выводе ответь:
- делает ли `group_element_set` test реально дальше от train, чем random;
- остаётся ли test всё ещё близким по shared elements;
- корректно ли называть этот split “OOD” или лучше “composition-aware / unseen element-combination split”.

## 4. Tail distribution by split

Используй target bins из первой итерации:

- low 10%;
- middle 80%;
- high 10%;
- также можно low 5% / high 5%.

Для split-ов:

- random_iid;
- group_reduced_formula;
- group_element_set;

посчитай по train/val/test:

- n_low_10;
- fraction_low_10;
- n_middle_80;
- fraction_middle_80;
- n_high_10;
- fraction_high_10;
- n_low_5;
- n_high_5.

Сохрани:

- `split_tail_distribution.csv`

Построй:

- `fig_tail_distribution_by_split.png`

В выводе ответь:
- сохраняет ли group_element_set достаточное число low/high-tail объектов в test;
- нет ли ситуации, где group split случайно делает test слишком простым или слишком экстремальным;
- можно ли использовать tail metrics на этих split-ах.

## 5. Quick baseline difficulty check

Это не финальное обучение моделей, а sanity check сложности split-ов.

Сделай быстро:

A. DummyRegressor:
- strategy = mean;
- train on train;
- evaluate on test.

B. Если готовые descriptor features доступны без долгой генерации:
- используй уже сохранённые descriptor features;
- обучи быстрый RandomForestRegressor или XGBoost, если XGBoost уже используется в проекте;
- без долгого hyperparameter tuning;
- одинаковые параметры для всех split-ов.

Split-ы:
- random_iid;
- group_reduced_formula;
- group_element_set.

Train sizes:
- 100%;
- если быстро, добавить 25%;
- 2.5% только если не сильно увеличивает время.

Метрики:
- MAE;
- RMSE;
- R²;
- MAE low 10%;
- MAE middle 80%;
- MAE high 10%.

Сохрани:

- `baseline_split_difficulty.csv`
- predictions, если удобно:
  - `baseline_predictions_random_iid.csv`
  - `baseline_predictions_group_reduced_formula.csv`
  - `baseline_predictions_group_element_set.csv`

В выводе ответь:
- насколько group_element_set сложнее random_iid;
- отличается ли сложность на low/high target tails;
- подтверждает ли baseline, что group split является содержательно более строгой проверкой.

Если descriptor features недоступны:
- обязательно сделай DummyRegressor;
- зафиксируй, почему descriptor baseline не был запущен;
- не трать время на переписывание большого feature pipeline.

## 6. Optional descriptor-space visualization

Если per-sample descriptor matrix уже доступна или её можно быстро получить:
- сделай PCA до 2D;
- опционально UMAP, только если установлен и не требует новых сложных зависимостей;
- построить scatter plots:
  - color by target;
  - color by random_iid split;
  - color by group_element_set split;
  - color by target bins.

Сохрани:

- `fig_pca_target.png`
- `fig_pca_random_iid.png`
- `fig_pca_group_element_set.png`
- `fig_pca_target_bins.png`
- `descriptor_space_summary.csv`

Не делай descriptor-space split как обязательный результат. Это только визуальная диагностика.

## 7. Исправить формулировку про reduced_formula

В первой итерации была предварительная формулировка, что `reduced_formula` может быть слишком строгим ключом “если singleton-групп много”.

Во второй итерации переформулируй аккуратнее:

- Не утверждай, что проблема `reduced_formula` именно в singleton groups, если по фактам их немного.
- Главная разница такая:
  - `group_reduced_formula` убирает exact formula leakage;
  - но может оставлять overlap по `element_set`;
  - `group_element_set` является более строгой проверкой unseen element combinations.

Добавь это в итоговый markdown.

## 8. Итоговый отчёт

Создай:

`reports/eda_split_selection/iteration_2/eda_iteration_2_summary.md`

Структура:

# EDA split selection — iteration 2

## 1. What was checked
Кратко: stability, low-data subsets, nearest-train similarity, tail distribution, quick baseline.

## 2. Split stability across seeds
Факты + вывод.

## 3. Low-data subset coverage
Факты + вывод.

## 4. Approximate similarity to train
Факты + вывод.

## 5. Tail distribution
Факты + вывод.

## 6. Quick baseline difficulty
Факты + вывод. Если baseline не удалось запустить — честно написать почему.

## 7. Updated split recommendation
Сделай таблицу:

| Split | Status | What it tests | Why / risks |
|---|---|---|---|

Ожидаемые статусы:
- `random_iid`: keep as IID baseline;
- `group_element_set`: recommended main non-IID split, если stability/difficulty checks не выявят проблем;
- `group_reduced_formula`: optional / softer group split;
- `target_tail_eval`: use as evaluation slice, not main split;
- `site_aware`: not recommended without manual validation;
- `descriptor_space_ood`: optional future extension.

## 8. Files produced
Список всех новых файлов.

## 9. Критерии качества

Работа выполнена, если:

- результаты первой итерации не перезаписаны;
- есть `iteration_2` директория;
- есть stability check по 10 seeds;
- есть low-data diagnostics для `group_element_set`;
- есть nearest-train Jaccard analysis;
- есть tail distribution by split;
- есть хотя бы DummyRegressor baseline difficulty;
- если доступны descriptors, есть быстрый RF/XGB baseline;
- есть итоговый markdown с фактическими числами и аккуратным выводом;
- выводы не преувеличивают: `group_element_set` — это unseen element combinations, не unseen elements.

Главный принцип:
не усложнять систему без необходимости. Вторая итерация должна дать уверенность в выборе split-а, а не превратиться в отдельный большой research project.