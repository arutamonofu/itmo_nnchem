Ты работаешь в репозитории проекта Data-efficient Screening of Perovskites.

Контекст проекта:
Мы сравниваем descriptor-based models, CGCNN и MatGL/MEGNet для предсказания formation energy на датасете matbench_perovskites. Главный исследовательский вопрос — data efficiency: как модели работают при 2.5%, 25% и 100% обучающих данных. На майлстоуне уже был random train/validation/test split, но преподаватели предложили обдумать более содержательные стратегии split-ования. Нужно провести EDA, чтобы не выбирать split произвольно.

Цель задачи:
Провести EDA для выбора и обоснования split-стратегий:
1. random / IID split;
2. composition-aware или chemical group split;
3. element/site-aware split, если это реализуемо;
4. target-tail analysis: центр распределения vs хвосты formation energy;
5. descriptor-space split / OOD split, если в проекте уже есть descriptors или их можно быстро получить без чрезмерного усложнения.

Режим вывода:
Сделай два уровня результата:
A. Фактическая статистика — строго числа, таблицы, графики, без сильных интерпретаций.
B. Предварительный вывод — аккуратный текстовый вывод о том, какие split-стратегии выглядят реалистичными и почему.

Не делай финальных научных утверждений без проверки. Формулируй выводы осторожно: “по EDA видно”, “предварительно”, “требует проверки”, “не рекомендуется как основной split”.

Ограничения:
- Не ломай существующий pipeline.
- Не изменяй текущие split-файлы без явного создания копии.
- Все новые файлы складывай в отдельную папку, например:
  `reports/eda_split_selection/`
  и/или
  `notebooks/eda_split_selection/`.
- Код должен быть воспроизводимым: фиксируй random seed.
- Все графики сохраняй в `.png`.
- Все таблицы сохраняй в `.csv`.
- Краткий итог сохрани в `.md`.

Что нужно сделать:

1. Инвентаризация данных
- Найди, как в проекте загружается `matbench_perovskites`.
- Найди текущие split-файлы, если они уже существуют.
- Определи:
  - число объектов;
  - формат структуры;
  - имя target-колонки;
  - наличие пропусков;
  - наличие дубликатов структур / формул, если это можно быстро проверить;
  - текущие размеры train/validation/test;
  - размеры low-data subsets: 2.5%, 25%, 100%, если они уже есть.

Сохрани:
- `dataset_inventory.csv`
- `current_split_inventory.csv`
- короткий текстовый блок в `eda_summary.md`.

2. Target EDA для formation energy
Посчитай:
- count;
- mean;
- std;
- min;
- max;
- quantiles: 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%;
- skewness, если удобно;
- число объектов в low 5%, low 10%, middle 80%, high 10%, high 5%.

Построй:
- histogram formation energy;
- KDE или histogram с большим числом bins;
- boxplot;
- target distribution по текущим train/validation/test, если split уже есть.

Сохрани:
- `target_summary.csv`
- `target_quantiles.csv`
- `target_bins_summary.csv`
- `fig_target_histogram.png`
- `fig_target_boxplot.png`
- `fig_target_by_current_split.png`

В предварительном выводе ответь:
- есть ли выраженные хвосты;
- сколько объектов в хвостах;
- разумно ли использовать target-tail split как основной;
- или лучше использовать tail analysis как дополнительную метрику на test set.

Важно:
Target-tail split использует target для формирования test. Поэтому не называй его обычным deployment-realistic split. Называй его “stress test” или “property extrapolation test”.

3. Composition EDA
Для каждой структуры извлеки:
- formula;
- reduced_formula;
- anonymous_formula;
- sorted element set, например `"Ba-Ca-O-Ti"`;
- число уникальных элементов;
- химические элементы и их counts.

Посчитай:
- число уникальных reduced_formula;
- число уникальных element_set;
- число уникальных anonymous_formula;
- distribution размера групп для reduced_formula;
- distribution размера групп для element_set;
- top-20 most frequent reduced_formula;
- top-20 most frequent element_set;
- сколько групп имеют размер 1, 2–5, 6–20, 21–100, >100;
- target mean/std/min/max для крупных групп.

Сохрани:
- `composition_metadata.csv`
- `group_stats_reduced_formula.csv`
- `group_stats_element_set.csv`
- `group_stats_anonymous_formula.csv`
- `top_groups_reduced_formula.csv`
- `top_groups_element_set.csv`
- `fig_group_size_distribution_reduced_formula.png`
- `fig_group_size_distribution_element_set.png`

В предварительном выводе ответь:
- подходит ли reduced_formula как group key;
- подходит ли element_set как group key;
- много ли singleton-групп;
- будет ли group split сильно отличаться от random split;
- какой group key выглядит наиболее разумным для первого composition-aware split.

4. Проверка leakage в текущем random split
Если текущий random split существует, проверь overlap между train/val/test по:
- reduced_formula;
- element_set;
- anonymous_formula.

Посчитай:
- сколько reduced_formula встречаются и в train, и в test;
- сколько element_set встречаются и в train, и в test;
- долю test-объектов, чья reduced_formula уже встречалась в train;
- долю test-объектов, чей element_set уже встречался в train.

Сохрани:
- `current_split_leakage_summary.csv`

В предварительном выводе ответь:
- является ли random split in-distribution;
- есть ли признаки chemical leakage;
- насколько это критично для цели проекта.

Не пиши, что random split “плохой”. Формулируй так:
“random split remains useful as an IID baseline, but it may overestimate generalization to chemically new compositions.”

5. Element / site-aware EDA
Попробуй аккуратно понять, можно ли восстановить A/B/X роли для perovskite-like structures.

Сначала не придумывай химию вручную. Проверь:
- структура формулы похожа на ABX3 или производные;
- есть ли в данных site labels или metadata;
- можно ли устойчиво определить A/B/X по структуре, координатам, oxidation states или stoichiometry;
- насколько это сложно и рискованно.

Если robust site assignment сделать быстро нельзя:
- явно зафиксируй это в отчёте;
- предложи fallback: composition-aware group split по element_set / reduced_formula.

Если можно сделать простой site-aware proxy:
- извлеки предполагаемые A-site, B-site, X-site элементы;
- посчитай частоты элементов по site;
- посчитай частоты A/B/X combinations;
- оцени размеры групп для A-site, B-site, X-site, A-B pair, B-X pair.

Сохрани при наличии:
- `site_proxy_metadata.csv`
- `site_group_stats.csv`
- `fig_site_element_frequencies.png`

В предварительном выводе обязательно раздели:
- что удалось определить надёжно;
- что является эвристикой;
- что не стоит использовать как основной split без ручной проверки.

6. Candidate split diagnostics
Сгенерируй или хотя бы диагностически оцени несколько candidate split-стратегий.

Обязательные:
A. `random_iid`
- текущий split, если он есть;
- или новый random 80/10/10 split с seed.

B. `group_reduced_formula`
- group split, где одинаковые reduced_formula не пересекаются между train/val/test.

C. `group_element_set`
- group split, где одинаковые element_set не пересекаются между train/val/test.

D. `target_tail_eval`
- не обязательно отдельный train/test split;
- как минимум разметь объекты на low 10%, middle 80%, high 10%;
- также low 5%, middle 90%, high 5%.

Опциональные:
E. `site_aware_group`, только если site proxy достаточно надёжен.
F. `descriptor_space_ood`, только если descriptors уже есть или их можно быстро посчитать.

Для каждого candidate split посчитай:
- train/val/test sizes;
- target mean/std/min/max/quantiles по train/val/test;
- overlap по reduced_formula между train/test;
- overlap по element_set между train/test;
- element coverage: какие элементы есть в test, но отсутствуют в train;
- number of groups in train/val/test;
- largest groups in test;
- насколько размер test близок к желаемому 10%;
- предупреждения, если split получился слишком несбалансированным.

Сохрани:
- `candidate_split_diagnostics.csv`
- `candidate_split_target_stats.csv`
- `candidate_split_group_overlap.csv`
- `candidate_split_element_coverage.csv`

Не перезаписывай production split-файлы. Candidate split indices сохрани отдельно:
- `candidate_splits/random_iid.csv`
- `candidate_splits/group_reduced_formula.csv`
- `candidate_splits/group_element_set.csv`
- и т.д.

7. Descriptor-space EDA, если реализуемо
Если в проекте уже есть descriptor features для RF/XGBoost:
- загрузи их;
- стандартизируй признаки;
- удали константные и проблемные признаки;
- сделай PCA до 2D;
- опционально UMAP, если уже установлен;
- раскрась точки по formation energy;
- раскрась точки по текущему random split;
- раскрась точки по group split;
- попробуй k-means или другой простой clustering, только если это не требует сложной настройки.

Если descriptors ещё не готовы и их генерация долгая:
- не трать много времени;
- зафиксируй descriptor-space split как опциональный будущий вариант.

Сохрани:
- `fig_pca_target.png`
- `fig_pca_random_split.png`
- `fig_pca_group_split.png`
- `descriptor_space_summary.csv`

В предварительном выводе ответь:
- есть ли видимые области/кластеры;
- random split перемешивает ли всё пространство;
- group split создаёт ли более OOD-like test;
- стоит ли делать descriptor-space split сейчас или оставить как расширение.

8. Итоговый отчёт
Создай файл:

`reports/eda_split_selection/eda_split_selection_report.md`

Структура отчёта:

# EDA for Split Strategy Selection

## 1. Dataset inventory
Только факты.

## 2. Target distribution
Таблицы + ссылки на графики + краткая фактическая статистика.

## 3. Composition and group structure
Таблицы + вывод о пригодности group keys.

## 4. Current random split diagnostics
Проверка overlap/leakage.

## 5. Site-aware split feasibility
Что можно / нельзя сделать надёжно.

## 6. Candidate split diagnostics
Сравнительная таблица:
- split name;
- what it tests;
- pros;
- risks;
- feasibility;
- recommended status.

Статусы:
- keep as baseline;
- recommended;
- optional;
- not recommended as main split;
- requires manual validation.

## 7. Preliminary recommendation
Сформулируй аккуратный предварительный вывод.

Ожидаемый стиль вывода:
- Random split оставить как IID baseline.
- Composition-aware group split рассмотреть как главный дополнительный split, если group sizes и target distribution позволяют.
- Element/site-aware split использовать только если site assignment достаточно надёжен.
- Target-tail split не использовать как основной, но использовать tail-bin evaluation или stress test.
- Descriptor-space OOD split оставить как optional, если descriptors уже готовы и PCA/cluster structure выглядит осмысленно.

## 8. Files produced
Список всех созданных файлов.

9. Критерии качества
Работа считается выполненной, если:
- есть воспроизводимый код;
- есть сохранённые таблицы и графики;
- есть диагностика текущего random split;
- есть фактические статистики по target и composition groups;
- есть хотя бы два candidate split-а: random_iid и group-based;
- есть аккуратный markdown-отчёт;
- не изменены существующие production split-файлы;
- все выводы отделены от фактической статистики.

10. Важное
Не делай слишком сложную систему split-ования на первом проходе. Главная задача — получить EDA-основание для решения, а не сразу построить идеальный split.

Сначала факты, потом осторожная рекомендация.