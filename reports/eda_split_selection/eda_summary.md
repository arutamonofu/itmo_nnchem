# EDA for split selection

## A. Factual statistics

- Dataset: `18928` objects from `matbench_perovskites`; structure column type is `pymatgen` `Structure`; target column is `target`.
- Existing split files and low-data subsets are summarized in `current_split_inventory.csv`.
- Missing values and duplicate checks are in `dataset_inventory.csv`.
- Target summary is in `target_summary.csv`; target skewness is `0.939514`.
- Target-tail counts: low 10% = `1994`, high 10% = `1921`. Full tail table is in `target_bins_summary.csv`.
- Composition metadata is in `composition_metadata.csv`.
- Unique reduced formulas: `9646`; unique element sets: `8268`.
- Singleton groups: reduced_formula = `364`, element_set = `260`.
- Current random split overlap with train/test: reduced_formula groups = `1442`, element_set groups = `1486`.
- Test objects whose group already appears in train: reduced_formula = `0.762`, element_set = `0.829`.
- Site proxy complete fraction from ideal cubic perovskite coordinates = `0.045`; details are in `site_proxy_metadata.csv` and `site_group_stats.csv` if generated.
- Candidate split diagnostics are saved in `candidate_split_diagnostics.csv`, `candidate_split_target_stats.csv`, `candidate_split_group_overlap.csv`, and `candidate_split_element_coverage.csv`.

## B. Preliminary conclusion

По EDA видно, что current random split remains useful as an IID baseline, but it may overestimate generalization to chemically new compositions: часть формульных и element-set групп одновременно встречается в train и test.

`reduced_formula` выглядит слишком строгим ключом для первого composition-aware split, если доля singleton-групп высока: такой split может стать близким к разбиению почти по отдельным объектам и давать нестабильные размеры групп. `element_set` предварительно выглядит более реалистичным первым group key, потому что он напрямую проверяет перенос на новые химические комбинации элементов и обычно формирует более крупные группы. Это требует проверки по сохраненным candidate diagnostics, особенно по размеру test и target distribution.

Target-tail разметка полезна как stress test / property extrapolation test, а не как deployment-realistic split: она использует target для формирования eval-подмножеств. Предварительно ее лучше использовать как дополнительную диагностику качества на low/high formation-energy tails.

Site-aware split требует осторожности. Простой координатный proxy для идеальной кубической ABX3 ячейки сработал только для малой доли объектов, поэтому robust site assignment быстро не восстановлен. Это эвристика, и ее не стоит использовать как основной split без ручной проверки site assignment и химической интерпретации ролей A/B/X. Fallback для основной non-IID проверки: composition-aware group split по `element_set` или, вторым вариантом, по `reduced_formula`.

Descriptor-space OOD split не был выбран как обязательный candidate: в репозитории есть descriptor-model code и сохраненные aggregate diagnostics, но нет готовой per-sample descriptor table. Чтобы не усложнять EDA и не ломать pipeline, этот split лучше делать отдельным шагом после явного сохранения descriptor matrix.
