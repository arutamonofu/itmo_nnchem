# Team Guide

Короткие правила для участников проекта. Цель: каждый training script читает одни и те же данные и splits, пишет predictions и metrics в одинаковом формате.

## 1. Как подготовить данные

Один раз создайте и активируйте базовое conda-окружение:

```bash
conda env create -f environment.yml
conda activate itmo_nnchem
```

Базового окружения достаточно для загрузки данных, split generation, mean baseline, descriptor baseline, validation и summary.

Затем из корня репозитория подготовьте данные:

```bash
make data
```

Если `make` недоступен, эквивалентные команды:

```bash
python scripts/01_load_dataset.py
python scripts/02_make_splits.py
```

## 2. Зависимости для моделей

Дополнительные зависимости для конкретных моделей ставятся поверх активированного окружения:

```bash
# CGCNN scaffold и PyTorch experiments
pip install -r requirements/cgcnn.txt

# MatGL scaffold с текущим backend по умолчанию
pip install -r requirements/matgl.txt

# MatGL-модели, которым нужен legacy DGL backend
pip install -r requirements/matgl-dgl.txt
```

Если нужен GPU, сначала поставьте PyTorch под свою CUDA/ROCm-версию по официальной инструкции PyTorch, затем ставьте файл нужной модели.

Короткое правило выбора:

```text
requirements/cgcnn.txt      для scripts/models/train_cgcnn.py
requirements/matgl.txt      для scripts/models/train_matgl.py
requirements/matgl-dgl.txt  только если MatGL-модель требует legacy DGL backend
```

## 3. Как взять train / val / test

В training scripts используйте общий helper:

```python
from src.project_data import make_train_val_test_dataframes

train_df, val_df, test_df = make_train_val_test_dataframes(train_fraction=0.025)
```

Разрешённые значения `train_fraction`: `0.025`, `0.25`, `1.0`.

## 4. Куда класть training scripts

Все запускаемые training scripts лежат в:

```text
scripts/models/
```

Код конкретной модели должен оставаться в одном файле `scripts/models/train_*.py`: featurization, обучение, prediction и сохранение результата.

`src/` уже содержит общие helper-функции проекта для данных, метрик и result format. Их можно импортировать, но участникам не нужно создавать там новые модули для своих моделей.

Новый эксперимент удобно начинать с копии:

```text
scripts/models/train_template.py
```

Готовые starter scripts:

```text
scripts/models/train_template.py
scripts/models/train_descriptor_baseline.py
scripts/models/train_cgcnn.py
scripts/models/train_matgl.py
```

Примеры запуска из корня репозитория:

```bash
python scripts/models/train_template.py --train-fraction 0.025 --seed 42
python scripts/models/train_descriptor_baseline.py --train-fraction 0.025 --seed 42
python scripts/models/train_cgcnn.py --train-fraction 0.025 --seed 42 --epochs 2
python scripts/models/train_matgl.py --train-fraction 0.025 --seed 42 --epochs 2
```

Что уже есть в starter scripts:

- `train_template.py`: минимальный mean baseline, самый удобный файл для копирования.
- `train_descriptor_baseline.py`: RandomForest на простых structure/composition features.
- `train_cgcnn.py`: scaffold для CGCNN; если `torch` не установлен, запускает mean baseline fallback.
- `train_matgl.py`: scaffold для MatGL / M3GNet / MEGNet; если `matgl` не установлен, запускает mean baseline fallback.

CGCNN и MatGL scripts пока не являются финальной реализацией моделей. В них есть TODO-блоки: заменяйте fallback/MLP на настоящий graph pipeline или pretrained MatGL fine-tuning, но оставляйте общий формат входов и выходов.

## 5. Куда сохранять predictions

Predictions сохраняем в:

```text
results/predictions/
```

Обязательные колонки:

```text
sample_id,split,y_true,y_pred
```

Prediction-файлы не коммитим.

## 6. Куда сохранять metrics

Result CSV каждого эксперимента сохраняем в:

```text
results/raw/
```

Обязательные поля, которые заполняет участник или template:

```text
model,model_family,train_fraction,seed,mae,rmse,r2,predictions_path,notes
```

Поля `split_id`, `n_train`, `n_val`, `n_test`, `target_unit` добавляйте через `src.result_schema.make_result_row()`, чтобы не считать их вручную.

## 7. Как обновить summary.csv

Перед сборкой summary проверьте result-файлы:

```bash
make validate
make summary
```

Или напрямую:

```bash
python scripts/98_validate_results.py
python scripts/99_collect_results.py
```

## 8. Checklist перед отправкой результата

- Training script лежит в `scripts/models/`.
- Используется фиксированный split `seed_42_80_10_10`.
- `train_fraction` равен `0.025`, `0.25` или `1.0`.
- Result CSV лежит в `results/raw/`.
- `predictions_path` указывает на существующий файл в `results/predictions/`.
- Prediction CSV содержит `sample_id,split,y_true,y_pred`.
- `python scripts/98_validate_results.py` проходит без ошибок.
- `python scripts/99_collect_results.py` обновляет `results/summary.csv`.

## Что коммитим

```text
scripts/
src/
configs/
data/splits/
results/raw/
results/summary.csv
README.md
TEAM_GUIDE.md
environment.yml
requirements.txt
requirements/
```

## Что не коммитим

```text
data/raw/
data/processed/
results/predictions/
artifacts/models/
```
