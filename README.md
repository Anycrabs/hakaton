# Income Prediction AI Service

## Структура
- income_service/ml — обучение, метрики, инференс
- income_service/api — FastAPI
- income_service/frontend — SPA на nginx
- income_service/data — положите исходные CSV (hackathon_income_train.csv, hackathon_income_test.csv)
- income_service/model_store — сюда сохраняется модель/фа-й важностей при обучении

## Подготовка данных
1) Поместите файлы:
   - income_service/data/hackathon_income_train.csv
   - income_service/data/hackathon_income_test.csv
   Код читает их с параметрами `sep=';'`, `encoding='cp1251'`.
2) (Опционально) проверить описание фичей: income_service/data/features_description.csv

## Запуск в Docker
В корне репозитория выполните:

```bash
docker compose -f income_service/docker-compose.yml build
```

### Обучение модели (profile train)
```bash
docker compose -f income_service/docker-compose.yml --profile train run --rm trainer
```
- модель: income_service/model_store/model.pkl
- важности: income_service/model_store/feature_importance.png

### Запуск API и фронта
```bash
docker compose -f income_service/docker-compose.yml up -d api frontend
```
- API: http://localhost:8000 (эндпоинты /health, /predict_income)
- Frontend: http://localhost:8080 (делает запросы к API)

Остановить:
```bash
docker compose -f income_service/docker-compose.yml down
```

## Локальный запуск без Docker
- Обучение: `python -m income_service.ml.train`
- API: `uvicorn income_service.api.main:app --host 0.0.0.0 --port 8000`
- Открыть frontend/index.html в браузере (API на 8000).

## Замечания
- Целевая переменная: `target`; игнорируем столбцы `id`, `sample_weight` (если есть).
- Предобработка: численные — медианный импьют, категориальные — most_frequent + OneHotEncoder (ignore unknown).
- Метрика: WMAE (ml/metrics.py).
- Модель: XGBoostRegressor (ml/features.py :: build_model).
