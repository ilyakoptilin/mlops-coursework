# Отчёт о проверке курсового проекта

## 1. Идентификаторы сдачи

- **Репозиторий:** https://github.com/ilyakoptilin/mlops-coursework
- **Основная ветка:** `main`
- **Итоговый релизный тег:** `v1.0.0` (указывает на финальный коммит с отчётом)
- **Полный Commit SHA (ревизия кода и CI):** `e807676403737723ce9cedf4b29e4d879f1e3d7c`
- **Успешный CI Pipeline (GitHub Actions):** https://github.com/ilyakoptilin/mlops-coursework/actions (прогон `#14` на коммите `e807676403737723ce9cedf4b29e4d879f1e3d7c`)
- **Эксперимент MLflow:** `ai4i-predictive-maintenance` (зафиксированы 3 модели: `logistic_regression`, `random_forest`, `hist_gradient_boosting`)
- **Версии данных (DVC):**
  - `v1` (сырые данные): `data/raw/ai4i2020.csv.dvc`
  - `v2` (подготовленные признаки): `data/processed/ai4i2020_processed.csv.dvc`
- **Артефакт модели:** `models/model.joblib` (версия `v1.0.0`, `HistGradientBoostingClassifier`, `F1 = 0.7891`, `PR-AUC = 0.8929`; воспроизводится внутри Docker при сборке образа и через `src/mlops_coursework/train.py`)
- **Демо-видео (1–3 минуты): https://drive.google.com/file/d/1Ya6szFNAlrf54gWxLHt3R-GgX_ECIrkF/view?usp=share_link

## 2. Результаты проверки компонентов

| Компонент | Статус | Подтверждение |
|---|---|---|
| Git, ветки, PR и тег `v1.0.0` | Выполнено | Разработка велась в `feat/*`-ветках с вливанием в `main` через Pull Request; зависимости зафиксированы в `uv.lock` |
| Docker (non-root, без `:latest`) | Выполнено | Базовый образ `python:3.12-slim`, `uv:0.5.11`, автообучение модели в `docker build`, запуск от `appuser` |
| Эксперименты и данные | Выполнено | 3 сравнимых эксперимента в MLflow + 2 версии данных в DVC с контрольными суммами |
| Инференс-сервис и тесты | Выполнено | FastAPI (`/health`, `/predict`, `/docs`), 7 автотестов (`pytest`), покрытие кода `90.53%` (порог 80%) |
| Конфигурация и секреты | Выполнено | Разделение `train` / `inference` в `configs/config.yaml`, валидация при старте, шаблон `.env.example` |
| Документация | Выполнено | `README.md`, `DATASET_CARD.md`, `MODEL_CARD.md`, `REPORT.md`, схемы ML-процесса и CI |
| Доп. компонент 1: CI | Выполнено | GitHub Actions: `ruff`, `pytest-cov` (>= 80%), выгрузка XML-артефактов, сборка Docker и smoke-тест |
| Доп. компонент 2: UI | Выполнено | Клиентский веб-интерфейс на Streamlit (`src/mlops_coursework/ui.py`), обращающийся к `/predict` |

## 3. Сценарий демонстрации (1–3 минуты)

1. Сборка Docker-образа: `docker build -t ai4i-maintenance-api:v1.0.0 .`
2. Запуск контейнера: `docker run -d --name ai4i-api -p 8000:8000 ai4i-maintenance-api:v1.0.0`
3. Проверка готовности: `curl -i http://127.0.0.1:8000/health` (ожидается HTTP `200`, `"model_version": "v1.0.0"`)
4. Корректный запрос `/predict`: отправка валидного JSON с показаниями датчиков (ожидается HTTP `200`, предсказание и вероятность отказа)
5. Некорректный запрос `/predict`: отправка невалидных данных (например, `"machine_type": "UNKNOWN"`, отрицательная температура) — ожидается контролируемый ответ HTTP `422 Unprocessable Entity`
6. Проверка UI: запуск Streamlit (`uv run streamlit run src/mlops_coursework/ui.py`) и расчёт риска отказа через веб-интерфейс.
