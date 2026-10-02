# Скоринговая карта розничного кредитования (German Credit)

Модель вероятности дефолта (PD) на уровне заявки и кредитная скоринговая карта, построенные
на датасете [German Credit (UCI)](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data).
По профилю заявителя система оценивает его PD и переводит её в интерпретируемую карту баллов —
такую, какая используется при ручном андеррайтинге и принятии решений кредитным специалистом,
в отличие от портфельных моделей риска.

## Результаты

| Метрика | Значение |
|---|---|
| Gini (тестовая выборка) | 0.604 |
| KS-статистика | 0.571 |
| Средний предсказанный PD | 0.305 |
| Фактическая доля дефолтов (тест) | 0.300 |
| Отобрано признаков (из 20, IV > 0.02) | 15 |

Полная методология, разбор признаков и примеры расчёта — в
[`reports/executive_summary.md`](reports/executive_summary.md).

## Структура проекта

```
.
├── src/scorecard/         # основная библиотека: загрузка данных, WOE-биннинг, модель, перевод в баллы, инференс
├── api/                   # FastAPI сервис скоринга
├── app/                   # Streamlit-интерфейс (на русском)
├── notebooks/             # scorecard.ipynb —> конечный ноутбук 
├── tests/                 # набор тестов pytest
├── artifacts/             # обученные объекты модели/биннинга/скоркарты 
├── reports/figures/       # графики ROC, калибровки, IV
├── data/raw/              # исходные файлы German Credit
└── environment.yml        # окружение conda + pip
```

## Установка

```bash
conda env create -f environment.yml
conda activate scorecard_env
```

Устанавливает все зависимости и сам пакет `scorecard`. Проверка:

```bash
pytest
```

### Данные

Скачайте [датасет German Credit](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)
и поместите `german.data` в `data/raw/german_credit/`.

## Запуск

**Ноутбуки** 

```bash
jupyter lab notebooks/
```

- `01_german_credit.ipynb` — proof-of-concept на 5 признаках (Gini 0.595) —> прототип
- `02_full_feature_scorecard.ipynb` — полный пайплайн: отбор по IV, WOE/логистическая модель,
  перевод в баллы, диагностика, сохранение артефактов черновик
- `scorecard.ipynb` —> финальный ноутбук, понятный и читаемый

**API:**

```bash
uvicorn api.main:app --reload
```

Swagger на `http://localhost:8000/docs`. Пример `POST` запроса к `api`:

```bash
curl -X POST http://localhost:8000/credit-scorecard/score \
  -H 'Content-Type: application/json' \
  -d '{
    "checking_status": "A14", "credit_history": "A34", "purpose": "A43",
    "savings_status": "A65", "employment": "A75", "personal_status_sex": "A93",
    "other_parties": "A101", "property_magnitude": "A121",
    "other_payment_plans": "A143", "housing": "A152", "foreign_worker": "A201",
    "duration": 6, "credit_amount": 1000, "installment_rate": 1, "age": 50
  }'
```

**Streamlit-интерфейс** (требует запущенный API):

```bash
streamlit run app/streamlit_app.py
```

## Методология

1. **EDA и очистка** — загрузка, валидация, проверка распределений и пропусков
2. **WOE-биннинг и отбор по IV** — редкие категории объединяются, непрерывные признаки
   биннятся по квантилям с расширенными внешними границами (обрабатывает значения вне
   диапазона при инференсе); признаки с IV < 0.02 отбрасываются
3. **Логистическая регрессия** на WOE-преобразованных признаках
4. **Перевод в баллы скоркарты** — масштабирование Offset=600, PDO=20
5. **Оценка качества** — Gini, KS, калибровка, проверка корреляции признаков
6. **API и streamlit** — обученные артефакты сохраняются, скоринг через FastAPI + Streamlit

Ключевые формулы:

$$p = \frac{1}{1+e^{-(\beta_0+\sum_i \beta_i x_i)}} \qquad WOE_b = \ln\frac{\%good_b}{\%bad_b} \qquad IV = \sum_b (\%good_b-\%bad_b)\cdot WOE_b$$

$$\text{Score} = \text{Offset} - \text{Factor}\cdot\ln(\text{odds}), \quad \text{Factor} = \frac{PDO}{\ln 2} \qquad \text{Gini} = 2\cdot AUC - 1$$

## Тестирование и инструменты

```bash
pytest -v              # полный набор тестов
ruff check .            # линтинг
ruff format --check .   # форматирование
```

`nbstripout` установлен для очистки вывода ноутбуков при коммите.
