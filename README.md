# hack_semantic

# Предсказание метрику взаимодействия пользоватлей (x1) на основе пользовательских взаимодействий

Проект машинного обучения для предсказания значения `x1` на основе пользовательских взаимодействий с использованием модели LightGBM. Включает веб-приложение на Streamlit для ручных предсказаний и RESTful API на FastAPI для автоматизированных предсказаний.

## Структура Проекта

project/ 
├── app.py # Streamlit приложение 
├── api.py # FastAPI сервис 
├── weights/ 
│ └── lgb_model.pkl # Обученная модель LightGBM 
├── data/ 
│ └── attr.csv # Атрибуты пользователей 
├── requirements.txt # Зависимости Python 
└── README.md # Документация проекта

## Установка

1. **Клонирование Репозитория**
    ```bash
    git clone https://github.com/yourusername/user-interaction-prediction.git
    cd user-interaction-prediction
    ```

2. **Создание Виртуального Окружения**
    ```bash
    python -m venv venv
    source venv/bin/activate  # На Windows: venv\Scripts\activate
    ```

3. **Установка Зависимостей**
    ```bash
    pip install -r requirements.txt
    ```

## Использование

### Streamlit Приложение

1. **Убедитесь, что модель и данные на месте**
    - Поместите `lgb_model.pkl` в директорию `weights/`.
    - Убедитесь, что `attr.csv` находится в директории `data/`.

2. **Запуск Приложения**
    ```bash
    streamlit run app.py
    ```

3. **Доступ к Приложению**
    - Откройте браузер и перейдите по адресу, указанному в терминале (обычно `http://localhost:8501`).

### FastAPI Сервис

1. **Убедитесь, что модель и данные на месте**
    - Поместите `lgb_model.pkl` в директорию `weights/`.
    - Убедитесь, что `attr.csv` находится в директории `data/`.

2. **Запуск API**
    ```bash
    uvicorn api:app --reload
    ```

3. **Доступ к API**
    - Откройте браузер и перейдите по адресу `http://127.0.0.1:8000/docs` для интерактивной документации API.

## Зависимости

- **Основные Библиотеки**
  - pandas
  - numpy
  - scikit-learn
  - lightgbm
  - joblib

- **Веб-Фреймворки**
  - streamlit
  - fastapi
  - uvicorn

- **Другие**
  - pydantic

Все зависимости указаны в файле `requirements.txt`.

