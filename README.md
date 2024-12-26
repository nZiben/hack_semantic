# Предсказание x1 на основе пользовательских взаимодействий

Проект машинного обучения для предсказания значения `x1` на основе пользовательских взаимодействий с использованием модели LightGBM. Включает веб-приложение на Streamlit и RESTful API на FastAPI для предсказаний.

## Структура Проекта

- **app.py** — Streamlit приложение
- **api.py** — FastAPI сервис
- **weights/**
  - `lgb_model.pkl` — Обученная модель LightGBM
- **data/**
  - `attr.csv` — Атрибуты пользователей
- **requirements.txt** — Зависимости Python
- **README.md** — Документация проекта

## Установка

1. **Клонирование Репозитория**
    ```bash
    git clone https://github.com/yourusername/user-interaction-prediction.git
    cd user-interaction-prediction
    ```

2. **Создание Виртуального Окружения**
    ```bash
    python -m venv venv
    ```

3. **Активация Виртуального Окружения**

    - **Windows (PowerShell)**
        ```powershell
        .\venv\Scripts\Activate.ps1
        ```
    - **Windows (cmd.exe)**
        ```cmd
        venv\Scripts\activate.bat
        ```
    - **Unix/Linux/MacOS**
        ```bash
        source venv/bin/activate
        ```

4. **Установка Зависимостей**
    ```bash
    pip install -r requirements.txt
    ```

## Использование

### Streamlit Приложение

1. **Запуск Приложения**
    ```bash
    streamlit run app.py
    ```

2. **Доступ к Приложению**
    - Откройте браузер и перейдите по адресу `http://localhost:8501`

### FastAPI Сервис

1. **Запуск API**
    ```bash
    uvicorn api:app --reload
    ```

2. **Доступ к API Документации**
    - Откройте браузер и перейдите по адресу `http://127.0.0.1:8000/docs`

