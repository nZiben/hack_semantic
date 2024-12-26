# api.py
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import os
from typing import List

app = FastAPI(title="LightGBM x1 Prediction API")

# =============================================================================
# Загрузка Модели и Атрибутов
# =============================================================================

MODEL_PATH = 'weights/lgb_model.pkl'
ATTR_PATH = 'data/attr.csv'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена по пути: {MODEL_PATH}")

model = joblib.load(MODEL_PATH)

if not os.path.exists(ATTR_PATH):
    raise FileNotFoundError(f"Атрибуты не найдены по пути: {ATTR_PATH}")

attr_df = pd.read_csv(ATTR_PATH)

# =============================================================================
# Предобработка Данных
# =============================================================================

def preprocess_input(data: pd.DataFrame, attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Предобработка данных:
    - Объединение с атрибутами для u и v
    - Создание новых фич
    - Обработка пропусков
    """
    # Объединение с атрибутами для u
    data = data.merge(
        attr_df.rename(columns={
            "age": "age_u",
            "city_id": "city_id_u",
            "sex": "sex_u",
            "school": "school_u",
            "university": "university_u"
        }),
        on=['ego_id', 'u'],
        how='left'
    )
    
    # Объединение с атрибутами для v
    data = data.merge(
        attr_df.rename(columns={
            "age": "age_v",
            "city_id": "city_id_v",
            "sex": "sex_v",
            "school": "school_v",
            "university": "university_v"
        }),
        on=['ego_id', 'v'],
        how='left'
    )
    
    # Создание новых фич
    data['same_sex'] = (data['sex_u'] == data['sex_v']).astype(int)
    data['same_city'] = ((data['city_id_u'] == data['city_id_v']) & (~data['city_id_u'].isna())).astype(int)
    data['same_school'] = ((data['school_u'] == data['school_v']) & (~data['school_u'].isna())).astype(int)
    data['same_university'] = ((data['university_u'] == data['university_v']) & (~data['university_u'].isna())).astype(int)
    data['age_diff'] = (data['age_u'] - data['age_v']).abs()
    
    # Удаление ненужных колонок
    data = data.drop(['sex_v', 'city_id_v', 'school_v', 'university_v'], axis=1)
    
    # Заполнение пропусков
    data = data.fillna(0)
    
    # Дополнительные фичи могут быть добавлены здесь
    return data

# =============================================================================
# Pydantic Модели
# =============================================================================

class PredictionRequest(BaseModel):
    ego_id: int
    u: int
    v: int
    t: float
    x2: float
    x3: int

class PredictionResponse(BaseModel):
    x1_prediction: float

class RMSEResponse(BaseModel):
    rmse: float

# =============================================================================
# Эндпоинты API
# =============================================================================

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Предсказывает значение x1 для одной записи.
    """
    input_data = pd.DataFrame([request.dict()])
    processed_data = preprocess_input(input_data, attr_df)
    try:
        prediction = model.predict(processed_data)
        return PredictionResponse(x1_prediction=prediction[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {e}")

@app.post("/predict_batch", response_model=RMSEResponse)
async def predict_batch(file: UploadFile = File(...)):
    """
    Принимает CSV-файл, выполняет предсказания и возвращает RMSE.
    Файл должен содержать колонки: ego_id, u, v, t, x2, x3, x1
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы поддерживаются.")
    
    try:
        # Чтение загруженного файла
        contents = await file.read()
        test_data = pd.read_csv(pd.compat.StringIO(contents.decode('utf-8')))
        
        # Проверка наличия целевой переменной
        required_columns = {'ego_id', 'u', 'v', 't', 'x2', 'x3', 'x1'}
        if not required_columns.issubset(set(test_data.columns)):
            missing = required_columns - set(test_data.columns)
            raise HTTPException(status_code=400, detail=f"Отсутствуют колонки: {missing}")
        
        # Предобработка данных
        processed_test = preprocess_input(test_data, attr_df)
        
        # Предсказания
        y_true = test_data['x1'].values
        y_pred = model.predict(processed_test)
        
        # Расчёт RMSE
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        
        return RMSEResponse(rmse=rmse)
    
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="Файл пуст.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Произошла ошибка при обработке файла: {e}")
