# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_squared_error
import os

# =============================================================================
# Загрузка Модели
# =============================================================================

MODEL_PATH = 'weights/lgb_model.pkl'
ATTR_PATH = 'data/attr.csv'

@st.cache(allow_output_mutation=True)
def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"Модель не найдена по пути: {model_path}")
        return None
    model = joblib.load(model_path)
    return model

@st.cache(allow_output_mutation=True)
def load_attributes(attr_path):
    if not os.path.exists(attr_path):
        st.error(f"Атрибуты не найдены по пути: {attr_path}")
        return None
    attr_df = pd.read_csv(attr_path)
    return attr_df

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
# Основная Функция Приложения
# =============================================================================

def main():
    st.title("Предсказание x1 с помощью LightGBM")
    
    # Загрузка модели
    model = load_model(MODEL_PATH)
    if model is None:
        st.stop()
    
    # Загрузка атрибутов
    attr_df = load_attributes(ATTR_PATH)
    if attr_df is None:
        st.stop()
    
    st.header("Ввод данных для предсказания")
    
    # Поля ввода данных
    ego_id = st.number_input("ego_id", min_value=0, step=1, value=0)
    u = st.number_input("u", min_value=0, step=1, value=0)
    v = st.number_input("v", min_value=0, step=1, value=0)
    t = st.number_input("t", min_value=0.0, step=0.1, value=0.0)
    x2 = st.number_input("x2", value=0.0, step=0.1)
    x3 = st.number_input("x3", min_value=0, step=1, value=0)
    
    if st.button("Предсказать x1"):
        input_data = pd.DataFrame({
            'ego_id': [ego_id],
            'u': [u],
            'v': [v],
            't': [t],
            'x2': [x2],
            'x3': [x3]
        })
        processed_data = preprocess_input(input_data, attr_df)
        prediction = model.predict(processed_data)
        st.success(f"Предсказанное значение x1: {prediction[0]:.6f}")
    
    st.header("Загрузка файла для расчёта RMSE")
    
    uploaded_file = st.file_uploader("Выберите CSV-файл", type=["csv"])
    if uploaded_file is not None:
        try:
            test_data = pd.read_csv(uploaded_file)
            st.write("Загруженные данные:")
            st.dataframe(test_data.head())
            
            # Проверка наличия целевой переменной
            if 'x1' not in test_data.columns:
                st.error("Файл должен содержать колонку 'x1' для расчёта RMSE.")
            else:
                # Предобработка данных
                processed_test = preprocess_input(test_data, attr_df)
                
                # Предсказания
                y_true = test_data['x1'].values
                y_pred = model.predict(processed_test)
                
                # Расчёт RMSE
                rmse = mean_squared_error(y_true, y_pred, squared=False)
                st.write(f"RMSE для загруженного файла: {rmse:.6f}")
        except Exception as e:
            st.error(f"Произошла ошибка при обработке файла: {e}")

if __name__ == "__main__":
    main()
