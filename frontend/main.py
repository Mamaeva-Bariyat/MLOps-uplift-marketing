import streamlit as st
import pandas as pd
from pathlib import Path
import joblib

# Импорты из наших модулей
from src.data.get_data import load_test_data
from src.evaluate.evaluate import load_model_and_preprocessor, predict_uplift, analyze_top_clients
from src.plotting.charts import plot_qini_curve, plot_uplift_distribution

# === Определение корня проекта ===
# Если файл main.py находится в frontend/, то корень проекта — на два уровня выше
project_root = Path(__file__).parent.parent

# Загрузка модели и препроцессора (с проверкой)
model_path = project_root / 'models' / 'uplift_model.joblib'
preprocessor_path = project_root / 'models' / 'preprocessor.joblib'

if not model_path.exists() or not preprocessor_path.exists():
    st.error("Модель или препроцессор не найдены в папке models/. Запустите 01_EDA_and_Train.ipynb")
    st.stop()

# Загружаем один раз при старте
model, preprocessor = load_model_and_preprocessor(project_root)

# === Основной дашборд ===
st.set_page_config(page_title="Uplift Marketing Dashboard", layout="wide")
st.title("Uplift Marketing Dashboard")

# === Сайдбар: загрузка файла ===
st.sidebar.header("Загрузка данных")
uploaded_file = st.sidebar.file_uploader("Upload new customer CSV", type="csv")

if uploaded_file:
    # Если пользователь загрузил файл
    df_new = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:", df_new.head())

    if st.button("Calculate Uplift"):
        with st.spinner("Рассчитываем uplift..."):
            results = predict_uplift(df_new.copy(), model, preprocessor)
            st.success("Uplift рассчитан!")

            st.write("Uplift Predictions:", results.head(20))

            top_clients = analyze_top_clients(results)
            st.write("Top Clients Analysis:", top_clients.describe())

            fig = plot_uplift_distribution(results)
            st.pyplot(fig)

else:
    # Если файла нет — используем тестовые данные
    try:
        df_test = load_test_data(project_root)
        st.write("Test Data Preview:", df_test.head())

        if st.button("Evaluate on Test Data"):
            with st.spinner("Рассчитываем uplift на тестовых данных..."):
                results = predict_uplift(df_test.copy(), model, preprocessor)
                st.success("Анализ завершён!")

                st.write("Uplift on Test Data:", results.head(20))

                fig_qini = plot_qini_curve(
                    df_test['conversion'],
                    results['uplift_score'],
                    df_test['treatment']
                )
                st.pyplot(fig_qini)
    except Exception as e:
        st.error(f"Не удалось загрузить тестовые данные: {e}")
        st.info("Загрузите свой CSV-файл для расчёта uplift.")

# === Опциональная кнопка переобучения (если есть функция) ===
if st.sidebar.button("Retrain Model"):
    try:
        from src.train.training import retrain_model

        retrain_model()
        st.success("Модель переобучена!")
        st.experimental_rerun()  # перезапуск дашборда
    except Exception as e:
        st.error(f"Ошибка при переобучении: {e}")

st.caption("Uplift-моделирование на датасете Customer Retention")