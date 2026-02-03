from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import pandas as pd
import joblib
from pathlib import Path

# Автоматически определяем корень проекта
PROJECT_ROOT = Path(__file__).parent.parent

# Загрузка модели и препроцессора
model_path = PROJECT_ROOT / 'models' / 'uplift_model.joblib'
preprocessor_path = PROJECT_ROOT / 'models' / 'preprocessor.joblib'

model = joblib.load(model_path)
preprocessor = joblib.load(preprocessor_path)

app = FastAPI(
    title="Uplift Marketing API",
    description="API для расчёта uplift_score на новых клиентах",
    version="1.0"
)


@app.get("/")
def home():
    return {"message": "Uplift API работает! Используйте POST /predict для загрузки CSV"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Чтение загруженного CSV
        df_new = pd.read_csv(file.file)

        # Подготовка признаков
        feature_cols = [col for col in df_new.columns if col not in ['conversion', 'treatment', 'offer']]
        X_new = preprocessor.transform(df_new[feature_cols])

        # Предсказание uplift
        prop_score = model['prop_model'].predict_proba(X_new)[:, 1]
        effect_t = model['effect_t'].predict(X_new)
        effect_c = model['effect_c'].predict(X_new)
        uplift_pred = prop_score * effect_t + (1 - prop_score) * effect_c
        uplift_pred = np.clip(uplift_pred, -0.15, 0.30)

        df_new['uplift_score'] = uplift_pred
        results = df_new.sort_values('uplift_score', ascending=False)

        return JSONResponse(content=results.to_dict(orient="records"))

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)