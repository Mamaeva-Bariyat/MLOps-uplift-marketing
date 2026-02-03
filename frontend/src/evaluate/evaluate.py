import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def load_model_and_preprocessor(project_root: Path):
    """Load saved model and preprocessor from models folder."""
    model_path = project_root / 'models' / 'uplift_model.joblib'
    preprocessor_path = project_root / 'models' / 'preprocessor.joblib'
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    return model, preprocessor

def predict_uplift(df: pd.DataFrame, model, preprocessor):
    """Predict uplift scores using loaded model."""
    feature_cols = [col for col in df.columns if col not in ['conversion', 'treatment', 'offer']]
    X = preprocessor.transform(df[feature_cols])
    prop_score = model['prop_model'].predict_proba(X)[:, 1]
    effect_t = model['effect_t'].predict(X)
    effect_c = model['effect_c'].predict(X)
    uplift_scores = prop_score * effect_t + (1 - prop_score) * effect_c
    uplift_scores = np.clip(uplift_scores, -0.15, 0.30)
    df['uplift_score'] = uplift_scores
    return df.sort_values('uplift_score', ascending=False)

def analyze_top_clients(df: pd.DataFrame, top_fraction: float = 0.05):
    """Analyze top clients by uplift."""
    top_n = int(len(df) * top_fraction)
    top_clients = df.nlargest(top_n, 'uplift_score')
    print(f"Top {top_fraction:.1%} clients analysis:")
    print(f"Mean uplift: {top_clients['uplift_score'].mean():.4f}")
    print(f"Mean conversion: {top_clients['conversion'].mean():.4f}")
    return top_clients