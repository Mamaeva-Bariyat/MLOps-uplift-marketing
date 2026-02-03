import joblib
import json
from lightgbm import LGBMClassifier, LGBMRegressor
import optuna
from sklearn.metrics import roc_auc_score
from . import metrics  # Import if needed
from ..data.get_data import load_raw_data
from ..data.split_dataset import split_data
from ..pipelines.pipeline import build_preprocessing_pipeline
import yaml
from pathlib import Path


def train(config):
    df = load_raw_data(config)

    # Feature engineering
    df['history_discount'] = df['history'] * df['used_discount']
    df['history_bogo'] = df['history'] * df['used_bogo']
    df['recency_history'] = df['recency'] * df['history']
    df['treatment'] = (df['offer'] != 'No Offer').astype(int)

    preprocessor = build_preprocessing_pipeline(config)
    X_preprocessed = preprocessor.fit_transform(df.drop(['offer', 'conversion', 'treatment'], axis=1))

    train_df, test_df = split_data(df, config)

    X_train = preprocessor.transform(train_df.drop(['offer', 'conversion', 'treatment'], axis=1))
    X_test = preprocessor.transform(test_df.drop(['offer', 'conversion', 'treatment'], axis=1))
    t_train, t_test = train_df['treatment'], test_df['treatment']
    y_train, y_test = train_df['conversion'], test_df['conversion']

    # Optuna objective
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 20),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
            'num_leaves': trial.suggest_int('num_leaves', 20, 200),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'random_state': config['train']['random_state'],
            'n_jobs': -1
        }
        model = LGBMClassifier(**params)
        model.fit(X_train[t_train == 1], y_train[t_train == 1])
        pred = model.predict_proba(X_test[t_test == 1])[:, 1]
        return roc_auc_score(y_test[t_test == 1], pred)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=config['train']['optuna_trials'])

    best_params = study.best_params
    best_params.update({'random_state': config['train']['random_state'], 'n_jobs': -1})

    with open(Path(config['folders']['report']) / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=2)

    # Train final X-Learner
    prop_model = LGBMClassifier(**best_params)
    prop_model.fit(X_train, t_train)

    outcome_t = LGBMClassifier(**best_params)
    outcome_c = LGBMClassifier(**best_params)
    outcome_t.fit(X_train[t_train == 1], y_train[t_train == 1])
    outcome_c.fit(X_train[t_train == 0], y_train[t_train == 0])

    pseudo_t = y_train[t_train == 1] - outcome_c.predict_proba(X_train[t_train == 1])[:, 1]
    pseudo_c = outcome_t.predict_proba(X_train[t_train == 0])[:, 1] - y_train[t_train == 0]

    effect_t = LGBMRegressor(**best_params)
    effect_c = LGBMRegressor(**best_params)
    effect_t.fit(X_train[t_train == 1], pseudo_t)
    effect_c.fit(X_train[t_train == 0], pseudo_c)

    uplift_model = {
        'prop_model': prop_model,
        'outcome_t': outcome_t,
        'outcome_c': outcome_c,
        'effect_t': effect_t,
        'effect_c': effect_c
    }

    joblib.dump(uplift_model, Path(config['folders']['models']) / 'uplift_model.joblib')
    joblib.dump(preprocessor, Path(config['folders']['models']) / 'preprocessor.joblib')

    print("Training completed. Model and artifacts saved.")


if __name__ == "__main__":
    config = yaml.safe_load(open('config/params.yml'))
    train(config)