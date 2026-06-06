import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import optuna
import xgboost as xgb

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'balanced_dataset.csv')

print("=" * 60)
print("OPTUNA XGBOOST OPTIMIZATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)

drop_cols = [
    'cme_time',
    'arrival_time',
    'max_kp',
    'storm_label'
]

X = df.drop(columns=drop_cols)
y = df['storm_label']

print(f"\nDataset shape: {df.shape}")
print(f"Feature count : {X.shape[1]}")

# ============================================================
# TIME SERIES CROSS VALIDATION
# ============================================================

tscv = TimeSeriesSplit(n_splits=5)

# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',

        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 100, 800),

        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),

        'gamma': trial.suggest_float('gamma', 0.0, 5.0),

        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),

        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 5.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 5.0),

        'random_state': 42
    }

    auc_scores = []

    for train_idx, val_idx in tscv.split(X):

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)

        model.fit(X_train, y_train)

        probs = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, probs)

        auc_scores.append(auc)

    return np.mean(auc_scores)

# ============================================================
# RUN OPTUNA
# ============================================================

print("\nRunning hyperparameter optimization...\n")

study = optuna.create_study(direction='maximize')

study.optimize(objective, n_trials=30)

# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BEST RESULTS")
print("=" * 60)

print(f"\nBest ROC AUC: {study.best_value:.4f}")

print("\nBest Parameters:\n")

for k, v in study.best_params.items():
    print(f"{k:20s}: {v}")

