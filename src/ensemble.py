import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'large_dataset.csv')

print("=" * 60)
print("ENSEMBLE SOLAR STORM PREDICTION")
print("=" * 60)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=['storm_label'])
y = df['storm_label']

print(f"Dataset shape: {df.shape}")

# -------------------------------------------------------
# SPLIT
# -------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------------------------------
# XGBOOST
# -------------------------------------------------------
print("\nTraining XGBoost...")

xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    tree_method='hist',
    random_state=42
)

xgb_model.fit(X_train, y_train)

xgb_prob = xgb_model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# CATBOOST
# -------------------------------------------------------
print("\nTraining CatBoost...")

cat_model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=8,
    loss_function='Logloss',
    eval_metric='AUC',
    verbose=200,
    random_seed=42,
    task_type='CPU'
)

cat_model.fit(X_train, y_train)

cat_prob = cat_model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# ENSEMBLE
# -------------------------------------------------------
print("\nCombining predictions...")

ensemble_prob = (
    0.5 * xgb_prob +
    0.5 * cat_prob
)

ensemble_pred = (ensemble_prob >= 0.5).astype(int)

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------
accuracy = accuracy_score(y_test, ensemble_pred)
auc = roc_auc_score(y_test, ensemble_prob)

print("\n" + "=" * 60)
print("ENSEMBLE PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, ensemble_pred))

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
cm = confusion_matrix(y_test, ensemble_pred)

print("\nConfusion Matrix:\n")
print(cm)

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
joblib.dump(xgb_model, 'models/ensemble_xgb.pkl')
joblib.dump(cat_model, 'models/ensemble_cat.pkl')

print("\nModels saved.")

print("\n" + "=" * 60)
print("ENSEMBLE COMPLETE")
print("=" * 60)