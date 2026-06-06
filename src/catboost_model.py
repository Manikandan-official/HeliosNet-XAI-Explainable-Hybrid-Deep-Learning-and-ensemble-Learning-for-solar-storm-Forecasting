import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    auc
)

import matplotlib.pyplot as plt

from catboost import CatBoostClassifier

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'large_dataset.csv'
)

MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR  = os.path.join(BASE_DIR, 'outputs', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("=" * 60)
print("CATBOOST SOLAR STORM PREDICTION")
print("=" * 60)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")

# -------------------------------------------------------
# FEATURES
# -------------------------------------------------------
drop_cols = [
    'storm_label'
]

X = df.drop(columns=drop_cols)
y = df['storm_label']

print(f"\nFeature count: {X.shape[1]}")

# -------------------------------------------------------
# SPLIT
# -------------------------------------------------------
print("\nSplitting train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
print("\nTraining CatBoost model...")

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=10,
    eval_metric='AUC',
    loss_function='Logloss',
    verbose=100,
    random_seed=42,
    thread_count=-1
)

model.fit(X_train, y_train)

# -------------------------------------------------------
# PREDICTIONS
# -------------------------------------------------------
print("\nGenerating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# -------------------------------------------------------
# METRICS
# -------------------------------------------------------
accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_prob)

precision_curve, recall_curve, _ = precision_recall_curve(
    y_test,
    y_prob
)

pr_auc = auc(recall_curve, precision_curve)

print("\n" + "=" * 60)
print("CATBOOST PERFORMANCE")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {roc_auc:.4f}")
print(f"PR AUC   : {pr_auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

cm_path = os.path.join(
    PLOT_DIR,
    'catboost_confusion_matrix.png'
)

plt.savefig(cm_path, bbox_inches='tight')
plt.close()

print(f"\nSaved confusion matrix:")
print(cm_path)

# -------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------
RocCurveDisplay.from_predictions(
    y_test,
    y_prob
)

roc_path = os.path.join(
    PLOT_DIR,
    'catboost_roc_curve.png'
)

plt.savefig(roc_path, bbox_inches='tight')
plt.close()

print(f"\nSaved ROC curve:")
print(roc_path)

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})

importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance_df.head(10))

plt.figure(figsize=(10, 6))

top_features = importance_df.head(10)

plt.barh(
    top_features['feature'][::-1],
    top_features['importance'][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("CatBoost Feature Importance")

fi_path = os.path.join(
    PLOT_DIR,
    'catboost_feature_importance.png'
)

plt.savefig(fi_path, bbox_inches='tight')
plt.close()

print(f"\nSaved feature importance plot:")
print(fi_path)

# -------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------
model_path = os.path.join(
    MODEL_DIR,
    'catboost_model.cbm'
)

model.save_model(model_path)

print(f"\nSaved model:")
print(model_path)

print("\n" + "=" * 60)
print("CATBOOST TRAINING COMPLETE")
print("=" * 60)
