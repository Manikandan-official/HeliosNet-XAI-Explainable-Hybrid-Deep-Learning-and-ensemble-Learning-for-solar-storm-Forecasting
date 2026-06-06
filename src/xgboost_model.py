import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from xgboost import XGBClassifier

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'large_dataset.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'models')
PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("XGBOOST SOLAR STORM PREDICTION")
print("=" * 60)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("\nLoading final dataset...")

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
# TRAIN TEST SPLIT
# -------------------------------------------------------
print("\nSplitting train/test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train samples: {len(X_train)}")
print(f"Test samples : {len(X_test)}")

# -------------------------------------------------------
# CLASS BALANCING
# -------------------------------------------------------
neg = (y_train == 0).sum()
pos = (y_train == 1).sum()

scale_pos_weight = neg / pos

print(f"\nscale_pos_weight = {scale_pos_weight:.3f}")

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------
print("\nTraining XGBoost model...")

model = XGBClassifier(
    n_estimators=1200,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42,
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
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC AUC  : {auc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# -------------------------------------------------------
# CONFUSION MATRIX
# -------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['No Storm', 'Storm'],
    yticklabels=['No Storm', 'Storm']
)

plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')

cm_path = os.path.join(PLOT_DIR, 'confusion_matrix.png')

plt.savefig(cm_path, dpi=300, bbox_inches='tight')

print(f"\nSaved confusion matrix:")
print(cm_path)

# -------------------------------------------------------
# ROC CURVE
# -------------------------------------------------------
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(7,6))

plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1], [0,1], linestyle='--')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')

plt.legend()

roc_path = os.path.join(PLOT_DIR, 'roc_curve.png')

plt.savefig(roc_path, dpi=300, bbox_inches='tight')

print(f"\nSaved ROC curve:")
print(roc_path)

# -------------------------------------------------------
# FEATURE IMPORTANCE
# -------------------------------------------------------
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
})

importance = importance.sort_values(
    'importance',
    ascending=False
)

print("\nTop 10 Important Features:\n")
print(importance.head(10))

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance.head(10),
    x='importance',
    y='feature'
)

plt.title('Top 10 Feature Importances')

fi_path = os.path.join(PLOT_DIR, 'feature_importance.png')

plt.savefig(fi_path, dpi=300, bbox_inches='tight')

print(f"\nSaved feature importance plot:")
print(fi_path)

# -------------------------------------------------------
# SAVE MODEL
# -------------------------------------------------------
model_path = os.path.join(MODEL_DIR, 'xgboost_model.pkl')

joblib.dump(model, model_path)

print(f"\nSaved model:")
print(model_path)

print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETE")
print("=" * 60)
