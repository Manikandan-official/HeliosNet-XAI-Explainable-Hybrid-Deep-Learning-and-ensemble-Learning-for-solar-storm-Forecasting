
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score
)

from xgboost import XGBClassifier

print("="*60)
print("STRATIFIED K-FOLD XGBOOST")
print("="*60)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------

df = pd.read_csv("data/reduced_dataset.csv")

drop_cols = ['storm_label']

X = df.drop(columns=drop_cols)
y = df['storm_label']

print(f"\nDataset shape: {df.shape}")
print(f"Feature count : {X.shape[1]}")

# ---------------------------------------------------
# K-FOLD SETUP
# ---------------------------------------------------

skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

acc_scores = []
auc_scores = []
f1_scores = []

fold = 1

# ---------------------------------------------------
# CROSS VALIDATION
# ---------------------------------------------------

for train_idx, test_idx in skf.split(X, y):

    print(f"\n========== Fold {fold} ==========")

    X_train = X.iloc[train_idx]
    X_test  = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test  = y.iloc[test_idx]

    model = XGBClassifier(
        max_depth=6,
        learning_rate=0.09,
        n_estimators=500,
        subsample=0.77,
        colsample_bytree=0.96,
        gamma=2.7,
        min_child_weight=2,
        reg_alpha=4.0,
        reg_lambda=0.22,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1  = f1_score(y_test, y_pred)

    acc_scores.append(acc)
    auc_scores.append(auc)
    f1_scores.append(f1)

    print(f"Accuracy : {acc:.4f}")
    print(f"ROC AUC  : {auc:.4f}")
    print(f"F1 Score : {f1:.4f}")

    fold += 1

# ---------------------------------------------------
# FINAL RESULTS
# ---------------------------------------------------

print("\n" + "="*60)
print("FINAL CROSS VALIDATION RESULTS")
print("="*60)

print(f"\nMean Accuracy : {np.mean(acc_scores):.4f}")
print(f"Std Accuracy  : {np.std(acc_scores):.4f}")

print(f"\nMean ROC AUC  : {np.mean(auc_scores):.4f}")
print(f"Std ROC AUC   : {np.std(auc_scores):.4f}")

print(f"\nMean F1 Score : {np.mean(f1_scores):.4f}")
print(f"Std F1 Score  : {np.std(f1_scores):.4f}")

print("\nDone.")
