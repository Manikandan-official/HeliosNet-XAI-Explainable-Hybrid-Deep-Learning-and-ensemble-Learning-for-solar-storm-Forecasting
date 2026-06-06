import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

# ============================================================
# LOAD DATA
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    'data',
    'balanced_dataset.csv'
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    'outputs',
    'plots'
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("FEATURE SELECTION ANALYSIS")
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
# CORRELATION MATRIX
# ============================================================

print("\nGenerating correlation matrix...")

corr = X.corr()

plt.figure(figsize=(14, 10))
sns.heatmap(corr, cmap='coolwarm', center=0)

plt.title("Feature Correlation Matrix")

corr_path = os.path.join(
    OUTPUT_DIR,
    'feature_correlation_matrix.png'
)

plt.tight_layout()
plt.savefig(corr_path)
plt.close()

print(f"Saved:\n{corr_path}")

# ============================================================
# RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\nTraining Random Forest for feature ranking...")

rf = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

rf.fit(X, y)

importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by='importance',
    ascending=False
)

print("\nTop Features:\n")
print(importance_df.head(15))

# ============================================================
# FEATURE IMPORTANCE PLOT
# ============================================================

plt.figure(figsize=(10, 8))

sns.barplot(
    data=importance_df.head(15),
    x='importance',
    y='feature'
)

plt.title("Top Feature Importances")

importance_path = os.path.join(
    OUTPUT_DIR,
    'rf_feature_importance.png'
)

plt.tight_layout()
plt.savefig(importance_path)
plt.close()

print(f"\nSaved:\n{importance_path}")

# ============================================================
# SELECT IMPORTANT FEATURES
# ============================================================

selector = SelectFromModel(
    rf,
    threshold='median',
    prefit=True
)

selected_features = X.columns[selector.get_support()]

print("\nSelected Features:\n")

for f in selected_features:
    print(f" - {f}")

# ============================================================
# SAVE REDUCED DATASET
# ============================================================

reduced_df = df[
    list(selected_features) + ['storm_label']
]

reduced_path = os.path.join(
    BASE_DIR,
    'data',
    'reduced_dataset.csv'
)

reduced_df.to_csv(reduced_path, index=False)

print("\nReduced dataset saved:")
print(reduced_path)

print("\nReduced shape:")
print(reduced_df.shape)

print("\n" + "=" * 60)
print("FEATURE SELECTION COMPLETE")
print("=" * 60)

