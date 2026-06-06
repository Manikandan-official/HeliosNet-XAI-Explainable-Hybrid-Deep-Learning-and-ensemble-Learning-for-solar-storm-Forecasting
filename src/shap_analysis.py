import os
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'final_dataset.csv')

MODEL_PATH = os.path.join(BASE_DIR, 'models', 'xgboost_model.pkl')

PLOT_DIR = os.path.join(BASE_DIR, 'outputs', 'plots')

os.makedirs(PLOT_DIR, exist_ok=True)

print("=" * 60)
print("SHAP EXPLAINABILITY ANALYSIS")
print("=" * 60)

# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)

drop_cols = [
    'cme_time',
    'arrival_time',
    'storm_label',
    'max_kp'
]

X = df.drop(columns=drop_cols)

print(f"\nDataset shape: {X.shape}")

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------
print("\nLoading XGBoost model...")

model = joblib.load(MODEL_PATH)

# -------------------------------------------------------
# SHAP EXPLAINER
# -------------------------------------------------------
print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(model)

sample_X = X.sample(min(200, len(X)), random_state=42)

shap_values = explainer.shap_values(sample_X)

# -------------------------------------------------------
# SUMMARY PLOT
# -------------------------------------------------------
print("\nGenerating SHAP summary plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    sample_X,
    show=False
)

summary_path = os.path.join(PLOT_DIR, 'shap_summary.png')

plt.savefig(summary_path, dpi=300, bbox_inches='tight')

print(f"Saved:")
print(summary_path)

# -------------------------------------------------------
# BAR PLOT
# -------------------------------------------------------
print("\nGenerating SHAP importance bar plot...")

plt.figure()

shap.summary_plot(
    shap_values,
    sample_X,
    plot_type='bar',
    show=False
)

bar_path = os.path.join(PLOT_DIR, 'shap_bar.png')

plt.savefig(bar_path, dpi=300, bbox_inches='tight')

print(f"Saved:")
print(bar_path)

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETE")
print("=" * 60)
