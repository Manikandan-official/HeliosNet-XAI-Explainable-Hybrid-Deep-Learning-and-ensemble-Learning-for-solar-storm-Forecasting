import os
import pandas as pd

from sklearn.utils import resample

# -------------------------------------------------------
# PATHS
# -------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR, 'data', 'final_dataset.csv')

OUTPUT_PATH = os.path.join(BASE_DIR, 'data', 'balanced_dataset.csv')

print("=" * 60)
print("REBALANCING DATASET")
print("=" * 60)

# -------------------------------------------------------
# LOAD
# -------------------------------------------------------
df = pd.read_csv(DATA_PATH)

print("\nOriginal distribution:\n")
print(df['storm_label'].value_counts())

# -------------------------------------------------------
# SPLIT CLASSES
# -------------------------------------------------------
storm_df = df[df['storm_label'] == 1]
nostorm_df = df[df['storm_label'] == 0]

# -------------------------------------------------------
# UNDERSAMPLE MAJORITY CLASS
# -------------------------------------------------------
storm_downsampled = resample(
    storm_df,
    replace=False,
    n_samples=len(nostorm_df),
    random_state=42
)

# -------------------------------------------------------
# COMBINE
# -------------------------------------------------------
balanced_df = pd.concat([
    storm_downsampled,
    nostorm_df
])

# -------------------------------------------------------
# SHUFFLE
# -------------------------------------------------------
balanced_df = balanced_df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# -------------------------------------------------------
# SAVE
# -------------------------------------------------------
balanced_df.to_csv(OUTPUT_PATH, index=False)

print("\nBalanced distribution:\n")
print(balanced_df['storm_label'].value_counts())

print(f"\nSaved balanced dataset:")
print(OUTPUT_PATH)

print("\nFinal shape:")
print(balanced_df.shape)

print("\n" + "=" * 60)
print("REBALANCING COMPLETE")
print("=" * 60)
