import os
import pandas as pd
import numpy as np
from datetime import timedelta

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

print("=" * 60)
print("SOLAR STORM DATA PREPROCESSING")
print("=" * 60)

# ---------------------------------------------------
# LOAD DATASETS
# ---------------------------------------------------
print("\nLoading datasets...")

sharp = pd.read_csv(os.path.join(DATA_DIR, 'sharp_data.csv'))
cme   = pd.read_csv(os.path.join(DATA_DIR, 'lasco_cme.csv'))
omni  = pd.read_csv(os.path.join(DATA_DIR, 'omni_solarwind.csv'))
kp    = pd.read_csv(os.path.join(DATA_DIR, 'kp_index.csv'))

# Convert timestamps
sharp['time'] = pd.to_datetime(sharp['time'])
cme['cme_time'] = pd.to_datetime(cme['cme_time'])
omni['time'] = pd.to_datetime(omni['time'])
kp['time'] = pd.to_datetime(kp['time'])

print(f"SHARP records : {len(sharp):,}")
print(f"CME records   : {len(cme):,}")
print(f"OMNI records  : {len(omni):,}")
print(f"Kp records    : {len(kp):,}")

# ---------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------
print("\nEngineering physics-informed features...")

cme['earth_directed_score'] = np.cos(
    np.radians(cme['position_angle'].fillna(0))
)

cme['cme_energy_proxy'] = (
    (cme['speed'].fillna(0) ** 2) *
    cme['width'].fillna(0)
)

# ---------------------------------------------------
# TEMPORAL ALIGNMENT
# ---------------------------------------------------
print("\nPerforming temporal alignment...")

training_rows = []

for idx, row in cme.iterrows():

    try:
        cme_time = row['cme_time']
        speed = row['speed']

        if pd.isna(speed) or speed <= 0:
            continue

        # ------------------------------------------------
        # CME ARRIVAL TIME ESTIMATION
        # ------------------------------------------------
        transit_hours = (1.5e8 / speed) / 3600
        arrival_time = cme_time + timedelta(hours=transit_hours)

        # ------------------------------------------------
        # SHARP FEATURES BEFORE CME
        # ------------------------------------------------
        sharp_window = sharp[
            (sharp['time'] >= cme_time - timedelta(hours=2)) &
            (sharp['time'] <= cme_time)
        ]

        if len(sharp_window) == 0:
            continue

        sharp_features = sharp_window.mean(numeric_only=True)

        # ------------------------------------------------
        # OMNI FEATURES AROUND ARRIVAL
        # ------------------------------------------------
        omni_window = omni[
            (omni['time'] >= arrival_time - timedelta(hours=12)) &
            (omni['time'] <= arrival_time + timedelta(hours=12))
        ]

        if len(omni_window) == 0:
            continue

        bz_min = omni_window['bz_gsm'].min()
        bz_mean = omni_window['bz_gsm'].mean()
        bt_mean = omni_window['bt'].mean()
        speed_max = omni_window['solar_wind_speed'].max()

        southward_bz_hours = (omni_window['bz_gsm'] < -5).sum()

        # ------------------------------------------------
        # KP LABEL CREATION
        # ------------------------------------------------
        kp_window = kp[
            (kp['time'] >= arrival_time + timedelta(days=1)) &
            (kp['time'] <= arrival_time + timedelta(days=3))
        ]

        if len(kp_window) == 0:
            continue

        max_kp = kp_window['kp_index'].max()

        label = 1 if max_kp >= 5 else 0

        # ------------------------------------------------
        # PHYSICS-INFORMED FEATURES
        # ------------------------------------------------
        shear_flux_product = (
            sharp_features.get('meanshr', 0) *
            sharp_features.get('usflux', 0) / 1e22
        )

        magnetic_complexity = (
            sharp_features.get('shrgt45', 0) *
            sharp_features.get('usflux', 0) / 1e22
        )

        storm_potential = (
            (speed / 1000) *
            (row['width'] / 100) *
            (1 + sharp_features.get('shrgt45', 0) / 50)
        )

        coupling_function = (
            (speed_max ** 1.33) *
            (bt_mean ** 0.67) *
            abs(bz_min)
        ) / 1000

        final_row = {

            # TIMES
            'cme_time': cme_time,
            'arrival_time': arrival_time,

            # CME FEATURES
            'cme_speed': speed,
            'cme_width': row['width'],
            'is_halo': row['is_halo'],
            'earth_directed_score': row['earth_directed_score'],
            'cme_energy_proxy': row['cme_energy_proxy'],

            # SHARP FEATURES
            'usflux': sharp_features.get('usflux', 0),
            'meanshr': sharp_features.get('meanshr', 0),
            'shrgt45': sharp_features.get('shrgt45', 0),

            # ENGINEERED FEATURES
            'shear_flux_product': shear_flux_product,
            'magnetic_complexity': magnetic_complexity,
            'storm_potential': storm_potential,

            # OMNI FEATURES
            'bz_min': bz_min,
            'bz_mean': bz_mean,
            'bt_mean': bt_mean,
            'speed_max': speed_max,
            'southward_bz_hours': southward_bz_hours,
            'coupling_function': coupling_function,

            # LABEL
            'max_kp': max_kp,
            'storm_label': label
        }

        training_rows.append(final_row)

    except:
        continue

# ---------------------------------------------------
# FINAL DATASET
# ---------------------------------------------------
dataset = pd.DataFrame(training_rows)

print("\nCleaning dataset...")

dataset.replace([np.inf, -np.inf], np.nan, inplace=True)
dataset.dropna(inplace=True)

dataset.sort_values('cme_time', inplace=True)

output_path = os.path.join(DATA_DIR, 'final_dataset.csv')
dataset.to_csv(output_path, index=False)

print("\n" + "=" * 60)
print("PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nFinal dataset shape: {dataset.shape}")

print("\nStorm distribution:")
print(dataset['storm_label'].value_counts())

print("\nSaved:")
print(output_path)

print("\nFirst 5 rows:")
print(dataset.head())
