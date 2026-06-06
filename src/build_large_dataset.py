# ============================================================
# OFFLINE LARGE SOLAR STORM DATASET GENERATOR
# ============================================================

import pandas as pd
import numpy as np

print("=" * 60)
print("GENERATING LARGE OFFLINE SOLAR STORM DATASET")
print("=" * 60)

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

N_SAMPLES = 50000

np.random.seed(42)

records = []

# ------------------------------------------------------------
# GENERATE EVENTS
# ------------------------------------------------------------

for i in range(N_SAMPLES):

    # CME features
    cme_speed = np.random.normal(900, 400)
    cme_speed = np.clip(cme_speed, 200, 3000)

    cme_width = np.random.uniform(10, 360)

    is_halo = 1 if cme_width >= 180 else 0

    earth_directed_score = (
        (cme_speed / 3000) * 0.6 +
        (cme_width / 360) * 0.4
    )

    cme_energy_proxy = cme_speed * cme_width

    # SHARP magnetic parameters
    usflux = np.random.uniform(1e21, 1e23)

    meanshr = np.random.uniform(5, 90)

    shrgt45 = np.random.uniform(1e4, 1e6)

    shear_flux_product = meanshr * np.log10(usflux)

    magnetic_complexity = (
        meanshr * 0.5 +
        np.log10(usflux)
    )

    storm_potential = (
        cme_speed * 0.4 +
        cme_width * 0.2 +
        magnetic_complexity * 0.4
    )

    # OMNI solar wind features
    bz_min = np.random.uniform(-35, 15)

    bz_mean = np.random.uniform(-12, 12)

    bt_mean = np.random.uniform(1, 35)

    speed_max = cme_speed + np.random.uniform(0, 500)

    southward_bz_hours = np.random.uniform(0, 30)

    coupling_function = (
        bt_mean *
        abs(min(bz_min, 0)) *
        np.sqrt(speed_max)
    )

    # --------------------------------------------------------
    # PHYSICS-INFORMED STORM SCORE
    # --------------------------------------------------------

    storm_score = (
        earth_directed_score * 25 +
        abs(min(bz_min, 0)) * 1.5 +
        coupling_function / 120 +
        southward_bz_hours * 0.8 +
        is_halo * 8
    )

    # add noise
    storm_score += np.random.normal(0, 5)

    # balanced labeling
    storm_label = 1 if storm_score > 35 else 0

    records.append({
        "cme_speed": cme_speed,
        "cme_width": cme_width,
        "is_halo": is_halo,
        "earth_directed_score": earth_directed_score,
        "cme_energy_proxy": cme_energy_proxy,
        "usflux": usflux,
        "meanshr": meanshr,
        "shrgt45": shrgt45,
        "shear_flux_product": shear_flux_product,
        "magnetic_complexity": magnetic_complexity,
        "storm_potential": storm_potential,
        "bz_min": bz_min,
        "bz_mean": bz_mean,
        "bt_mean": bt_mean,
        "speed_max": speed_max,
        "southward_bz_hours": southward_bz_hours,
        "coupling_function": coupling_function,
        "storm_label": storm_label
    })

# ------------------------------------------------------------
# CREATE DATAFRAME
# ------------------------------------------------------------

df = pd.DataFrame(records)

print("\nDataset created")

print("\nClass distribution:")
print(df["storm_label"].value_counts())

print("\nDataset shape:")
print(df.shape)

# ------------------------------------------------------------
# SAVE
# ------------------------------------------------------------

save_path = "data/large_dataset.csv"

df.to_csv(save_path, index=False)

print("\nSaved dataset:")
print(save_path)

print("\nFirst rows:")
print(df.head())

print("\n" + "=" * 60)
print("DATASET GENERATION COMPLETE")
print("=" * 60)

