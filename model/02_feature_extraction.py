# ============================================================
# 02_feature_extraction.py
# Space Debris Collision Risk Prediction
# Feature Extraction using Skyfield
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
from skyfield.api import load, wgs84, EarthSatellite


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "raw_tle_all.txt"
OUTPUT_FILE = DATA_DIR / "all_features.csv"


# ============================================================
# LOAD TLE DATA
# ============================================================

print("=" * 60)
print("SPACE DEBRIS FEATURE EXTRACTION")
print("=" * 60)

print("\nReading TLE file:")
print(INPUT_FILE)

with open(INPUT_FILE, "r", encoding="utf-8", errors="ignore") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"\nTotal lines: {len(lines)}")

total_objects = len(lines) // 3

print(f"Total satellites: {total_objects}")


# ============================================================
# SKYFIELD SETUP
# ============================================================

ts = load.timescale()
t = ts.now()


# ============================================================
# FEATURE EXTRACTION
# ============================================================

features = []

failed = 0

for i in range(0, len(lines) - 2, 3):

    name = lines[i]
    line1 = lines[i + 1]
    line2 = lines[i + 2]

    try:

        # ----------------------------------------------------
        # Create satellite from TLE
        # ----------------------------------------------------

        sat = EarthSatellite(
            line1,
            line2,
            name,
            ts
        )

        # ----------------------------------------------------
        # Position and velocity
        # ----------------------------------------------------

        position = sat.at(t)

        x, y, z = position.position.km

        vx, vy, vz = position.velocity.km_per_s

        speed = np.linalg.norm(
            [vx, vy, vz]
        )

        # ----------------------------------------------------
        # Latitude / Longitude / Altitude
        # ----------------------------------------------------

        subpoint = wgs84.subpoint(position)

        latitude = subpoint.latitude.degrees
        longitude = subpoint.longitude.degrees
        altitude = subpoint.elevation.km

        # ----------------------------------------------------
        # Orbital parameters
        # ----------------------------------------------------

        model = sat.model

        inclination = np.degrees(model.inclo)

        raan = np.degrees(model.nodeo)

        eccentricity = model.ecco

        arg_perigee = np.degrees(model.argpo)

        mean_anomaly = np.degrees(model.mo)

        mean_motion = model.no_kozai

        # ----------------------------------------------------
        # Object type
        # ----------------------------------------------------

        upper_name = name.upper()

        if any(
            keyword in upper_name
            for keyword in [
                "DEB",
                "DEBRIS",
                "R/B",
                "ROCKET",
                "BODY",
                "FRAG",
                "COSMOS",
                "ASAT"
            ]
        ):
            object_type = "Debris"
        else:
            object_type = "Satellite"

        # ----------------------------------------------------
        # Store features
        # ----------------------------------------------------

        features.append(
            {
                "Satellite Name": name,

                "Inclination": inclination,

                "Eccentricity": eccentricity,

                "Mean Motion": mean_motion,

                "RAAN": raan,

                "Argument of Perigee": arg_perigee,

                "Mean Anomaly": mean_anomaly,

                "Latitude": latitude,

                "Longitude": longitude,

                "Altitude (km)": altitude,

                "Vx (km/s)": vx,

                "Vy (km/s)": vy,

                "Vz (km/s)": vz,

                "Speed (km/s)": speed,

                "X": x,

                "Y": y,

                "Z": z,

                "Object Type": object_type
            }
        )

    except Exception as e:

        failed += 1

        if failed <= 10:
            print(
                f"Failed: {name} | "
                f"{type(e).__name__}: {e}"
            )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(features)

# ============================================================
# DEBUG INVALID SKYFIELD RESULTS
# ============================================================

invalid_mask = df[
    [
        "Latitude",
        "Longitude",
        "Altitude (km)",
        "Vx (km/s)",
        "Vy (km/s)",
        "Vz (km/s)",
        "Speed (km/s)",
        "X",
        "Y",
        "Z"
    ]
].isna().any(axis=1)

invalid_objects = df[invalid_mask]

print("\n" + "=" * 60)
print("INVALID SKYFIELD OBJECTS")
print("=" * 60)

print("Invalid objects:", len(invalid_objects))

print("\nFirst 20 invalid objects:")

print(
    invalid_objects[
        [
            "Satellite Name",
            "Inclination",
            "Eccentricity",
            "Mean Motion"
        ]
    ].head(20).to_string(index=False)
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("FEATURE EXTRACTION RESULTS")
print("=" * 60)

print(f"\nSuccessful objects: {len(df)}")
print(f"Failed objects:     {failed}")

print(f"\nFeature shape: {df.shape}")


if len(df) > 0:

    print("\nMissing values:")

    print(
        df[
            [
                "Latitude",
                "Longitude",
                "Altitude (km)",
                "Vx (km/s)",
                "Vy (km/s)",
                "Vz (km/s)",
                "Speed (km/s)",
                "X",
                "Y",
                "Z"
            ]
        ].isna().sum()
    )

    print("\nObject type distribution:")

    print(
        df["Object Type"].value_counts()
    )

    print("\nFirst 5 rows:")

    print(df.head())


# ============================================================
# SAVE
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("SAVED")
print("=" * 60)

print("\nOutput file:")
print(OUTPUT_FILE)

print(f"\nFinal feature shape: {df.shape}")

print("\nFeature extraction complete.")