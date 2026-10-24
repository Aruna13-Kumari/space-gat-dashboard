import pandas as pd
import json
from pathlib import Path


def create_earth():
    print("CREATING OBJECT JSON")
    df = pd.read_csv(
        "dashboard_data/labeled_features.csv"
    )

    objects = []

    for _, row in df.iterrows():

        objects.append({

            "name": str(row["Satellite Name"]),

            "lat": float(row["Latitude"]),
            "lon": float(row["Longitude"]),

            # km → meters for Cesium
            "alt": float(row["Altitude (km)"]) * 1000,

            "risk": str(row["Risk Label"]) if "Risk Label" in df.columns else "Low",

            # change column name if yours differs
            "type": str(row["Object Type"]) if "Object Type" in df.columns else "Satellite"

        })


    Path("dashboard_data/objects.json").write_text(
        json.dumps(objects)
    )

    Path("dashboard_data/view_mode.json").write_text(
        '{"mode":"type"}'
    )

    return "earth_view.html"