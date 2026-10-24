from pathlib import Path
import math
import pickle

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "all_features.csv"
GRAPH_FILE = DATA_DIR / "space_graph.pkl"
OUTPUT_FILE = DATA_DIR / "labeled_features.csv"


# Load data
df = pd.read_csv(INPUT_FILE)

df = df.dropna(
    subset=[
        "X",
        "Y",
        "Z",
        "Vx (km/s)",
        "Vy (km/s)",
        "Vz (km/s)"
    ]
).reset_index(drop=True)

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

print(G)


# Calculate risk scores
risk_scores = {}

for node in G.nodes():

    neighbors = list(G.neighbors(node))

    x1 = df.loc[node, "X"]
    y1 = df.loc[node, "Y"]
    z1 = df.loc[node, "Z"]

    vx1 = df.loc[node, "Vx (km/s)"]
    vy1 = df.loc[node, "Vy (km/s)"]
    vz1 = df.loc[node, "Vz (km/s)"]

    pairwise_risks = []
    weights = []

    for neighbor in neighbors:

        x2 = df.loc[neighbor, "X"]
        y2 = df.loc[neighbor, "Y"]
        z2 = df.loc[neighbor, "Z"]

        vx2 = df.loc[neighbor, "Vx (km/s)"]
        vy2 = df.loc[neighbor, "Vy (km/s)"]
        vz2 = df.loc[neighbor, "Vz (km/s)"]

        distance = math.sqrt(
            (x2 - x1) ** 2 +
            (y2 - y1) ** 2 +
            (z2 - z1) ** 2
        )

        # Skip identical positions
        if distance == 0:
            continue

        relative_speed = math.sqrt(
            (vx2 - vx1) ** 2 +
            (vy2 - vy1) ** 2 +
            (vz2 - vz1) ** 2
        )

        risk = relative_speed / distance

        pairwise_risks.append(risk)
        weights.append(1 / (distance + 1e-6))

    if len(pairwise_risks) > 0:

        weighted_risk = np.average(
            pairwise_risks,
            weights=weights
        )

        max_risk = max(pairwise_risks)

        risk_scores[node] = (
            0.7 * weighted_risk +
            0.3 * max_risk
        )

    else:
        risk_scores[node] = 0


# Convert to DataFrame
risk_df = pd.DataFrame(
    list(risk_scores.items()),
    columns=["Node", "Risk Score"]
)


# Binary labels
threshold = risk_df["Risk Score"].median()

print("Threshold:", threshold)


def assign_label(risk):

    if risk <= threshold:
        return "Low Risk"

    else:
        return "High Risk"


risk_df["Risk Label"] = risk_df["Risk Score"].apply(assign_label)


# Add labels to original dataframe
df["Risk Score"] = risk_df["Risk Score"]
df["Risk Label"] = risk_df["Risk Label"]


# Save
df.to_csv(OUTPUT_FILE, index=False)

print("\nRisk distribution:")
print(risk_df["Risk Label"].value_counts())

print(f"\nSaved successfully: {OUTPUT_FILE}")