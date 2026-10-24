# ============================================================
# 03_graph_construction.py
# Space Debris Collision Risk Prediction
# KNN Graph Construction
# ============================================================

from pathlib import Path
import pickle

import pandas as pd
import networkx as nx
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

INPUT_FILE = DATA_DIR / "all_features.csv"
OUTPUT_FILE = DATA_DIR / "space_graph.pkl"


# ============================================================
# LOAD FEATURES
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("=" * 60)
print("GRAPH CONSTRUCTION")
print("=" * 60)

print("\nFeature shape:", df.shape)


# ============================================================
# REMOVE INVALID POSITION ROWS
# ============================================================

before = len(df)

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

removed = before - len(df)

print(f"\nRemoved invalid objects: {removed}")
print(f"Objects remaining: {len(df)}")


# ============================================================
# CREATE GRAPH
# ============================================================

G = nx.Graph()


# ============================================================
# ADD NODES
# ============================================================

for index, row in df.iterrows():

    G.add_node(
        index,

        name=row["Satellite Name"],

        object_type=row["Object Type"],

        inclination=row["Inclination"],

        eccentricity=row["Eccentricity"],

        mean_motion=row["Mean Motion"],

        raan=row["RAAN"],

        arg_perigee=row["Argument of Perigee"],

        mean_anomaly=row["Mean Anomaly"],

        latitude=row["Latitude"],

        longitude=row["Longitude"],

        altitude=row["Altitude (km)"],

        vx=row["Vx (km/s)"],

        vy=row["Vy (km/s)"],

        vz=row["Vz (km/s)"],

        speed=row["Speed (km/s)"],

        x=row["X"],

        y=row["Y"],

        z=row["Z"]
    )


# ============================================================
# KNN FEATURES
# ============================================================

features = df[
    [
        "X",
        "Y",
        "Z"
    ]
]


# ============================================================
# KNN GRAPH
# ============================================================

knn = NearestNeighbors(
    n_neighbors=6,
    metric="euclidean"
)

knn.fit(features)

distances, indices = knn.kneighbors(features)


# ============================================================
# ADD EDGES
# ============================================================

for i, neighbors in enumerate(indices):

    # neighbors[0] is the object itself
    for j in neighbors[1:]:

        G.add_edge(i, j)


# ============================================================
# RESULTS
# ============================================================

print("\nGraph created")

print(
    "Number of nodes:",
    G.number_of_nodes()
)

print(
    "Number of edges:",
    G.number_of_edges()
)


# ============================================================
# SAVE GRAPH
# ============================================================

with open(OUTPUT_FILE, "wb") as f:

    pickle.dump(G, f)


print(
    "\nGraph saved to:",
    OUTPUT_FILE
)