from pathlib import Path
import pickle

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

from gat_model import GATModel

# ---------------------------------
# Paths
# ---------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

FEATURE_FILE = DATA_DIR / "labeled_features.csv"
GRAPH_FILE = DATA_DIR / "space_graph.pkl"
MODEL_FILE = DATA_DIR / "gat_model_final.pth"
GRAPH_OUTPUT = DATA_DIR / "graph_final.pt"


# ---------------------------------
# Load data
# ---------------------------------

df = pd.read_csv(FEATURE_FILE)

with open(GRAPH_FILE, "rb") as f:
    G = pickle.load(f)

print("Dataset shape:", df.shape)
print(G)


# ---------------------------------
# Convert NetworkX → PyTorch Geometric
# ---------------------------------

graph_data = from_networkx(G)

print(graph_data)


# ---------------------------------
# Object type mapping
# ---------------------------------

df["Object Type"] = df["Object Type"].map({
    "Satellite": 0,
    "Debris": 1
})


# ---------------------------------
# Feature matrix
# ---------------------------------

feature_columns = [
    "Object Type",
    "Inclination",
    "Eccentricity",
    "Mean Motion",
    "RAAN",
    "Argument of Perigee",
    "Mean Anomaly",
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


# ---------------------------------
# Standardize features
# ---------------------------------

scaler = StandardScaler()

scaled_features = scaler.fit_transform(
    df[feature_columns]
)


# ---------------------------------
# Features → PyTorch tensor
# ---------------------------------

x = torch.tensor(
    scaled_features,
    dtype=torch.float
)


# ---------------------------------
# Risk label mapping
# ---------------------------------

label_mapping = {
    "Low Risk": 0,
    "High Risk": 1
}

y = torch.tensor(
    df["Risk Label"].map(label_mapping).values,
    dtype=torch.long
)


# ---------------------------------
# Create PyG Data object
# ---------------------------------

data = Data(
    x=x,
    edge_index=graph_data.edge_index,
    y=y
)

print(data)

print("\nFeatures:", data.x.shape)
print("Edges:", data.edge_index.shape)
print("Labels:", data.y.shape)


# ---------------------------------
# Train / validation / test split
# ---------------------------------

num_nodes = data.num_nodes

indices = torch.arange(num_nodes)

train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=42,
    stratify=data.y.numpy()
)

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=42,
    stratify=data.y[temp_idx].numpy()
)

print("\nTraining:", len(train_idx))
print("Validation:", len(val_idx))
print("Testing:", len(test_idx))


# ---------------------------------
# Create masks
# ---------------------------------

data.train_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

data.val_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

data.test_mask = torch.zeros(
    num_nodes,
    dtype=torch.bool
)

data.train_mask[train_idx] = True
data.val_mask[val_idx] = True
data.test_mask[test_idx] = True


# ---------------------------------
# Clean graph object
# ---------------------------------

clean_data = Data(
    x=data.x,
    edge_index=data.edge_index,
    y=data.y
)

clean_data.train_mask = data.train_mask
clean_data.val_mask = data.val_mask
clean_data.test_mask = data.test_mask


# Save processed graph
torch.save(
    clean_data,
    GRAPH_OUTPUT
)

print("\nProcessed graph saved.")


# ---------------------------------
# Create GAT model
# ---------------------------------

model = GATModel(
    input_dim=data.x.shape[1],
    hidden_dim=64,
    output_dim=2,
    heads=8,
    dropout=0.3
)

print("\n", model)


# ---------------------------------
# Loss + optimizer
# ---------------------------------

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.005
)


# ---------------------------------
# Training
# ---------------------------------

train_losses = []

num_epochs = 200

for epoch in range(num_epochs):

    model.train()

    optimizer.zero_grad()

    out = model(
        data.x,
        data.edge_index
    )

    loss = criterion(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()

    optimizer.step()

    train_losses.append(loss.item())

    print(
        f"Epoch {epoch + 1}/{num_epochs}, "
        f"Loss: {loss.item():.4f}"
    )


# ---------------------------------
# Save trained model
# ---------------------------------

torch.save(
    model.state_dict(),
    MODEL_FILE
)

print("\nModel saved to:")
print(MODEL_FILE)