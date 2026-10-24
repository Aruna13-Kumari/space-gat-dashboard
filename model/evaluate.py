from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

from gat_model import GATModel


# -----------------------------
# Paths
# -----------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

GRAPH_FILE = DATA_DIR / "graph_final.pt"
MODEL_FILE = DATA_DIR / "gat_model_final.pth"

METRICS_FILE = DATA_DIR / "metrics_final.txt"
CONFUSION_FILE = DATA_DIR / "confusion_matrix_final.png"


# -----------------------------
# Load processed graph
# -----------------------------

data = torch.load(
    GRAPH_FILE,
    weights_only=False
)


# -----------------------------
# Create model
# -----------------------------

model = GATModel(
    input_dim=data.x.shape[1],
    hidden_dim=64,
    output_dim=2,
    heads=8,
    dropout=0.3
)


# -----------------------------
# Load trained weights
# -----------------------------

model.load_state_dict(
    torch.load(
        MODEL_FILE,
        weights_only=True
    )
)


# -----------------------------
# Evaluation
# -----------------------------

model.eval()

with torch.no_grad():

    out = model(
        data.x,
        data.edge_index
    )

    pred = out.argmax(dim=1)


# -----------------------------
# Test predictions
# -----------------------------

y_true = (
    data.y[data.test_mask]
    .cpu()
    .numpy()
)

y_pred = (
    pred[data.test_mask]
    .cpu()
    .numpy()
)


# -----------------------------
# Metrics
# -----------------------------

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted"
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted"
)


print("\nGAT Model\n")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# -----------------------------
# Save metrics
# -----------------------------

metrics = f"""
GAT Model

Accuracy : {accuracy:.4f}
Precision: {precision:.4f}
Recall   : {recall:.4f}
F1 Score : {f1:.4f}
"""

METRICS_FILE.write_text(metrics)

print("\nMetrics saved.")


# -----------------------------
# Confusion matrix
# -----------------------------

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\nConfusion Matrix:")
print(cm)


disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Low Risk",
        "High Risk"
    ]
)

disp.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Model Confusion Matrix")

plt.savefig(
    CONFUSION_FILE,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nConfusion matrix saved.")