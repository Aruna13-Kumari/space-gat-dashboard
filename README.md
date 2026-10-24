# Space Debris Collision Risk Prediction — Space GAT

A graph-based machine learning project that analyzes satellite and space debris data to classify objects according to their estimated collision risk. The project uses a Graph Attention Network (GAT) and an interactive dashboard to visualize orbital objects and risk information.

## Features

* Satellite and space debris data processing
* Orbital feature extraction using TLE data
* Graph construction based on spatial proximity
* Risk label generation using relative distance and velocity
* Graph Attention Network (GAT) for risk classification
* Interactive dashboard for orbital visualization and system analytics

## Tech Stack

* **Language:** Python
* **Machine Learning:** PyTorch, PyTorch Geometric
* **Data Processing:** Pandas, NumPy, Scikit-learn
* **Graph Processing:** NetworkX
* **Orbital Data:** Skyfield, Two-Line Element (TLE) data
* **Dashboard:** Streamlit, HTML, CSS, CesiumJS
* **Development:** VS Code, Google Colab

## Project Structure

  text
Space_Debris_Project/
├── data/
│   ├── all_features.csv
│   ├── labeled_features.csv
│   ├── space_graph.pkl
│   ├── metrics_final.txt
│   └── confusion_matrix_final.png
├── dashboard_data/
├── model/
│   ├── 01_data_collection.py
│   ├── 02_feature_extraction.py
│   ├── 03_graph_construction.py
│   ├── 04_label_generation.py
│   ├── gat_model.py
│   ├── train.py
│   └── evaluate.py
├── app.py
├── earth.py
├── earth_view.html
├── style.css
├── requirements.txt
└── README.md
```

## Methodology

1. **Data Collection:** Gather satellite and space debris Two-Line Element (TLE) data.
2. **Feature Extraction:** Extract orbital, positional, and velocity-related features.
3. **Graph Construction:** Represent objects as nodes and their spatial relationships as edges.
4. **Risk Label Generation:** Generate binary risk labels using a distance- and relative-velocity-based scoring method.
5. **Model Training:** Train a Graph Attention Network to classify objects into risk categories.
6. **Evaluation:** Evaluate model predictions on a held-out test set.
7. **Visualization:** Display orbital objects and risk analytics through an interactive dashboard.

## Model Architecture

The model uses two Graph Attention Network layers:

* **Input:** 17 node features
* **First layer:** 8 attention heads with 64 hidden features per head
* **Second layer:** 2 output classes
* **Dropout:** 0.3

The model is implemented using PyTorch Geometric.

## Evaluation

Evaluation metrics and the confusion matrix are provided in the `data/` directory.

The model predicts labels derived from the project's risk-scoring methodology. These predictions should not be interpreted as validated real
