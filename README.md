# Limit Order Book (LOB) Network Analysis & GNN Price Prediction

This repository hosts an end-to-end framework for advanced financial market microstructure analysis and price movement prediction. 

Utilizing high-frequency Limit Order Book (LOB) data, the project explores the spatial and temporal relationships between various price and volume levels. Starting from an in-depth exploratory analysis, the framework extracts the market's topology via Network Filtering algorithms (GLASSO and TMFG). These graph structures are then designed to feed a **Graph Neural Network (GNN)** aimed at predicting the future direction of the asset's price (Up, Down, or Flat).

---

## 🏗️ Project Architecture

The repository is divided into sequential modules that reflect the data processing and predictive modeling pipeline:

### 📂 1. LOB Similarity & Exploratory Analysis
*Contains the analytical foundation and data preparation.*
- **Main File:** `csvAnalytics.ipynb`
- **Description:** Leverages GPU acceleration (via NVIDIA RAPIDS) to process massive amounts of LOB data. It calculates multi-lag similarity matrices and statistically analyzes the dependencies between book sides (Ask/Bid), depth levels, and the temporal persistence of information.

### 📂 2. GLASSO Network Modeling
*Network construction via precision matrix estimation.*
- **Main Files:** `glasso_v4_lag20.ipynb`, `glasso_v5_lag20.ipynb` and `glassoLagComparison.ipynb`
- **Description:** Applies the Graphical Lasso to extract sparse networks of conditional dependencies among LOB features. Includes an advanced comparative analysis on the impact of temporal memory (Lag Comparison) and generates interactive 3D visualizations of market dynamics.

### 📂 3. TMFG Topological Filtering
*Extraction of the core information infrastructure (Backbone).*
- **Main Files:** `TMFG_core.py`, `build_graph_tmfg.ipynb`
- **Description:** Implements the Triangulated Maximally Filtered Graph (TMFG) algorithm to filter noise from the similarity matrix while guaranteeing a planar structure. This allows for the identification of market "Hubs", calculation of network centralities, and analysis of clique composition.

### ⏳ 4. GNN Price Prediction (Work in Progress)
*The predictive engine of the framework.*
- **Goal:** Train a Graph Neural Network (e.g., GCN or GAT) using the graphs extracted in the previous modules.
- **Task:** 3-way multiclass classification to predict the asset's price direction in the next tick/time horizon:
  1. **Up** (Increase)
  2. **Down** (Decrease)
  3. **Flat** (Stationary)
- The adjacency matrices (from GLASSO/TMFG) will guide the neural network's *message passing*, allowing the model to capture not just the current state of the LOB, but the complex structural relationships between its levels.

---

## 🚀 Workflow (Pipeline)

1. **Ingestion & EDA:** Raw CSV data is loaded and analyzed in the `1_LOB_Similarity_Analysis` folder to extract baseline metrics.
2. **Graph Generation:** Processed data passes through the `2_GLASSO` and/or `3_TMFG` modules to generate the adjacency matrices (weighted and unweighted) that define the network topology at specific time steps.
3. **Graph Machine Learning:** Node features (LOB prices/volumes) and edges (structural relationships) are passed to the GNN for training and directional price inference.

---

## 🛠️ Prerequisites and Installation

Given the volume of data (High-Frequency Trading) and the use of neural networks, a GPU-accelerated environment is **highly recommended**.

### Base Requirements
- Python 3.10+
- NVIDIA GPU (compatible with CUDA 12.x)

### Environment Setup
It is recommended to use `conda` to properly manage RAPIDS libraries and deep learning frameworks:

```bash
# For data analysis and GPU acceleration
conda install -y -c rapidsai -c conda-forge -c nvidia \
    cudf=26.04 cupy pandas scikit-learn matplotlib seaborn \
    plotly networkx pyvis tqdm

# For the upcoming GNN module (PyTorch & PyTorch Geometric)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install pyg -c pyg
