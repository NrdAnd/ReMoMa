# GLASSO - Limit Order Book (LOB) Network Analysis

This repository contains tools for analyzing market microstructure dynamics using the **Graphical Lasso (GLASSO)** algorithm. The goal is to estimate sparse precision matrices (inverse covariance) to uncover conditional dependency networks between different levels of the Limit Order Book, spanning both spatial (cross-sectional) and temporal (time-lagged) dimensions.

## 📂 Repository Structure

The folder includes two primary Jupyter Notebooks:

1.  **`glasso_v4_lag20.ipynb`**: 
    The main pipeline for data processing and network generation.
    - **GPU Acceleration**: Utilizes `cuDF` (NVIDIA RAPIDS) for high-performance loading and processing of large-scale LOB CSV files.
    - **Feature Engineering**: Builds state variables from book snapshots (prices and volumes for multiple bid/ask levels).
    - **3D Interactive Visualization**: Implements a 3D network visualization using `Plotly`. It exports an interactive HTML file allowing exploration of node clusters (e.g., price vs. volume families), edge weights, and connection strengths.
    - **State Only**: it's built using only size features
    - **Considered Lag**: 20

2.  **`glasso_v5_lag20.ipynb`**: 
    The main pipeline for data processing and network generation.
    - **GPU Acceleration**: Utilizes `cuDF` (NVIDIA RAPIDS) for high-performance loading and processing of large-scale LOB CSV files.
    - **Feature Engineering**: Builds state variables from book snapshots (prices and volumes for multiple bid/ask levels).
    - **3D Interactive Visualization**: Implements a 3D network visualization using `Plotly`. It exports an interactive HTML file allowing exploration of node clusters (e.g., price vs. volume families), edge weights, and connection strengths.
    - **Engineered Version**: it's built using sizes, prices and engineered features like spread, mid-price, bid total depth, ask total depth and imbalance
    - **Considered Lag**: 20

3.  **`glassoLagComparison.ipynb`**: 
    A specialized comparison report designed to study how temporal memory (lags) affects the resulting network.
    It's built on version 4, so on the engineered one.
    - **Lag Evaluation**: Compares models trained with different time windows (e.g., `lag_3`, `lag_10`, `lag_20`).
    - **Comparative Metrics**: Analyzes differences in graph density, computation time (fit time), and connection types (e.g., *cross-lag*, *same-level*, or *global-level* shares).
    - **Model Selection**: Provides insights into selecting the optimal lag based on the balance between temporal richness and interpretability.

## 🛠️ Prerequisites

To run these notebooks, you need a Python environment configured for data science. Given the GPU-accelerated components, an **NVIDIA GPU** is highly recommended.

**Core Dependencies:**
```bash
# Standard data science and visualization
pip install numpy pandas scikit-learn plotly

# GPU Acceleration (RAPIDS)
# Note: cuDF installation depends on your CUDA version. 
# Visit [https://rapids.ai/start.html](https://rapids.ai/start.html) for instructions.
