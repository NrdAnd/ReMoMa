# TMFG - Triangulated Maximally Filtered Graph Analysis

This repository contains the implementation and analysis tools for the **TMFG (Triangulated Maximally Filtered Graph)** algorithm. This method is used to extract the most significant information from complex correlation or similarity matrices (e.g., financial Limit Order Book data) by filtering noise and maintaining a planar triangulated structure.

## 📂 Repository Structure

The folder consists of two main components:

1.  **`TMFG_core.py`**: 
    A core Python module containing the `TMFG` class implementation. The algorithm starts with an initial 4-clique and iteratively adds vertices to maximize a specific gain criterion while maintaining planarity.
    - **Supported Output Modes**: 
        - `LOGO`: Local-Global Shrinkage estimator for sparse inverse covariance matrices.
        - `WEIGHTED_SPARSE_W_MATRIX`: A weighted adjacency matrix based on the input similarity.
        - `UNWEIGHTED_SPARSE_W_MATRIX`: A binary adjacency matrix representing the filtered network topology.

2.  **`build_graph_tmfg.ipynb`**: 
    A comprehensive Jupyter Notebook that demonstrates the practical application of the TMFG algorithm.
    - **Data Processing**: Loads similarity matrices (e.g., NMI - Normalized Mutual Information) and handles time-lagged redundancies.
    - **Network Analysis**: Calculates key topological metrics such as Degree Centrality, Betweenness Centrality, Average Path Length, and Clustering Coefficients.
    - **Visualization**: Generates interactive graph visualizations using `PyVis` and static plots with `NetworkX` and `Seaborn`.

## 🛠️ Prerequisites

Ensure you have Python installed. You can install the required dependencies using the following command:

```bash
pip install numpy pandas networkx matplotlib seaborn tqdm pyvis