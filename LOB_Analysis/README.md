# LOB Similarity Analysis

This repository contains the **`csvAnalytics.ipynb`** notebook, a dedicated tool for exploring and analyzing the dependency structure within the Limit Order Book (LOB). 

The notebook implements a Multi-Lag Dependency Study aimed at constructing and evaluating similarity matrices between market variables over time. This is a crucial preliminary step for Exploratory Data Analysis (EDA) before building more complex graph-based representations (such as GLASSO or TMFG) for downstream modeling.

## 📂 File Contents

- **`csvAnalytics.ipynb`**: A comprehensive Jupyter Notebook for extracting and analyzing similarity metrics. Key sections and features include:
  - **GPU Acceleration**: Leverages the RAPIDS suite (`cuDF`, `CuPy`) to efficiently handle and process massive amounts of high-frequency LOB data.
  - **LOB Relationship Metrics**: Calculates the means and distributions of dependencies among different book components, categorizing them into:
    - *Ask-Ask* and *Bid-Bid* (same-side interactions).
    - *Ask-Bid* (cross-side interactions).
    - *Same Level* / *Different Level* (impact between identical or different price/volume levels).
  - **Temporal Analysis**: Studies "Temporal Persistence" (the behavior of the same node across different lags) and compares *Same Lag* vs. *Different Lag* dynamics.
  - **Reporting**: Generates styled pivot tables ("Summary Tables") and concise visualizations ready for interpretation.

## 🛠️ Prerequisites

Because this code is designed to operate on large datasets via GPU, a specific **Conda** environment is required.

You can install the necessary dependencies by running the following command (which is also included at the beginning of the notebook):

```bash
conda install -y -c rapidsai -c conda-forge -c nvidia \
    cudf=26.04 cupy pandas matplotlib seaborn tqdm jinja2 ipython \
    python=3.11 cuda-version=12