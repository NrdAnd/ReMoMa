# Graphical Lasso experiments

`glasso_v4_lag20.ipynb` and `glasso_v5_lag20.ipynb` retain two exploratory pipelines for estimating sparse precision-matrix networks. `lag_comparison.ipynb` compares graph behavior across lag settings.

Inspect each notebook's feature construction and parameter cells as the source of truth. Historical version names do not define a stable API. GPU preprocessing and interactive plotting require the dependencies described in the [notebook guidance](../README.md).

The maintained training scripts currently consume a supplied labeled adjacency. They do not automatically run these GLASSO notebooks.
