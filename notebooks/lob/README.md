# LOB similarity analysis

`similarity_analysis.ipynb` explores lagged dependence among bid/ask volume levels and constructs similarity matrices. `similarity_experiment.ipynb` retains an alternative research experiment.

Both notebooks require their own input settings. GPU cells use CuPy/cuDF; the core model environment does not install RAPIDS. Follow the [notebook guidance](../README.md) and [setup](../../docs/setup.md).

For model evaluation, compute graph-building similarities on training data only. The notebooks do not infer the intended held-out period automatically.
