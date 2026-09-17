# Integration validation record

Date: 2026-09-17. Platform: macOS arm64, Python 3.10.20, CPU. Runtime: PyTorch 2.2.0, PyG 2.7.0, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, PyYAML 6.0.3, tqdm 4.67.3.

## Completed correctness checks

The eleven-test synthetic suite passed. It covered:

- Forward and backward computation for all nine GNN architectures and recurrent sparse STHNN, including extra feature channels.
- Static/PyG output equivalence for supported architectures.
- Graph-label reordering, graph-content cache invalidation, and invalid graph rejection.
- Disjoint file splits and nonoverlapping temporal input/target windows.
- Training-only volume-bin fitting.
- Independent sampling/initialization seeds and raw-content cache invalidation.
- Exact orderbook/message filename pairing.
- Zero-threshold labels and metrics with absent classes.
- Checkpoint rejection when model settings or graph content change.
- A single small walk-forward fold, stable-threshold evaluation, a two-member seed ensemble, and hyperparameter-grid configuration generation.
- A complete synthetic train/save/reload/evaluate cycle for GCN, GAT, and recurrent models, with threshold reload and a spawned GAT data-loader worker.

The complete-cycle tests used five synthetic files of 140 rows each, lag depth 2, small hidden widths, and one epoch. They completed as part of a roughly 62-second suite. **No training was run against the real local LOBSTER files.** Additional small synthetic checks were explicitly authorized by the owner. Full-data training remains reserved for remote compute.

The sandbox initially denied OpenMP shared-memory access in subprocesses. Repeating the same small synthetic suite with the required process permissions passed; this was an environment restriction, not a failed model assertion.

## Reference inputs and repository checks

All five distributed graph matrices were loaded and checked for matching unique labels, dimensions, finite nonnegative weights, and symmetry. The reconstructed weighted graph's nonzero topology matches the existing unweighted lag-100 reference. Exact hashes and counts are in the [graph manifest](../data/graphs/manifest.json).

Source compilation, undefined-name lint checks, notebook cleanliness/syntax checks, YAML reference-path checks, and Markdown link/fence checks passed. All 30 current/compatibility CLI help entry points passed. Both wheel and source-distribution builds passed locally. The source tips and recorded merge parents are documented in [merging](merging.md).

## Limits

- Full-data training, predictive performance reproduction, CUDA execution, and remote deployment were not performed.
- GPU research notebooks were not executed. Notebook outputs were cleared and source text reviewed; dataset-specific paths still require configuration.
- The Docker recipe and Linux/Python 3.11 CI are provided but require execution on their respective environments. Local macOS results do not certify those platforms.
- Raw data and historical graph-estimation dates were not independently reconstructed or certified.
- Historical prototypes are outside the supported runtime and retain documented defects for research history.
- No branch was pushed and no GitHub pull request or release was published during preparation.

Use remote compute for full model training. The local regression suite's synthetic training is deliberately small; it is still an execution of training code, not merely a static check.
