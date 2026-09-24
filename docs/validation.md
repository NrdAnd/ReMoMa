# Validation record

This record separates synthetic correctness checks from full-data research results. The dated results below describe checks performed at those revisions; they do not certify subsequent changes automatically.

## Documentation checks — 2026-09-24

All 29 tracked Markdown files passed local-link and heading-anchor checks. All 31 CLI help entry points passed; fixed and walk-forward plans matched the documented settings using filename-only fixtures. The 32 GNN result rows matched the source CSV, and all five graph hashes matched the manifest. Threshold equality/tie rules and order-flow signs matched the implementation. Repository checks passed. No model training or raw-data processing was performed.

## CPU validation — 2026-09-18

**22 tests discovered: 21 passed, one CUDA-only test skipped.** The suite completed in 189.286 seconds on macOS arm64, Python 3.10.20, CPU.

Runtime: PyTorch 2.2.0, PyG 2.7.0, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, PyYAML 6.0.3, tqdm 4.67.3.

| Area | Checks |
| --- | --- |
| Models | Forward/backward for all nine GNNs and recurrent sparse STHNN; supported static/PyG equivalence; extra feature channels |
| Data | Chronological disjoint splits, purged temporal windows, training-only volume bins, independent sampling seeds, exact message-file pairing |
| Graphs | Label ordering, symmetry and finite weights, cache invalidation, training-only source hashes, weighted/unweighted reference topology |
| NMI/TMFG | Full NMI against scikit-learn; relative-lag representatives against full pairs; constant/binary columns; tied and zero-gain TMFG inputs |
| Storage | Indexed/materialized feature equality, including order flow; content-mutation and cache-corruption detection |
| Decisions | Threshold search against brute force, float32 boundaries and ties, zero-threshold labels, metrics with absent classes |
| Checkpoints | Train/save/reload/evaluate, threshold reload, rejection of incompatible settings or graph/binner contents |
| Orchestration | Two-fold GCN/GAT/recurrent experiment, spawned loader worker, expanding/rolling plans, completed-run reuse, changed-setting rejection, graph-only full-NMI stage |
| Aggregation | Pooled fold metrics per seed; seed repetitions excluded from test sample counts |

End-to-end fixtures used five synthetic files of 140 rows, lag depth two, small hidden widths, and one epoch per model/fold. The earlier eleven-test suite also passed on 2026-09-17, including legacy walk-forward, stable-threshold, ensemble, and grid-generation workflows.

## Repository checks — 2026-09-18

Source compilation, configured Ruff checks, notebook cleanliness/syntax checks, YAML reference paths, Markdown links/fences, and whitespace checks passed. All 31 current and compatibility CLI help entry points passed. Wheel and source-distribution builds passed.

The documentation audit checked all 28 Markdown files then tracked, including the pull-request template. The GNN results table matched all 32 source CSV rows. All five reference graphs passed structural checks; hashes and counts are recorded in the [manifest](../data/graphs/manifest.json). The two-parent integration is documented in [merging](merging.md).

## Verification limits

- No full-data CSCO training, NMI computation, or historical performance reproduction was performed in these checks.
- CUDA execution and throughput were not tested. The [opt-in NMI parity test](pipeline.md#verification-boundaries) must run on a CUDA host.
- The Docker recipe, Linux/Python 3.11 CI, and remote deployment were not verified by these macOS runs.
- GPU notebooks were checked structurally but not executed. Historical graph-estimation dates and source-data alignment remain unverified.
- Historical prototypes are outside the supported runtime; their known defects are documented in [archive](../archive/README.md).

Synthetic training verifies implementation behavior, not predictive performance. See [historical reports](reports/README.md) for the scope of earlier measurements.
