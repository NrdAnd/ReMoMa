# Integration validation record

## Documentation consistency audit — 2026-09-18

All 28 tracked Markdown documents, including the pull-request template, were checked for local links, closed code fences, current paths, configuration names, CLI commands, preprocessing schema, generated artifacts, model selection, pipeline behavior, and branch-history instructions. The repository checker was expanded to include hidden tracked documentation while excluding local build/test caches. All 31 current and compatibility command help entry points passed, and the historical GNN table matched all 32 rows in its source CSV. Documentation now describes the main-only remote layout while preserving the two-parent integration proof by immutable commit ID. A tracked-file inventory found no raw-data paths or common credential filenames; this is a targeted publication check, not a substitute for GitHub secret scanning. This audit ran no model training and did not process raw CSCO contents.

## End-to-end pipeline validation — 2026-09-18

The expanded suite completed in 189.286 seconds: **22 tests discovered, 21 passed, one CUDA-only test explicitly skipped**. The local platform and numerical environment are the CPU environment recorded below. No training or NMI estimation was performed on the real CSCO files.

Additional checks covered:

- Full pairwise NMI against scikit-learn's arithmetic-normalized reference, relative-lag representatives against the corresponding full pairs, signed-lag reconstruction, constant/binary columns, and sparse joint counting.
- TMFG termination and topology for zero-gain and tied similarity matrices.
- Graph estimation from training files only, training-source mutation detection, and reuse unaffected by changes to held-out files.
- Exact equality between indexed and materialized features, including order-flow channels, and indexed-cache corruption detection.
- Optimized threshold counts and selected objective values against brute-force evaluation, including float32 boundary comparisons and ties.
- Expanding and rolling date plans, nonoverlapping test periods, and rejection of incompatible model comparison settings.
- A complete two-fold experiment with GCN, GAT, and recurrent sparse STHNN, one epoch per model/fold, and a spawned data-loader worker; saved GAT evaluation; completed-run reuse without launching training again; changed-setting rejection; and a separate full-NMI graph-only stage.
- Aggregation across folds within each seed, without counting repeated seeds as additional test observations.

The synthetic end-to-end fixtures use five files of 140 rows, lag depth two, and small hidden widths. CLI plan inspection on the local CSCO filenames performs no raw-content processing. CUDA numerical parity and full-data performance remain untested locally; the [pipeline guide](pipeline.md#verification-boundaries) provides an explicit, small opt-in CUDA parity check for the compute host.

Final source compilation, configured Ruff checks, and whitespace checks passed. The repository checker validated 70 Python files, six notebooks, ten configurations, and 26 Markdown documents. Both wheel and source-distribution builds passed. The default walk-forward plan resolved the intended five CSCO dates without processing their contents.

The following sections retain the earlier integration validation record and its historical scope.

## Earlier integration validation — 2026-09-17

Platform: macOS arm64, Python 3.10.20, CPU. Runtime: PyTorch 2.2.0, PyG 2.7.0, NumPy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, PyYAML 6.0.3, tqdm 4.67.3.

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
- At the time of this earlier validation, no branch had been pushed and no GitHub pull request or release had been published. This records that validation session, not the repository's later hosting state.

Use remote compute for full model training. The local regression suite's synthetic training is deliberately small; it is still an execution of training code, not merely a static check.
