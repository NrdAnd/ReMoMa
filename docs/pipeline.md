# End-to-end CSCO pipeline

The supported entry point is `scripts/run_pipeline.py`. It selects chronological files first, computes daily NMI exclusively from the training dates, builds that fold's TMFG, preprocesses the data, trains each requested model and seed, calibrates thresholds on validation, and evaluates the frozen model and decision rules on test.

The default specification is [configs/pipeline/cisco.yaml](../configs/pipeline/cisco.yaml). All code, inputs, settings, graphs, checkpoints, and final results have recorded identities. The older `run_walk_forward.py` accepts a pre-existing graph; it does **not** provide this graph-construction guarantee.

## Installation and input files

Use the model environment described in [setup](setup.md). The maintained pipeline uses NumPy/pandas on CPU and optionally CuPy for NMI; it does not require running Jupyter or installing cuDF/RAPIDS.

```bash
python -m pip install -e '.[dev]'
```

On a Linux compute host with a compatible NVIDIA driver and CUDA 12 toolkit, install the optional CuPy backend:

```bash
python -m pip install -e '.[nmi-cuda]'
python -c "import cupy; cupy.show_config()"
```

The extra pins CuPy 13.6.0. Its CUDA 12 wheel, supported Python versions, and toolkit requirements are documented in the [official installation guide](https://docs.cupy.dev/en/v13.6.0/install.html). PyTorch also needs a compatible CUDA build for GPU model training; follow [setup](setup.md). `--nmi-backend cuda` fails if CUDA NMI is unavailable; `auto` selects CuPy when available and otherwise uses CPU. NMI backend selection does not change PyTorch's device selection.

Place these orderbook files under `data/raw/`, or provide `--raw-dir /absolute/path/to/data`:

```text
CSCO_2019-01-22_34200000_57600000_orderbook_10.csv
CSCO_2019-01-23_34200000_57600000_orderbook_10.csv
CSCO_2019-01-24_34200000_57600000_orderbook_10.csv
CSCO_2019-01-25_34200000_57600000_orderbook_10.csv
CSCO_2019-01-28_34200000_57600000_orderbook_10.csv
```

There must be exactly one selected-ticker orderbook per date. Names and session bounds are validated. Other tickers are excluded. The default price/volume features and price-movement labels use orderbook files. Precisely paired `*_message_10.csv` files are required only when order-flow features are enabled; preprocessing checks equal row counts. These checks cannot establish event alignment after source files have been modified; retain the original paired exports. No sample window or prediction horizon crosses a daily file boundary.

## Inspect the plan without executing

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --mode walk_forward
```

The selected raw files must already exist, even for plan inspection. Without `--run`, the command prints dates, models, seeds, and NMI settings. It does not read the contents of raw data, create an experiment, compute NMI, or train models.

The supplied specification compares CGNN and recurrent sparse STHNN at the same lag depth of 100, horizon of 50 events, absolute price threshold of 100 raw LOBSTER units, and seeds 42/43/44. It uses all valid train/validation/test samples, class-weighted cross entropy, float32 features, and each family's architecture and batch-size settings from its base configuration. These are explicit benchmark settings, not a claim that the hyperparameters are optimal. There is no implicit hyperparameter search.

## Fixed by-file experiment

| Split | Dates |
| --- | --- |
| Train; NMI and TMFG estimation | 2019-01-22, 2019-01-23, 2019-01-24 |
| Validation; early stopping and decision thresholds | 2019-01-25 |
| Test; final metrics | 2019-01-28 |

Run on the compute host:

```bash
python scripts/run_pipeline.py --run --nmi-backend cuda
```

For a CPU compute host, use `--nmi-backend cpu`. Full datasets can be expensive even though the relative estimator reduces NMI work.

## Walk-forward experiment

```bash
python scripts/run_pipeline.py --mode walk_forward \
  --output-dir runs/experiments/cisco_walk_forward \
  --nmi-backend cuda --run
```

| Fold | Train; graph estimation | Validation | Test |
| --- | --- | --- | --- |
| 001 | Jan 22, Jan 23 | Jan 24 | Jan 25 |
| 002 | Jan 22, Jan 23, Jan 24 | Jan 25 | Jan 28 |

Each fold gets its own TMFG and training-fitted volume binner. Models are initialized from their specified seeds independently in every fold; there is no implicit weight transfer. An earlier validation or test day may become historical training/validation data later. Only information available before the current test period is used. Test blocks do not overlap; `step` must be at least `test_files`.

Set `walk_forward.train_window` to a positive number for a rolling training window, or leave it null for expanding training. A rolling window must be at least `min_train_files`. `validation_files`, `test_files`, `step`, and `max_folds` are configurable.

This is a predeclared walk-forward protocol. Do not revise model choices or settings based on earlier test results and then describe the complete set of test results as untouched. Validation is shared between early stopping and threshold calibration; a separate calibration set or nested validation would be a different experimental protocol.

## Model, seed, and NMI selection

```bash
# Choose a subset or any supported GNN/recurrent architecture.
python scripts/run_pipeline.py --models gcn gat cgnn recurrent_sparse_sthnn \
  --seeds 42 43 44 --output-dir runs/experiments/cisco_four_models --run

# Full pairwise NMI, with a distinct experiment identity.
python scripts/run_pipeline.py --nmi-method full --nmi-backend cuda \
  --output-dir runs/experiments/cisco_full_nmi --run

# Build and verify graphs without preprocessing or training.
python scripts/run_pipeline.py --stage graphs --run
```

Copy the specification to `configs/local/` for custom settings. Model implementations are chosen through `models`, family configurations through `model_configs`, and common data/training changes through their explicit override sections. The runner rejects unknown pipeline keys, overlapping/out-of-order splits, and comparisons with different lag depths, label definitions, or evaluation sampling. Runner-owned graph paths and split assignments cannot be overridden through a model's historical adjacency setting.

## NMI definition and precision

Daily input variables are the ten ask-size and ten bid-size series. For maximum lag L and a day of N rows, every shifted column contains N-L aligned observations. Quantile edges are fitted separately to each shifted column using float64 linear interpolation, then duplicate edges are removed. Constant columns remain constant; binary columns retain their two states. This corrects the old notebook's degenerate binary-column collapse. Quantile discretization is distinct from the model's global training-volume binner.

The estimator uses integer marginal/joint counts and arithmetic normalization, `NMI = 2 MI / (H(X) + H(Y))`. Two constant columns return 1, and a constant/nonconstant pair returns 0. Floating-point reductions use float64. Minor rounding outside [0,1] is clipped only within a checked tolerance. The estimator uses the same definition in CPU and GPU modes; bit-for-bit cross-device equality is not promised. Linear quantile interpolation is defined in the [NumPy reference](https://numpy.org/doc/1.26/reference/generated/numpy.quantile.html).

`relative` is the default. It stores `M[delta, base_i, base_j]` for nonnegative delta. Its canonical representative compares `base_i` at lag delta with `base_j` at lag zero. Negative differences use `M[delta, base_j, base_i]`, preserving temporal direction. The full lagged matrix is reconstructed only when building TMFG. The approximation assumes that dependence is primarily a function of relative lag; it does not guarantee equality with full pairwise estimates on finite, nonstationary data. The production implementation retains this approximation but uses float64 reductions, symmetry reuse, and corrected binary handling, so historical float32 outputs are not promised to be bit-identical.

`full` independently estimates every unordered pair of lagged columns. It needs much more computation and temporary disk storage for discrete columns. It is a complete calculation of the chosen empirical estimator, not knowledge of the true population mutual information.

Daily matrices receive equal weight, as in the historical mean-NMI procedure. This is not pooled estimation on concatenated days and does not weight days by event count. The NMI calculation uses the complete training day, including rows at its end that cannot themselves serve as supervised prediction origins. All such rows precede validation. No validation/test row enters NMI estimation, discretization for graph construction, or the daily average.

Differences between NMI estimates and differences between the resulting TMFGs are distinct: even small changes in nearly tied weights can change selected edges. The method, inputs, counts, backend, and hashes are recorded rather than assuming a universal approximation-error bound.

## Execution and storage optimizations

- Daily NMI is cached by input content, estimator settings, backend/library version, and implementation hash. It is shared across compatible folds, models, and initialization seeds.
- Expanding folds extend a running float64 sum with new training days. Rolling or changed windows reconstruct the sum from cached daily results. TMFG is rebuilt for each distinct training set.
- Relative mode needs `B(B-1)/2 + L B²` NMI calculations rather than `B(L+1)[B(L+1)-1]/2`. At B=20 and L=100 this is 40,190 versus 2,039,190 calculations; these counts are not wall-clock speedup guarantees.
- GPU residence in relative mode is bounded to B anchor columns, one shifted column, and histogram workspaces. Full mode uses disk-backed uint16 discrete columns and two GPU columns at a time. Neither mode constructs a float-valued lag tensor.
- Joint histograms use dense counting when economical and observed-category counting otherwise. Both compute the same statistic. Reused marginal histograms avoid repeated work.
- Relative NMI is saved as a compact float64 binary array, approximately 0.31 MiB at L=100, instead of a dense 31.1 MiB similarity matrix. Graphs use compressed labeled COO `.npz` files; CSV zero entries are not replicated.
- Parsed raw arrays are shared read-only across folds. Default `indexed` preprocessing stores raw-row references, labels, and pre-binned volumes. Overlapping windows are assembled vectorially for the current batch, including the original normalization and engineered features. Binning is refitted for every distinct training split. This trades some batch-time CPU work for substantially less disk storage and preprocessing time.
- `data_overrides.storage_mode: materialized` remains available when precomputing every window offers better throughput on the target machine. It has a much larger storage footprint. Choose based on measured end-to-end throughput, including preprocessing, rather than assuming either mode always wins.
- Threshold search uses exact two-dimensional cumulative counts, reducing repeated scans of validation predictions. Test predictions are collected once and used for both argmax and the already frozen validation-selected rule.
- Model jobs run sequentially to bound GPU/host memory. Increase `training_overrides.num_workers` only after measuring the host; zero avoids unnecessary worker overhead. No mixed precision or silent sampling is enabled by the supplied benchmark.

Content hashes require I/O. They are intentionally retained for reliable cache validation. Cache locks reject concurrent writes. No universal minimum runtime or storage usage is claimed; measurements depend on data, hardware, and selected models.

CuPy's cached GPU allocations are released after daily NMI computation before launching the model-training process. No GPU process is started by a plan-only invocation.

## Outputs and resuming

```text
runs/cache/
  nmi/<identity>/values.npy, manifest.json
  graphs/<identity>/adjacency.npz, manifest.json
  raw_v1/<source-hash>/orderbook.npy, manifest.json
  processed/<identity>/...             Shared by compatible models and seeds
runs/experiments/<name>/
  manifest.json                       Frozen inputs, settings, source hashes, environment
  fold_001/
    split.json
    graph_<model>.json                Graph reference and training provenance
    configs/<model>_seed_<seed>.yaml
    <model>/seed_<seed>/
      best.pt, resolved_config.yaml, metadata.json
      adjacency.npz, graph_manifest.json, binner.pkl, preprocessing_meta.json
      calibration.json, thresholds.json, metrics.csv
      results/metrics.json, summary.csv, <report>.txt
      complete.json                  Artifact integrity and completion record
  results.csv                         Per-fold, per-seed, per-rule test metrics
  summary.json                        Pooled folds per seed and across-seed statistics
  results.md                          Human-readable comparison
  complete.json                      Final completion record
```

`calibration.json` is written after selecting thresholds on validation and before test prediction. `thresholds.json` additionally includes test metrics after evaluation. When threshold tuning is disabled, only argmax results are produced.

Repeat the **same command and settings**, adding `--resume`, to reuse a completed graph-only stage or skip completed model runs:

```bash
python scripts/run_pipeline.py --nmi-backend cuda --run --resume
```

Inputs, settings, numerical environment, and source code must match the frozen experiment. Changing them requires a new `output_dir`; compatible shared caches remain reusable. An interrupted model is not marked complete. Add `--restart-incomplete` with `--resume` to archive that attempt and restart only that model from its seed. Optimizer/epoch-level training resumption is not implemented. Archived attempts are retained for inspection and can be removed manually.

Do not modify inputs/code during a run. After a forced process termination, inspect the PID/host in any remaining lock and verify the writer is gone before removing that specific lock. Never remove another active job's lock. Cache directories contain references needed by indexed evaluation; keep them with the experiment or regenerate preprocessing from the recorded inputs. The runner does not automatically delete shared caches or checkpoints.

For each model/rule/seed, test confusion matrices are pooled across nonoverlapping fold test periods; metrics are then recomputed from the pooled counts. Across seeds, report the mean and sample standard deviation of these metrics. A single seed has no estimated standard deviation. Seeds are repetitions on the same data, not additional independent test samples. Per-fold values remain available; sample standard deviations are not confidence intervals.

## Verification boundaries

The local regression suite uses tiny synthetic files and one-epoch models. It compares full NMI against an independent implementation, relative representatives against full pairs, indexed features against materialized features, and optimized threshold search against brute force. It checks train-only graph provenance, cache mutation detection, zero-gain TMFG termination, both model families, walk-forward execution, resume behavior, and seed aggregation. See the current [validation record](validation.md) for results.

Full CSCO model training, CUDA throughput, CPU/GPU numerical parity on the target hardware, and historical benchmark reproduction require execution on the compute host. Small tests establish implementation checks, not predictive performance or a universal bound on approximation error.

An opt-in CPU/CUDA NMI parity test is provided for that host:

```bash
PYTHONPATH=tests REMOMA_TEST_CUDA=1 python -m unittest \
  test_complete_pipeline.NMITests.test_cuda_parity -v
```

It uses 100 synthetic rows and tests both estimators at relative tolerance `1e-9` and absolute tolerance `1e-11`. Without the environment flag it is skipped, so CPU CI cannot silently claim GPU validation.
