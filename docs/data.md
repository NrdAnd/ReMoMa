# Data and scientific contracts

## Raw files

Each LOBSTER orderbook CSV has no header and exactly forty columns. For zero-based level `i`, columns `4*i` through `4*i+3` contain ask price, ask volume, bid price, and bid volume. Prices must be finite and positive. Malformed rows are rejected instead of silently skipped, which would break event alignment.

Raw prices use the integer-scaled LOBSTER representation. For the historical CSCO experiments, a USD 0.01 tick corresponds to 100 raw units. The loader does not convert prices to dollars.

Message data uses time, event type, order ID, size, price, and direction in the first six columns. Order-flow features require exact day-prefix matching and equal row counts. The file loader cannot establish the provenance of already modified files; retain the original paired exports.

## Labels

For reference price `p` and horizon `k`, define `difference = p[t+k] - p[t]`. The reference is either best-level mid-price or volume-weighted microprice.

- `label_mode: abs`: compare the raw price difference with `threshold`.
- `label_mode: pct`: compare `difference / p[t]` with `threshold`.
- Positive differences at or above the threshold are up (2).
- Negative differences at or below minus the threshold are down (0).
- Other samples are flat (1), including unchanged prices when threshold is zero.

The horizon counts events, not seconds. These are point-to-point labels; they are not future-window averages. Different label definitions are different prediction tasks.

## Features

The two base channels are relative price `(price_at_lag - current_mid) / current_mid` and global quantile-binned volume. `current_mid` is the best-level mid-price at the prediction origin; the same reference is used for all nodes in that window. `normalize_prices: false` preserves raw prices and generally requires `feature_dtype: float32` to avoid float16 overflow.

Optional channels are `spread`, `level_imbalance`, `depth_imbalance`, `microprice`, `order_trade_flow`, `order_limit_flow`, `order_cancel_flow`, and `side`. Order-flow channels use signed event size `q` and resting-order direction `d` (+1 bid, −1 ask): trade `−q*d` for types 4/5, limit `q*d` for type 1, and cancel `−q*d` for types 2/3. At each lag, each flow is divided by total ten-level depth, clipped to [−5, 5], and repeated across nodes at that lag; it is not a window sum.

Feature order is part of the checkpoint contract. Duplicate or unknown feature names are rejected. The recurrent and GNN constructors both derive their input width from this list.

Volume quantiles are fitted only on rows covered by training input windows, before class balancing or sample caps. Raw non-finite/negative volumes and non-finite prices/features are rejected. Store data-cleaning provenance separately from model configuration.

## Temporal splits

`by_file` uses disjoint, chronological file groups. Indices are zero-based positions in the lexically sorted file list. Duplicate, overlapping, out-of-range, empty, or nonchronological groups are rejected.

`by_lag` divides each file chronologically according to train/validation ratios. A sample uses raw rows `[t-n_lags, t+k]`. The implementation removes `n_lags + k` candidate samples before each boundary so adjacent splits do not share input or target rows. Short files may therefore produce empty splits and fail validation.

`by_lag` still uses the same trading days across all three subsets and does not measure generalization to a new day. The primary GNN example defaults to `by_file`. Existing recurrent exploratory presets retain their explicit split choices.

`data.subsample_seed` determines balanced training and validation/test subsamples. `training.seed` determines initialization and training randomness independently. Validation/test retain their natural class distributions unless a cap selects a random subset. A fixed training seed alone is not a guarantee of bitwise identical results across hardware or library versions.

## Processed cache

`data.processed_dir` contains `X_*.npy`, `y_*.npy`, `binner.pkl`, `meta.json`, and an optional legacy `price_stats.npy`. GNN commands also cache `edge_index.pt`.

The metadata signature covers preprocessing schema, storage mode, dimensions, feature order, labels, split settings, sampling caps, and sampling seed. Raw/message manifests contain filenames and SHA-256 hashes. Compatibility also checks tensor shapes and dtypes. Hashing raw inputs adds I/O during cache checks. Indexed storage additionally verifies the content hashes of raw binary arrays, pre-binned volumes, flow arrays, sample indices, labels, and binner. Materialized feature-array contents are checked by shape/dtype rather than a complete hash.

Schema version 3 distinguishes indexed and materialized storage and invalidates earlier caches. Version 2 introduced temporal purging and corrected scaler-fit boundaries. Metadata is removed before rebuilding, so an interrupted rebuild is not reusable. The complete pipeline locks cache writes; direct lower-level preprocessing commands must not write concurrently to the same directory.

With `storage_mode: materialized`, `X_*.npy` contains `[samples, nodes, channels]` tensors. With `storage_mode: indexed`, it contains int32 `[file_index, event_index]` rows; metadata identifies shared raw arrays and fold-specific pre-binned volumes. Dataset readers construct the same feature tensors batch by batch. `data.raw_files` freezes the exact chronological file list, preventing newly discovered files from shifting split indices. `data.raw_cache_dir` shares parsed CSV arrays between folds.

Raw inputs are required for verification by default. `data.allow_missing_raw: true` explicitly permits use of complete compatible caches when raw files are unavailable. In that mode, raw-content provenance cannot be rechecked. Keep this exception limited to controlled evaluation deployments.

## Graphs and provenance

CSV and TSV adjacency files must have identical unique row/column labels, square finite nonnegative weights, and symmetry. The pipeline also writes labeled compressed COO `.npz` graphs, with the same validation rules. Both label conventions are accepted:

- Legacy: `ask_0_lag_0`, `bid_0_lag_0` (zero-based levels).
- Canonical: `ASKs1_lag0`, `BIDs1_lag0` (one-based levels).

GNN edges are reordered to canonical feature order; nonzero weights become binary connectivity. Recurrent models retain matrix order internally and can use the weights when enabled. Graph caches are bound to graph content and requested dimensions.

The versioned graphs are historical reference inputs. Their filenames describe transformations but do not establish which dates were used to estimate the similarity matrix. A held-out test day does not eliminate leakage if the graph was estimated using that day. For a defensible evaluation, estimate similarities and graphs from training data only, freeze them before validation, and record source dates and hashes. The older matrix-conversion scripts accept a supplied matrix and cannot infer its temporal provenance. The [complete pipeline](pipeline.md) instead constructs each graph from its declared training dates, records hashes, and never uses the historical reference graphs for its runs.
