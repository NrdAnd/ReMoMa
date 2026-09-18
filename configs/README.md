# Experiment configurations

Configurations are complete YAML mappings; there is no implicit inheritance. Paths in them resolve from the repository root, independently of the YAML file's location.

`pipeline/cisco.yaml` is a separate orchestration specification. It explicitly loads one complete base configuration per model family and applies common data/training overrides. See the [complete pipeline guide](../docs/pipeline.md) for dates, NMI methods, cache rules, execution, and resumption.

## Available presets

| Preset | Purpose |
| --- | --- |
| `gnn/default.yaml` | Nine GNN presets; lag 150; chronological by-file evaluation |
| `gnn/test.yaml` | Historical alternative GNN experiment; inspect its values before use |
| `recurrent/recurrent_sparse_sthnn.yaml` | Lag 100, basic channels, exploratory by-lag split |
| `recurrent/recurrent_sparse_sthnn_by_file.yaml` | Lag 100 with held-out files |
| `recurrent/recurrent_sparse_sthnn_lag150.yaml` | Lag 150, by-lag split |
| `recurrent/recurrent_sparse_sthnn_lag150_by_file.yaml` | Lag 150 with held-out files |
| `recurrent/recurrent_sparse_sthnn_feature_rich.yaml` | Additional engineered node channels |
| `recurrent/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml` | Engineered/order-flow channels and weighted messages |
| `recurrent/recurrent_sparse_sthnn_feature_rich_unweighted_walkforward.yaml` | Unweighted feature-rich baseline for fold generation |

## Configuration keys

| Section | Keys | Meaning |
| --- | --- | --- |
| `data` | `raw_dir`, `message_dir`, `message_pattern` | Input files; message settings are needed for order-flow channels |
| `data` | `adj_matrix_path`, `processed_dir` | Versioned/source graph and generated feature-cache directory |
| `data` | `n_lags`, `n_levels`, `n_volume_bins` | History depth, ten-level loader, volume quantiles |
| `data` | `price_type`, `label_mode`, `threshold`, `prediction_horizon` | Exact prediction target |
| `data` | `split_strategy`, `train_files`, `val_files`, `test_files` | Chronological file split |
| `data` | `train_ratio`, `val_ratio` | Within-file ratios for purged by-lag splits |
| `data` | `max_samples_per_class`, `max_val_samples`, `max_test_samples` | Optional positive sampling caps; null means no cap |
| `data` | `subsample_seed` | Sampling seed independent of initialization |
| `data` | `normalize_prices`, `extra_node_features`, `feature_dtype` | Feature representation and disk precision |
| `data` | `use_precomputed`, `force_preprocess`, `preprocess_chunk_size` | Offline feature-cache behavior; training requires precomputed tensors |
| `data` | `raw_files`, `raw_cache_dir`, `storage_mode` | Frozen input list, shared parsed-CSV cache, `indexed` or `materialized` windows |
| `data` | `allow_missing_raw` | Explicit deployment exception; defaults to false |
| `model` | `family`, `type`, `overrides` | Family validation, selected architecture, per-architecture settings |
| GNN settings | `hidden_channels`, `num_layers`, `dropout`, `num_heads`, `add_lag_feature` | Graph architecture and positional input |
| CGNN/STGCN settings | `cnn_channels`, `cnn_kernel`, `use_bin` | Temporal convolution; use an odd positive kernel to preserve shape |
| Recurrent settings | `hidden_dim`, `message_iterations`, `readout_mode`, `readout_dropout` | Hidden width, recurrent updates, and classifier readout |
| Recurrent settings | `add_self_lag_edges`, `self_lag_edge_dropout` | Optional synthetic same-feature temporal connections |
| Recurrent settings | `use_edge_weights`, `edge_weight_normalization` | Weighted messages; normalization `none`, `mean`, or `max` |
| `training` | `epochs`, `batch_size`, `lr`, `weight_decay` | Optimization |
| `training` | `early_stopping_patience`, `lr_scheduler_patience`, `lr_scheduler_factor`, `grad_clip` | Stopping, learning-rate schedule, gradient clipping |
| `training` | `use_class_weights`, `tune_threshold`, `num_workers`, `seed`, `static_graph_batching` | Loss weighting, evaluation, loading, randomness, batching |
| `training` | `threshold_search`, `deterministic` | Validation-only threshold-grid settings; opt-in strict deterministic kernels |
| `paths` | `checkpoints` | Parent of automatically created run directories |

`model.adjacency_path` can override the recurrent adjacency. Prefer `data.adj_matrix_path` for a single explicit graph source. `data.node_feature_dim`, when supplied, must equal `2 + len(extra_node_features)`.

Copy a preset to `configs/local/` for private experiments. Keep `model.type`, graph dimensions, feature channels, and family consistent. Complete key semantics are documented in [data](../docs/data.md) and [architecture](../docs/architecture.md).
