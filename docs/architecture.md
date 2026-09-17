# Architecture

## Shared pipeline

1. `remoma.config.load_config` resolves paths and validates configuration basics.
2. `dataset.preprocessing` creates memory-mapped train/validation/test arrays.
3. `graph.adjacency` validates labeled adjacency matrices and canonical node ordering.
4. `models.build_model` selects a registered implementation and its effective parameters.
5. `training.Trainer` optimizes cross-entropy and selects the checkpoint by validation macro F1.
6. Evaluation reports classification metrics; optional decision thresholds are selected on validation only.

The package is organized by responsibility. Model-family changes belong in their own directories; data and evaluation logic remain shared.

## Tensor and label contract

- Feature arrays: `[samples, nodes, channels]`.
- Nodes: `2 * n_levels * (n_lags + 1)`, including lag zero.
- Loader depth: exactly ten levels, with forty raw CSV columns.
- Canonical node order: ask side, bid side; increasing level within each side; increasing lag within each level.
- Base channels: normalized price, quantile-binned volume. Optional engineered channels follow `data.extra_node_features` order.
- GNN positional lag features are appended inside the model and are not stored in the arrays.
- Output: logits `[batch_size, 3]`; class order is down, flat, up.

## Model families

| Model | Temporal processing | Spatial operator | Static tensor training |
| --- | --- | --- | --- |
| `gcn` | Lag nodes and optional lag feature | GCN | Yes |
| `gat` | Lag nodes and optional lag feature | GAT | No |
| `sage` | Lag nodes and optional lag feature | SAGE | No |
| `cgnn`, `cgnn_sage` | Temporal CNN before graph layers | GCN / SAGE | Yes |
| `cgnn_gat` | Temporal CNN before graph layers | GAT | No |
| `stgcn`, `stgcn_sage` | Repeated temporal / graph / temporal blocks | GCN / SAGE | Yes |
| `stgcn_gat` | Repeated temporal / graph / temporal blocks | GAT | No |
| `recurrent_sparse_sthnn` | Shared lag-distance messages and GRU updates | Recurrent message graph | Required |

GAT variants use flattened PyG batches because GATConv does not accept the shared `[B, N, F]` static path used here. Recurrent training automatically selects tensor batching. Standard GNN evaluation uses PyG batches; tests compare this path with static forward results for supported models.

## Registry and parameter precedence

`models/gnn/__init__.py` owns GNN registrations and static-batching capability. `models/recurrent/__init__.py` owns recurrent registrations. The common factory combines the two registries, so a new model in one family does not require changing the other family's implementation.

The order of precedence is constructor default, shared `model` setting, selected `model.overrides.<type>` setting, then CLI overrides written into that selected preset. `--hidden` sets `hidden_channels` for GNNs and `hidden_dim` for recurrent models. `--model` also updates `model.family`; a mismatched YAML family/type is rejected.

The recurrent classifier converts canonical tensor order to adjacency order before inference. It supports `all_lags` and `lag0` readouts, optional synthetic same-feature temporal edges, and optional edge weights. It is not a streaming RNN carrying hidden state between market samples: each lag-window sample initializes its own encoded node states.

## Checkpoint contract

Weights remain a PyTorch state dictionary for compatibility. Every new run also saves `resolved_config.yaml`, `metadata.json`, and a copy of its adjacency. The fitted binner and preprocessing metadata are also copied into the run. Metadata binds weights to the effective model, feature/label settings, graph SHA-256, and fitted-binner SHA-256. Evaluation rejects mismatches when metadata exists.

Legacy state dictionaries require an explicit original configuration. Their graph/configuration provenance cannot be recovered from weights alone. Relocating a new run to another host requires adjusting absolute paths in its saved configuration while retaining equivalent model settings and the same graph bytes.
