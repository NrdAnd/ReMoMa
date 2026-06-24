# GNN Pipeline (Precomputed + Memory-Mapped)

This module now uses an offline preprocessing stage so the training loop no longer rebuilds LOB node features sample-by-sample.

## What Changed

- Static TMFG graph stays unchanged (`edge_index.pt`).
- Label semantics stay unchanged (`0=DOWN, 1=FLAT, 2=UP`).
- Split semantics stay unchanged (`by_file` / `by_lag`).
- Expensive feature construction is moved offline into `.npy` tensors.

## Processed Artifacts

The preprocessing script writes:

```text
gnn/data/processed/
  X_train.npy
  y_train.npy
  X_val.npy
  y_val.npy
  X_test.npy
  y_test.npy
  edge_index.pt
  binner.pkl
  price_stats.npy   (legacy compatibility artifact)
  meta.json
```

`X_*.npy` are generated as memory-mappable arrays (default `float16` on disk).  
`FastLOBDataset` loads with `mmap_mode="r"` and casts to `float32` at runtime.

## Usage

1. Build or refresh processed tensors:

```bash
cd gnn
python scripts/preprocess_dataset.py --config config/default.yaml
```

2. Train from precomputed tensors:

```bash
python scripts/train.py --config config/default.yaml
```

Optional: enable GCN-only static tensor batching (no PyG `Data` objects in the
train loop) by setting:

```yaml
training:
  static_graph_batching: true
```

If `model.type` is not `gcn`, training automatically falls back to the
standard PyG batching path.

3. Evaluate checkpoint on precomputed test split:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config/default.yaml
```

## Recurrent Sparse STHNN Experiment

The recurrent sparse lag-distance model is integrated as a separate model type,
with its own 100-lag config and processed cache. The labeled adjacency currently
used by the experiment is converted from the already-built project TMFG graph:

```text
tmfg/cisco_tmfg_adj_matrix_2000_bins.csv
```

and written to:

```text
gnn/data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_full_tmfg_adjacency.tsv
```

To rebuild that adjacency from the existing TMFG CSV:

```bash
python scripts/build_recurrent_adjacency_from_tmfg.py
```

If the original NMI source is available, the TMFG can also be rebuilt directly
from:

```text
csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv
```

using:

```bash
python scripts/build_recurrent_tmfg_adjacency.py
```

```bash
python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn.yaml
python scripts/train.py --config config/recurrent_sparse_sthnn.yaml
python scripts/evaluate.py --checkpoint checkpoints/recurrent_sparse_sthnn/best.pt --config config/recurrent_sparse_sthnn.yaml
```

Its adjacency must be a labeled CSV/TSV matrix whose index and columns match
labels such as `ASKs1_lag100`. The model internally reorders the existing
processed LOB tensor order to match that adjacency order.

For cloud runs, keep the same relative layout:

```text
ReMoMa/
  lobster_cisco/                         # raw *_orderbook_10.csv files
  tmfg/
    cisco_tmfg_adj_matrix_2000_bins.csv
  gnn/
    data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_full_tmfg_adjacency.tsv
```

## Label Distribution Analysis

Use the utility below to inspect class imbalance across horizon/threshold/price settings:

```bash
python scripts/analyze_label_distribution.py --config config/default.yaml
```

Default grid:

- `k`: `1,5,10,20,50,100,200`
- `threshold`: `0,1e-5,5e-5,1e-4`
- `price_type`: `mid,micro`
