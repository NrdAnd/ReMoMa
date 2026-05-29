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

## Label Distribution Analysis

Use the utility below to inspect class imbalance across horizon/threshold/price settings:

```bash
python scripts/analyze_label_distribution.py --config config/default.yaml
```

Default grid:

- `k`: `1,5,10,20,50,100,200`
- `threshold`: `0,1e-5,5e-5,1e-4`
- `price_type`: `mid,micro`
