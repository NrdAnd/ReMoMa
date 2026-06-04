# LOB-GNN Updated Workflow

This document summarizes everything that has been improved and how the GNN pipeline now works end-to-end.

## 1) What Was Improved

### Data pipeline
- Heavy preprocessing was moved **out of the training loop**.
- Added offline preprocessing with disk-backed `.npy` memory-mapped outputs.
- Added processed-cache reuse through compatibility checks (`meta.json` + config signature).

### Dataset
- `FastLOBDataset`: reads `X_*.npy` / `y_*.npy` with `mmap_mode="r"` and returns PyG `Data`.
- `StaticLOBTensorDataset`: new tensor-only mode returning `(x, y)` without PyG `Data`.

### Training
- Existing PyG path remains fully available as fallback.
- New optional **GCN-only static batching path**:
  - `training.static_graph_batching: true`
  - no per-sample PyG `Data` objects in the training loop
  - static `edge_index` moved to device once
  - batches handled as `[B, N, F]` tensors

### Model
- Added `GCN.forward_static(x_batch, edge_index)`.
- Supports:
  - `x_batch: [B, 3020, 2]`
  - `edge_index: [2, 18108]`
- Graph-level pooling matches mean pooling over nodes.

### Trainer
- Clean dual-mode execution:
  - classic PyG mode (`model(batch)`)
  - static mode (`model.forward_static(x, edge_index)`)
- Shared train/val/test logic with minimal duplication.

### Utilities and scripts
- `scripts/preprocess_dataset.py`: builds offline processed tensors.
- `scripts/analyze_label_distribution.py`: class balance analysis over `k/threshold/price_type`.
- `scripts/evaluate.py` updated to use processed tensors.

### Documentation and housekeeping
- GNN README updated.
- `.gitignore` improved (processed data, checkpoints, logs, notebook checkpoints).
- Tracked `.DS_Store` files removed.

---

## 2) Preprocess Output Structure

Directory: `gnn/data/processed/`

- `X_train.npy`, `y_train.npy`
- `X_val.npy`, `y_val.npy`
- `X_test.npy`, `y_test.npy`
- `edge_index.pt`
- `binner.pkl`
- `price_stats.npy` (legacy/compatibility artifact)
- `meta.json` (signature + metadata)

Notes:
- `X_*` is typically stored as `float16` on disk to reduce space.
- Conversion to `float32` happens at runtime where needed.

---

## 3) Main Configuration Keys

In `gnn/config/default.yaml`:

### Data
- `use_precomputed: true`
- `force_preprocess: false`
- `preprocess_chunk_size: 2048`
- `feature_dtype: float16`
- `prediction_horizon: 1` (`k`)
- `split_strategy: by_lag`
- `train_ratio`, `val_ratio`

### Training
- `batch_size: 256`
- `num_workers: 8`
- `static_graph_batching: false` (default)

### Model
- `type: gcn` is required to enable static path.

---

## 4) How Training Works Now

## 4.1 Standard path (fallback, PyG)
Condition:
- `training.static_graph_batching: false`
  or
- `model.type != gcn` even if static mode is requested.

Flow:
1. Load processed cache if compatible.
2. `FastLOBDataset` returns `Data(x, edge_index, y)`.
3. Custom collate uses precomputed batched edges for full batches.
4. Standard forward: `model(batch)`.

## 4.2 Static path (GCN-only)
Condition:
- `training.static_graph_batching: true`
- `model.type: gcn`

Flow:
1. Load processed cache if compatible.
2. `StaticLOBTensorDataset` returns `(x, y)`.
3. Standard PyTorch DataLoader builds:
   - `x_batch: [B, N, F]`
   - `y_batch: [B]`
4. Static `edge_index` is moved to device once.
5. Forward: `model.forward_static(x_batch, edge_index)`.

---

## 5) Operational Commands

From `~/ReMoMa/gnn`:

```bash
conda activate lob-gnn
```

### 5.1 Offline preprocess
```bash
python scripts/preprocess_dataset.py --config config/default.yaml
```

### 5.2 Training
```bash
python scripts/train.py --config config/default.yaml
```

### 5.3 Evaluation
```bash
python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config/default.yaml
```

### 5.4 Label distribution analysis
```bash
python scripts/analyze_label_distribution.py --config config/default.yaml
```

---

## 6) Cache/Preprocess Debug

If you see `Loading raw LOBSTER files...`, the cache is not considered reusable.

Quick check:

```bash
python - <<'PY'
import yaml
from src.dataset.preprocessing import has_compatible_processed_dataset
cfg = yaml.safe_load(open("config/default.yaml"))
print("compatible:", has_compatible_processed_dataset(cfg))
PY
```

If `compatible: True`, training should print:
- `Using cached processed tensors in 'data/processed'.`

---

## 7) Realistic Performance Notes

- The main bottleneck is still message passing compute/memory on large graphs (`N=3020`, `E=18108`) and many samples.
- Static mode reduces PyG Data/collation overhead, but does not remove intrinsic convolution cost.
- `batch_size=512` may still OOM depending on device VRAM.
- For fair benchmarking, keep everything fixed and only toggle:
  - `training.static_graph_batching: true/false`

---

## 8) Key Updated Files

- `src/dataset/preprocessing.py`
- `src/dataset/lob_dataset.py`
- `src/models/gcn.py`
- `src/training/trainer.py`
- `scripts/train.py`
- `scripts/preprocess_dataset.py`
- `scripts/evaluate.py`
- `scripts/analyze_label_distribution.py`
- `config/default.yaml`
- `config/test.yaml`
- `gnn/README.md`
