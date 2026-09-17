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
with its own 100-lag config and processed cache. The canonical graph for this
experiment is rebuilt directly from the Cisco NMI mean matrix:

```text
csv_NMI_matrix/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv
```

Build the recurrent adjacency from that source with:

```bash
python scripts/build_recurrent_tmfg_adjacency.py
```

This writes:

```text
gnn/data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv
```

The earlier adjacency converted from `tmfg/cisco_tmfg_adj_matrix_2000_bins.csv`
was a bootstrap fallback used before the NMI mean source was available. Keep it
only for reproducing that first exploratory run; do not use it for new results.

Train the canonical recurrent sparse model:

```bash
python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn.yaml
python scripts/train.py --config config/recurrent_sparse_sthnn.yaml
```

Evaluate the trained checkpoint:

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml
```

After training, calibrate directional decision thresholds on validation and
evaluate the same checkpoint with the saved thresholds:

```bash
python scripts/tune_threshold.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml

python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean/best.pt \
  --config config/recurrent_sparse_sthnn.yaml \
  --thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean/thresholds.json
```

Optional probability ensemble / bagging run:

```bash
python scripts/train.py \
  --config config/recurrent_sparse_sthnn.yaml \
  --seed 42 \
  --checkpoint-dir checkpoints/recurrent_sparse_sthnn_nmi_mean_seed42

python scripts/train.py \
  --config config/recurrent_sparse_sthnn.yaml \
  --seed 123 \
  --checkpoint-dir checkpoints/recurrent_sparse_sthnn_nmi_mean_seed123

python scripts/train.py \
  --config config/recurrent_sparse_sthnn.yaml \
  --seed 777 \
  --checkpoint-dir checkpoints/recurrent_sparse_sthnn_nmi_mean_seed777
```

Evaluate the ensemble by averaging checkpoint probabilities. The default
threshold objective is plain `macro_f1`, with no flat false-positive penalty:

```bash
python scripts/ensemble_evaluate.py \
  --config config/recurrent_sparse_sthnn.yaml \
  --checkpoints \
    checkpoints/recurrent_sparse_sthnn_nmi_mean_seed42/best.pt \
    checkpoints/recurrent_sparse_sthnn_nmi_mean_seed123/best.pt \
    checkpoints/recurrent_sparse_sthnn_nmi_mean_seed777/best.pt \
  --save-thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean_ensemble/thresholds.json
```

For a stricter day-held-out experiment, use the by-file config. This trains on
files `[0, 1, 2]`, validates on file `[3]`, and tests on file `[4]`:

```bash
python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn_by_file.yaml
python scripts/train.py --config config/recurrent_sparse_sthnn_by_file.yaml
```

The first feature-engineering variant keeps the same lag100 NMI/TMFG graph and
target definition, but adds causal per-node channels derived only from the
historical LOB window: spread, level imbalance, total depth imbalance,
microprice, and side indicator. It writes to a separate processed directory and
checkpoint directory, so it does not overwrite the canonical run:

```bash
python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn_feature_rich.yaml
python scripts/train.py --config config/recurrent_sparse_sthnn_feature_rich.yaml
python scripts/tune_threshold.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich/best.pt \
  --config config/recurrent_sparse_sthnn_feature_rich.yaml
python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich/best.pt \
  --config config/recurrent_sparse_sthnn_feature_rich.yaml \
  --thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich/thresholds.json
```

To test the next roadmap block in one controlled experiment, build a weighted
TMFG adjacency, add order-flow channels from LOBSTER message files, and run the
combined config:

```bash
python scripts/build_recurrent_tmfg_adjacency.py \
  --weighted \
  --output data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean_weighted.tsv

python scripts/preprocess_dataset.py --config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml
python scripts/train.py --config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml
python scripts/tune_threshold.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted/best.pt \
  --config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25 \
  --save-thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted/thresholds_penalized.json
python scripts/evaluate.py \
  --checkpoint checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted/best.pt \
  --config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml \
  --thresholds checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted/thresholds_penalized.json
```

For expanding walk-forward validation, generate and optionally run by-file
folds from the same base config. With five Cisco days and the default
`--min-train-files 2`, this creates two folds:
`train=[0,1], val=[2], test=[3]` and
`train=[0,1,2], val=[3], test=[4]`.

```bash
python scripts/run_walk_forward.py \
  --base-config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml \
  --name feature_rich_orderflow_weighted \
  --run \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25
```

To isolate whether order-flow channels and weighted edges are actually helping,
run the same walk-forward protocol with the feature-rich unweighted baseline.
This uses separate processed/checkpoint roots and will not overwrite the
order-flow weighted run:

```bash
python scripts/run_walk_forward.py \
  --base-config config/recurrent_sparse_sthnn_feature_rich_unweighted_walkforward.yaml \
  --name feature_rich_unweighted \
  --run \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25
```

Step 4 runs a small walk-forward hyperparameter grid on the current strongest
setup. By default it evaluates:

```text
hidden_dim: 8, 16
readout_dropout: 0.15, 0.20, 0.25
message_iterations: 1, 2
```

Each combination gets separate processed/checkpoint folders and a separate
walk-forward name, so existing best checkpoints are not overwritten:

```bash
python scripts/run_hparam_grid.py \
  --base-config config/recurrent_sparse_sthnn_feature_rich_orderflow_weighted.yaml \
  --name feature_rich_orderflow_weighted_grid \
  --run \
  --skip-existing \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25
```

The aggregate summaries are written to:

```text
config/hparam_grid/feature_rich_orderflow_weighted_grid/fold_results.csv
config/hparam_grid/feature_rich_orderflow_weighted_grid/combo_summary.csv
```

Step 5 stabilizes the decision thresholds across validation folds. For the
current best grid combination (`hd16_do20_mi1`), this averages the fold-specific
validation thresholds, applies the same stable threshold pair to every fold test
split, and writes new reports without touching the trained checkpoints:

```bash
python scripts/run_stable_thresholds.py \
  --fold-results config/hparam_grid/feature_rich_orderflow_weighted_grid/fold_results.json \
  --combo hd16_do20_mi1 \
  --method mean \
  --output-dir config/stable_thresholds/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_mean
```

The stable-threshold outputs are written to:

```text
config/stable_thresholds/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_mean/stable_thresholds.json
config/stable_thresholds/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_mean/fold_results.csv
config/stable_thresholds/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_mean/summary.json
```

Step 6 builds the final stable ensemble only from the best stable model family.
It does not ensemble different temporal folds against each other. Instead, for
each walk-forward fold it trains extra random seeds of the same best model,
averages their probabilities with the original fold checkpoint, tunes ensemble
thresholds on each validation fold, then evaluates one stable threshold pair
across folds:

```bash
python scripts/run_walk_forward_ensemble.py \
  --fold-results config/hparam_grid/feature_rich_orderflow_weighted_grid/fold_results.json \
  --combo hd16_do20_mi1 \
  --seeds 123 777 \
  --include-base \
  --run-train \
  --skip-existing \
  --objective macro_f1_penalized \
  --flat-fp-penalty 0.25 \
  --stable-method mean \
  --output-dir config/ensembles/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_seed_ensemble
```

This creates new seed checkpoint directories such as:

```text
checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted_hd16_do20_mi1_fold_01_seed123/
checkpoints/recurrent_sparse_sthnn_nmi_mean_feature_rich_orderflow_weighted_hd16_do20_mi1_fold_01_seed777/
```

and writes ensemble summaries to:

```text
config/ensembles/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_seed_ensemble/fold_results.csv
config/ensembles/feature_rich_orderflow_weighted_grid_hd16_do20_mi1_seed_ensemble/summary.json
```

Its adjacency must be a labeled CSV/TSV matrix whose index and columns match
labels such as `ASKs1_lag100`. The model internally reorders the existing
processed LOB tensor order to match that adjacency order.

For cloud runs, keep the same relative layout:

```text
ReMoMa/
  lobster_cisco/                         # raw *_orderbook_10.csv files
  csv_NMI_matrix/
    lob_similarity_nmi_rellag_lag150_bins2000_mean.csv
  gnn/
    data/adjacency/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv
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
