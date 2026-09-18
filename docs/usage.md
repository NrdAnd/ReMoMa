# Usage

Run commands from the repository root in the installed environment. `--help` lists each command's actual options. YAML paths are repository-relative; explicit CLI file paths are interpreted as described by the command, so use absolute paths when invoking scripts from another directory.

## Select an experiment

Start from a complete configuration in `configs/gnn/` or `configs/recurrent/`. Copy it to `configs/local/` for private paths and experiment-specific changes. The `model.type` key chooses the architecture; `model.family` validates the family.

For a controlled family comparison, match raw files, labels, split, lag depth, base/engineered features, sampling seed, graph provenance, and evaluation rule. The default GNN and recurrent presets have different lag depths and are not a matched comparison by themselves. Use the [complete pipeline](pipeline.md) for the matched CSCO benchmark, with training-only NMI/TMFG generated automatically for every split.

```bash
python scripts/train.py --config configs/gnn/default.yaml --model sage --seed 43
python scripts/train.py --config configs/gnn/default.yaml --model cgnn \
  --hidden 350 --cnn-channels 128 --epochs 50 --split by_file
python scripts/train.py \
  --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml --hidden 8
```

`--model` selects any registered architecture, including switching families. When switching families, review the source configuration's architecture-specific settings: use a dedicated family configuration for an intentional experiment. Do not assume a CLI model change automatically makes the experimental protocols equivalent.

## Preprocess

```bash
python scripts/preprocess_dataset.py --config configs/gnn/default.yaml
python scripts/preprocess_dataset.py --config configs/gnn/default.yaml --force --chunk-size 512
```

Training also preprocesses automatically when the cache is missing or incompatible. A rebuild can be expensive: all raw files are loaded into RAM, while output tensors are memory-mapped on disk. Reduce chunk size to reduce temporary feature memory; this does not make raw-file loading streaming.

## Train and retain a run

```bash
python scripts/train.py --config configs/gnn/default.yaml --model cgnn \
  --checkpoint-dir runs/cgnn_seed42 --seed 42
```

`--checkpoint-dir` is an exact destination, which experiment runners also use. Without it, a timestamp and random suffix isolate the run under `paths.checkpoints`. Existing `best.pt` or `resolved_config.yaml` files cause a refusal to reuse that destination. Use a new directory for a new run.

A completed run contains:

```text
runs/cgnn_seed42/
  best.pt                 Best validation-macro-F1 state dictionary
  resolved_config.yaml    Effective configuration, including CLI overrides
  metadata.json           Git revision, environment, model contract, graph/binner hashes
  binner.pkl              Snapshot of the fitted training-volume bins
  preprocessing_meta.json Snapshot of data manifests and preprocessing semantics
  adjacency.csv           Snapshot of the graph (or adjacency.tsv / adjacency.npz)
  metrics.csv             Per-epoch training/validation metrics
  thresholds.json         Present when training.tune_threshold is enabled
  results/
    summary.csv           Argmax and, when enabled, tuned metrics for this run
    <model>_<split>_<timestamp>.txt
```

The optimizer is AdamW. Early stopping selects validation macro F1; the learning-rate scheduler tracks validation loss. The script restores the best checkpoint before final test evaluation. There is no training-resume command or optimizer-state checkpoint in this implementation.

## Evaluate and tune thresholds

```bash
python scripts/evaluate.py --checkpoint runs/cgnn_seed42/best.pt
python scripts/evaluate.py --checkpoint runs/cgnn_seed42/best.pt \
  --thresholds runs/cgnn_seed42/thresholds.json
python scripts/tune_threshold.py --checkpoint runs/cgnn_seed42/best.pt \
  --objective macro_f1 --save-thresholds runs/cgnn_seed42/thresholds_manual.json
```

Evaluation uses the saved configuration when `--config` is omitted. An old checkpoint without a saved configuration requires `--config`. New checkpoints validate model/feature settings and graph content before loading.

Argmax chooses the largest class probability. Threshold evaluation predicts down/up only when that probability exceeds its threshold; otherwise it predicts flat. If both directional probabilities qualify, the larger one wins, with up selected on an exact tie. Threshold fitting uses validation labels and is performed without changing model weights. Once a test set informs later model selection, it is no longer an untouched final test set.

Reported metrics include accuracy, macro/weighted/per-class F1, MCC in training reports, confusion matrices, directional precision/recall, flat-to-directional rate, and opposite-direction rate. Macro F1 always includes all three classes, even when a small split lacks a class.

## Multiple seeds and ensembles

Use one explicit directory per seed. Keep `data.subsample_seed` fixed.

```bash
python scripts/train.py --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --seed 42 --checkpoint-dir runs/recurrent_seed42
python scripts/train.py --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --seed 123 --checkpoint-dir runs/recurrent_seed123
python scripts/ensemble_evaluate.py \
  --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --checkpoints runs/recurrent_seed42/best.pt runs/recurrent_seed123/best.pt \
  --save-thresholds runs/recurrent_ensemble_thresholds.json
```

Members must share architecture, input semantics, graph, and fold. The ensemble averages probabilities and tunes ensemble thresholds on validation. It does not combine different architectures with a single configuration.

## Walk-forward and hyperparameter experiments

Generate fold configurations first; execution requires `--run`.

```bash
python scripts/run_walk_forward.py \
  --base-config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --name recurrent_baseline --min-train-files 2 --max-folds 2

python scripts/run_walk_forward.py \
  --base-config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --name recurrent_executed --min-train-files 2 --max-folds 2 --run --evaluate

python scripts/run_hparam_grid.py \
  --base-config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --name recurrent_grid --hidden-dims 5 8 --dropouts 0.15 \
  --message-iterations 1 --max-combos 2
```

Generated configurations and summaries live under `runs/generated_configs/`. These historical runners isolate processed caches and checkpoint directories per fold/combination, but retain their supplied graph across folds. Use `scripts/run_pipeline.py --mode walk_forward` for automatic training-only graph construction. The hyperparameter grid varies recurrent parameters; it is not a generic GNN grid.

`run_stable_thresholds.py` and `run_walk_forward_ensemble.py` retain historical experiments that aggregate thresholds across selected folds. Applying a threshold estimated from later folds to earlier tests is a **retrospective diagnostic**, not a prospective backtest. Use a separate earlier calibration period and a later untouched test period for a deployable threshold policy. These scripts do not automatically enforce that cross-fold temporal restriction.

## Graph construction and analysis

```bash
python scripts/build_recurrent_tmfg_adjacency.py \
  --input data/similarity/training_nmi.csv \
  --output runs/graphs/recurrent_training.tsv --max-lag 100
python scripts/build_recurrent_adjacency_from_tmfg.py --help
python scripts/analysis/average_matrices.py training_day1.csv training_day2.csv \
  --output runs/graphs/training_nmi_mean.csv
python scripts/analysis/plot_heatmap.py runs/graphs/training_nmi_mean.csv \
  --output runs/figures/training_nmi_mean.png
```

The mean-matrix utility rejects mismatched labels and orders instead of filling missing entries with zero. Include training-period matrices only when constructing a graph for held-out evaluation.

## Diagnostics

`analyze_label_distribution.py` reports class balance over horizon/threshold settings. `inspect_volume.py` examines input volumes and bins. `diagnose.py` checks features, applicable static/PyG equivalence, and single-batch overfitting. These commands require real compatible data; the test suite provides a small synthetic alternative.
