# ReMoMa

ReMoMa is a research framework for classifying future limit order book price movements as **down (0), flat (1), or up (2)**. It provides graph neural networks and a recurrent sparse spatio-temporal model through a shared preprocessing, training, and evaluation pipeline.

The two model families coexist in one branch and one Python package. Selecting a model does not require switching Git branches.

## Model selection

| Family | `model.type` values | Configuration |
| --- | --- | --- |
| GNN | `gcn`, `gat`, `sage`, `cgnn`, `cgnn_sage`, `cgnn_gat`, `stgcn`, `stgcn_sage`, `stgcn_gat` | [GNN default](configs/gnn/default.yaml) |
| Recurrent | `recurrent_sparse_sthnn` | [Recurrent by-file](configs/recurrent/recurrent_sparse_sthnn_by_file.yaml) |

The recurrent model uses a GRU cell and shared message transformations on feature/lag graph nodes. It is also graph-based; “GNN” and “recurrent” distinguish the two implementation families in this repository.

## Quick start

Use Python 3.10 or 3.11. Run commands from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m unittest discover -s tests -v
```

The test suite includes small synthetic one-epoch training checks; it does not load real market data. Use a remote compute host for the full training commands below.

Place five correctly ordered LOBSTER ten-level orderbook files in `data/raw/`. Raw market data is not distributed with the repository. See [setup](docs/setup.md) and the [data contract](docs/data.md) before preprocessing.

```bash
# GNN example: temporal CNN followed by graph convolutions.
python scripts/preprocess_dataset.py --config configs/gnn/default.yaml
python scripts/train.py --config configs/gnn/default.yaml --model cgnn \
  --checkpoint-dir runs/cgnn_example

# Recurrent example: separate graph, feature cache, and run directory.
python scripts/preprocess_dataset.py \
  --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml
python scripts/train.py \
  --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml \
  --checkpoint-dir runs/recurrent_example

# Reload the saved effective configuration automatically.
python scripts/evaluate.py --checkpoint runs/cgnn_example/best.pt
python scripts/evaluate.py --checkpoint runs/recurrent_example/best.pt \
  --thresholds runs/recurrent_example/thresholds.json
```

Without `--checkpoint-dir`, training creates a unique directory under the configured checkpoint root. An explicit directory is used exactly as supplied and must not already contain a run.

## Repository structure

```text
src/remoma/           Shared Python package
  dataset/           Labels, binning, engineered features, preprocessing
  graph/             Graph loading, validation, and TMFG construction
  models/gnn/        Nine graph-convolution architectures
  models/recurrent/  Recurrent sparse STHNN implementation
  training/          Trainer, metrics, decision thresholds
  utils/             Input alignment and checkpoint provenance
scripts/             Training, evaluation, experiments, and analysis commands
configs/             Separate GNN and recurrent experiment configurations
data/graphs/         Versioned reference adjacency matrices
docs/                Setup, architecture, usage, development, and deployment
notebooks/           Exploratory LOB, GLASSO, and TMFG notebooks
tests/               Synthetic integration and regression tests
deploy/              CPU batch-execution container
archive/legacy/      Historical prototypes outside the supported pipeline
gnn/                 Compatibility entry points for previous commands
```

## Documentation

- [Documentation index](docs/README.md)
- [Installation and environment](docs/setup.md)
- [Architecture and model contracts](docs/architecture.md)
- [Data, labels, splits, and graph provenance](docs/data.md)
- [Training, evaluation, and experiments](docs/usage.md)
- [Configuration reference](configs/README.md)
- [Development and validation](docs/development.md)
- [Branch integration and migration procedure](docs/merging.md)
- [Deployment and GitHub publication](docs/deployment.md)
- [Historical experiments and limitations](docs/reports/README.md)

## Research status

This integration includes synthetic CPU validation. It does not reproduce the historical full-data experiments or establish a live trading service. Historical results used earlier preprocessing and, in some cases, different test subsamples. The reference graphs lack a complete training-only provenance record; rebuild them from the training period before making leakage-free generalization claims. See the [validation record](docs/validation.md).

No repository-wide license has been selected. The owners must choose an appropriate license before presenting this work as open source; existing third-party license notices remain in place.
