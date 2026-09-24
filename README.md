# ReMoMa

ReMoMa is a research framework for classifying future limit order book price movements as **down (0), flat (1), or up (2)**. Graph neural networks and a recurrent sparse spatio-temporal model share preprocessing, training, and evaluation in one Python package.

The [complete CSCO pipeline](docs/pipeline.md) builds normalized mutual information (NMI) matrices and Triangulated Maximally Filtered Graphs (TMFGs) from each split's training dates, then trains and evaluates the selected models. Relative-lag NMI is the default; full pairwise NMI is also available.

## Contributors

Developed collaboratively by [NrdAnd](https://github.com/NrdAnd) and [SimoSaimon](https://github.com/SimoSaimon). Both contributors' commits and authorship are preserved in Git history.

## Models

| Family | `model.type` values | Configuration |
| --- | --- | --- |
| GNN | `gcn`, `gat`, `sage`, `cgnn`, `cgnn_sage`, `cgnn_gat`, `stgcn`, `stgcn_sage`, `stgcn_gat` | [GNN default](configs/gnn/default.yaml) |
| Recurrent | `recurrent_sparse_sthnn` | [Recurrent by-file](configs/recurrent/recurrent_sparse_sthnn_by_file.yaml) |

The recurrent model applies shared graph messages and GRU updates to feature/lag nodes. Both families are graph-based and available on `main`; model selection requires no branch change. The complete pipeline supplies matched experiment settings; the individual presets use different lag depths.

## Quick start

Use Python 3.10 or 3.11 and run commands from the repository root. See [setup](docs/setup.md) for CPU/CUDA installation.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python scripts/check_repository.py
```

Place the five original CSCO ten-level orderbook files listed in the [pipeline guide](docs/pipeline.md#installation-and-input-files) in `data/raw/`. Raw market data is not included. Once the files are present, inspect the fixed or walk-forward plan:

```bash
python scripts/run_pipeline.py
python scripts/run_pipeline.py --mode walk_forward
```

These commands inspect filenames and settings without reading raw contents or training. Run the complete fixed-split experiment on a compute host with:

```bash
python scripts/run_pipeline.py --run
```

The default NMI backend is `auto`; use `--nmi-backend cpu` or `cuda` to select it explicitly. GPU NMI requires the optional CuPy installation described in the [pipeline guide](docs/pipeline.md). Model training selects CUDA independently through PyTorch.

For individual preprocessing, model selection, training, and checkpoint evaluation commands, see [usage](docs/usage.md). Those commands accept a supplied graph; the complete pipeline constructs training-only graphs automatically.

## Repository structure

```text
src/remoma/           Shared Python package
  dataset/           Labels, binning, indexed/materialized features, preprocessing
  graph/             NMI estimation, graph validation, and TMFG construction
  models/gnn/        Nine graph-convolution architectures
  models/recurrent/  Recurrent sparse STHNN implementation
  pipeline/          Fixed-split and walk-forward experiment orchestration
  training/          Trainer, metrics, decision thresholds
  utils/             Caches, artifact locks, checkpoint provenance
scripts/             Training, evaluation, experiments, and analysis commands
configs/             GNN, recurrent, and complete-pipeline configurations
data/graphs/         Historical reference adjacency matrices
docs/                User guides, technical reference, and research reports
notebooks/           Exploratory LOB, GLASSO, and TMFG notebooks
tests/               Synthetic integration and regression tests
deploy/              CPU batch-execution container
archive/legacy/      Historical prototypes outside the supported pipeline
gnn/                 Compatibility entry points for previous commands
```

## Documentation

Start with [setup](docs/setup.md) and the [complete pipeline](docs/pipeline.md). Use the [architecture](docs/architecture.md), [data contract](docs/data.md), and [configuration reference](configs/README.md) for technical details.

The [documentation index](docs/README.md) also covers individual commands, development, deployment, integration history, and historical reports.

## Research status

Synthetic CPU tests cover implementation correctness. Historical full-data results have not been reproduced with the current pipeline, and the reference graphs lack complete training-date provenance. Rebuild graphs from training data for held-out evaluation. See [validation](docs/validation.md) and [historical results and limitations](docs/reports/README.md).

## License

Original code and documentation: [GNU Affero General Public License v3.0 only](LICENSE) (`AGPL-3.0-only`). Copyright 2026 Andrea Nardi and Simone Somazzi.

Bundled visualization assets retain their own licenses; see [third-party notices](THIRD_PARTY_NOTICES.md). Raw LOBSTER data is not included or licensed by this repository.
