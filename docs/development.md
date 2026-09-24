# Development

## Environment and checks

```bash
python -m pip install -e '.[dev]'
python -m compileall -q src scripts
ruff check src scripts tests
python -m unittest discover -s tests -v
python scripts/check_repository.py
python -m build
```

CI runs these checks on Linux CPU with Python 3.10 and 3.11. Tests use synthetic data in temporary directories, including one-epoch training for GCN, GAT, and recurrent sparse STHNN; no LOBSTER download is needed. Ruff checks undefined names and syntax errors.

`tests/test_complete_pipeline.py` covers the maintained orchestration, training-only graph construction, both NMI estimators, indexed feature equivalence, threshold-search equivalence, cache integrity, and resume behavior. Its CUDA parity test is opt-in; see [pipeline verification](pipeline.md#verification-boundaries). The complete experiment entry point is `scripts/run_pipeline.py`; omitting `--run` only inspects the plan. Record actual verification results and platform limits in [validation](validation.md).

## Adding or changing a model

1. Add the implementation under `src/remoma/models/gnn/` or `src/remoma/models/recurrent/`.
2. Register it in that family's `__init__.py`. Keep model names unique across families.
3. Implement graph-level logits `[B, 3]` through `forward`; implement `forward_static` only if supported.
4. Add a complete configuration or a selected-model preset, documenting input assumptions.
5. Extend tests for forward/backward behavior, batching equivalence where applicable, and checkpoint reload.

The shared factory provides dimensions and selected settings to constructors. Preserve compatible state-dictionary names within an existing model, or document the checkpoint migration explicitly. New preprocessing semantics require a cache-schema change and regression tests.

## Collaboration boundaries

GNN contributors should ordinarily modify `models/gnn/`, `configs/gnn/`, and related tests. Recurrent contributors should ordinarily modify `models/recurrent/`, `configs/recurrent/`, and related tests. Coordinate changes to the common factory, configuration loader, dataset, trainer, and checkpoint utilities.

Start feature branches from `main`. Use the [migration map and branch workflow](merging.md) when bringing older work forward.

## Repository hygiene

- Use English for identifiers, comments, docstrings, CLI messages, and documentation.
- Clear notebook outputs and execution counts before committing.
- Keep raw data, caches, checkpoints, generated experiment configurations, and machine-specific paths out of Git.
- Keep curated historical results under `docs/reports/`; annotate dataset and protocol limitations.
- Update documentation and tests when changing public commands or file contracts.
- Report predictive performance from declared data experiments separately from synthetic correctness checks.

Legacy prototypes in `archive/legacy/` are retained for historical inspection. Their normalization and feature-generation assumptions are not supported interfaces and must not be used for current reported experiments.
