# Compatibility directory

The maintained pipeline is now the `remoma` package at the repository root. Read the [main README](../README.md) and [migration record](../docs/merging.md).

Files in `gnn/scripts/` are thin wrappers around `scripts/`. They keep the former script names and translate old configuration arguments, such as `config/default.yaml`, to the new family configuration directories. They do not duplicate model or training code.

Use root commands for new work:

```bash
python scripts/train.py --config configs/gnn/default.yaml --model cgnn
python scripts/train.py --config configs/recurrent/recurrent_sparse_sthnn_by_file.yaml
```

These examples are run from the repository root. Existing ignored `gnn/data/`, `gnn/checkpoints/`, and local figures remain local and are not published by this migration. Old caches must be rebuilt under the new preprocessing schema.
