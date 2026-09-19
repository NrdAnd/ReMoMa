# Contributing to ReMoMa

This page explains how to propose changes to the research codebase. The project
authors are credited in the [README](README.md#contributors) and in Git history.

## Proposing a change

1. Create a focused branch from the current `main` and open a pull request that
   explains the problem, the change, and how it was checked.
2. Keep model-specific implementations and configurations in their respective
   `gnn` or `recurrent` directories. Coordinate changes to shared preprocessing,
   graph construction, training, and checkpoint formats.
3. Add regression coverage when a change affects data splits, feature meaning,
   graph provenance, model behavior, or saved artifacts. Update the relevant
   documentation when commands or experimental protocols change.
4. Run the checks in the [development guide](docs/development.md) before
   requesting review. The included training tests use small synthetic data;
   report real-data performance only for experiments actually performed.

Keep raw market data, generated caches, checkpoints, and notebook outputs out
of pull requests. For work originating from an older branch, consult the
[integration history](docs/merging.md) before applying it to the current tree.

## Licensing

Original ReMoMa code and documentation are under
[AGPL-3.0-only](LICENSE). Submit only material that you have the right to
contribute under that license. Bundled visualization assets have separate
licenses listed in the [third-party notices](THIRD_PARTY_NOTICES.md); preserve
their copyright and license notices. The repository does not distribute or
license raw LOBSTER data.
