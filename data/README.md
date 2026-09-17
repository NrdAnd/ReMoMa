# Data directories

| Directory | Version control | Purpose |
| --- | --- | --- |
| `raw/` | Ignored | Private original LOBSTER files |
| `processed/` | Ignored | Rebuildable memory-mapped features and metadata |
| `graphs/` | Tracked reference inputs | Historical labeled adjacency matrices |
| `similarity/` | README only | Locally generated similarity matrices |

See the [data contract](../docs/data.md). No raw market dataset is distributed. Keep generated experimental graphs under `runs/graphs/` until their provenance and publication suitability have been reviewed.
