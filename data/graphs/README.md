# Reference adjacency matrices

These historical inputs let both model families construct their expected graph structures. They are research references, not proof of training-only graph estimation. The [manifest](manifest.json) records file hashes, dimensions, edge counts, and known provenance.

| File | Nodes | Directed nonzero entries | Meaning |
| --- | ---: | ---: | --- |
| `tmfg_lag150_bins2000.csv` | 3020 | 18108 | Legacy TMFG used by the GNN examples |
| `recurrent_sparse_tmfg_lag100_bins2000_from_full_tmfg_adjacency.tsv` | 2020 | 12024 | Lag-100 subset of the full graph; an induced subset need not remain a maximal TMFG |
| `recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv` | 2020 | 12108 | TMFG rebuilt for lag 100 from the mean similarity input |
| `recurrent_sparse_tmfg_lag150_bins2000_from_nmi_mean.tsv` | 3020 | 18108 | Recurrent-labeled lag-150 graph |
| `recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean_weighted.tsv` | 2020 | 12108 | Weighted reconstruction supplied during integration |

The weighted configuration referenced a file absent from the source branch. It was reconstructed with `build_recurrent_tmfg_adjacency.py --max-lag 100 --weighted` from the existing local mean NMI matrix. Its nonzero topology exactly matches the versioned unweighted lag-100 reference. The source matrix hash is in the manifest; its source dates were not recorded, so the reconstruction does not establish training-only provenance.

To build a new evaluation graph, supply a similarity matrix estimated from the training period and record its dates and hash. Do not replace a graph used by an existing checkpoint: retain that run's graph snapshot.
