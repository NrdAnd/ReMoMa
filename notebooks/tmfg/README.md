# TMFG graph construction

`build_graph_tmfg.ipynb` filters a similarity matrix, constructs a TMFG, and explores degree centrality, separator participation, clique diversity, and path lengths.

The reusable implementation is [remoma.graph.tmfg](../../src/remoma/graph/tmfg.py). It supports binary adjacency, weighted adjacency, and LoGo output modes. Root-level graph builder scripts export labeled matrices from supplied similarities; the [complete pipeline](../../docs/pipeline.md) builds fold-specific TMFGs automatically from training-only NMI.

Run the bootstrap cell first, then select the intended training-only similarity matrix. The historical notebook's lag-zero filtering rule is an experimental choice, not a requirement of every model. Edge count alone does not prove planarity, and clustering/path length alone do not establish a small-world result.

`lib/` contains the existing third-party visualization assets. Their upstream notices and license texts are retained; see the [third-party notices](../../THIRD_PARTY_NOTICES.md). See the [notebook guidance](../README.md).
