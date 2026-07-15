from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def make_graph_conv(conv_type: str, in_dim: int, out_dim: int, num_heads: int = 2):
    """Factory for the spatial graph operator used inside the (spatio-temporal) models.

    Lets CGNN/STGCN swap operator via a single `conv_type` flag. GAT uses
    concat=False so the output stays `out_dim` (heads are averaged), keeping the
    residual/LayerNorm dimensions unchanged.
    """
    if conv_type == "gcn":
        return GCNConv(in_dim, out_dim)
    if conv_type == "sage":
        return SAGEConv(in_dim, out_dim)
    if conv_type == "gat":
        return GATConv(in_dim, out_dim, heads=num_heads, concat=False)
    raise ValueError(f"conv_type must be 'gcn'|'sage'|'gat', got '{conv_type}'")


class GNNClassifier(nn.Module, ABC):
    """Abstract base for graph-level GNN classifiers."""

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Returns logits of shape [batch_size, num_classes]."""

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ── Positional (lag) node feature ─────────────────────────────
    # The lag of a node is a deterministic function of its index, identical
    # for every sample, so it is built once as a buffer and concatenated to
    # the node features at forward time (no disk/storage cost).
    def _init_pos_features(self, num_nodes: int | None, n_lags: int, enabled: bool) -> None:
        self.num_nodes = num_nodes
        if not enabled or num_nodes is None:
            self.register_buffer("lag_feat", None)
            self.n_extra = 0
            return
        per_side = num_nodes // 2        # ask nodes [0:per_side), bid [per_side:)
        per_level = n_lags + 1           # lags 0..n_lags per (side, level)
        idx = torch.arange(num_nodes)
        within = torch.where(idx < per_side, idx, idx - per_side)
        lag = (within % per_level).float() / max(n_lags, 1)   # ∈ [0, 1]
        # persistent=False: deterministic, rebuilt in __init__, keep out of state_dict
        self.register_buffer("lag_feat", lag.unsqueeze(1), persistent=False)  # [num_nodes, 1]
        self.n_extra = 1

    def _cat_pos_flat(self, x: Tensor) -> Tensor:
        """Concat lag to PyG-batched node features x: [B*num_nodes, F]."""
        if self.lag_feat is None:
            return x
        within = torch.arange(x.size(0), device=x.device) % self.num_nodes
        return torch.cat([x, self.lag_feat[within]], dim=1)

    def _cat_pos_static(self, x: Tensor) -> Tensor:
        """Concat lag to static-batched node features x: [B, num_nodes, F]."""
        if self.lag_feat is None:
            return x
        pos = self.lag_feat.unsqueeze(0).expand(x.size(0), -1, -1)
        return torch.cat([x, pos], dim=2)
