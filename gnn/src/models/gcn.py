import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from src.models.base import GNNClassifier


class GCN(GNNClassifier):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_nodes: int | None = None,
        n_lags: int = 150,
        add_lag_feature: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.dropout = dropout

        self._init_pos_features(num_nodes, n_lags, add_lag_feature)
        self.node_encoder = nn.Linear(in_channels + self.n_extra, hidden_channels)
        dims = [hidden_channels] * (num_layers + 1)
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.LayerNorm(hidden_channels) for _ in range(num_layers)
        )
        # head input = concat[mean pool, max pool] → 2 * hidden_channels
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self._cat_pos_flat(x)
        x = F.relu(self.node_encoder(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.head(pooled)

    def forward_static(self, x_batch: Tensor, edge_index: Tensor) -> Tensor:
        """Static-graph forward path for batched node features.

        Args:
            x_batch:    [B, N, F] node features.
            edge_index: [2, E] shared topology for all batch elements.
        Returns:
            [B, num_classes] logits.
        """
        # node_encoder applies independently across trailing feature dim.
        x = self._cat_pos_static(x_batch)
        x = F.relu(self.node_encoder(x))  # [B, N, H]

        for conv, norm in zip(self.convs, self.norms):
            # GCNConv supports static mode with x shaped [B, N, H].
            x = conv(x, edge_index)  # [B, N, H]
            # LayerNorm normalizes the last (feature) dim → no transpose, and
            # identical behavior in train/eval (no running stats).
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Concat mean + max pool over the N nodes of each graph → [B, 2H].
        graph_emb = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.head(graph_emb)
