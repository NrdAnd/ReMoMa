import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool

from src.models.base import GNNClassifier


class GCN(GNNClassifier):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        **_kwargs,
    ):
        super().__init__()
        self.dropout = dropout

        self.node_encoder = nn.Linear(in_channels, hidden_channels)
        dims = [hidden_channels] * (num_layers + 1)
        self.convs = nn.ModuleList(
            GCNConv(dims[i], dims[i + 1]) for i in range(num_layers)
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(hidden_channels) for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.node_encoder(x))
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(global_mean_pool(x, batch))

    def forward_static(self, x_batch: Tensor, edge_index: Tensor) -> Tensor:
        """Static-graph forward path for batched node features.

        Args:
            x_batch:    [B, N, F] node features.
            edge_index: [2, E] shared topology for all batch elements.
        Returns:
            [B, num_classes] logits.
        """
        # node_encoder applies independently across trailing feature dim.
        x = F.relu(self.node_encoder(x_batch))  # [B, N, H]

        for conv, norm in zip(self.convs, self.norms):
            # GCNConv supports static mode with x shaped [B, N, H].
            x = conv(x, edge_index)  # [B, N, H]
            # BatchNorm1d expects channel dim in position 1 for 3D input.
            x = norm(x.transpose(1, 2)).transpose(1, 2)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        # Equivalent to graph-level global mean pool when every graph has N nodes.
        graph_emb = x.mean(dim=1)
        return self.head(graph_emb)
