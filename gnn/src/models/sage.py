import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool

from src.models.base import GNNClassifier


class GraphSAGE(GNNClassifier):
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
            SAGEConv(dims[i], dims[i + 1]) for i in range(num_layers)
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
