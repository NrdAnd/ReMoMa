import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, global_mean_pool

from src.models.base import GNNClassifier


class GAT(GNNClassifier):
    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_heads: int = 4,
        **_kwargs,
    ):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout, concat=False))
            self.norms.append(nn.BatchNorm1d(hidden_channels))
        else:
            # First layer: in → hidden * heads
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
            self.norms.append(nn.BatchNorm1d(hidden_channels * num_heads))

            # Intermediate layers: hidden*heads → hidden*heads
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=num_heads, dropout=dropout, concat=True))
                self.norms.append(nn.BatchNorm1d(hidden_channels * num_heads))

            # Last layer: hidden*heads → hidden (single head, average)
            self.convs.append(GATConv(hidden_channels * num_heads, hidden_channels, heads=1, dropout=dropout, concat=False))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.head(global_mean_pool(x, batch))
