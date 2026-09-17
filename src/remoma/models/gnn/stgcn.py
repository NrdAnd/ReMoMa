import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool

from remoma.models.base import GNNClassifier, make_graph_conv


class STGCN(GNNClassifier):
    """Spatio-temporal GNN with interleaved temporal/graph convolutions.

    Unlike CGNN (CNN once → GNN once), the temporal (1D conv over lags) and
    spatial (graph conv over the TMFG) operations are INTERLEAVED inside stacked
    ST-blocks over the same 3020-node graph:

        block = [ temporal conv → graph conv → temporal conv ]  (+ residual, LayerNorm)

    Repeating the block lets spatial and temporal information mix at multiple
    scales — e.g. "how a multi-level pressure pattern evolves over time" — which
    a single CNN→GNN pass cannot represent. It stays a pure (spatio-temporal)
    GNN: graph message passing happens inside every block.

    Nodes are ordered (group, lag) with group = side*n_levels + level (20 groups)
    and lag = 0..n_lags (151 steps).
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 110,
        num_layers: int = 2,        # number of ST-blocks
        num_classes: int = 3,
        dropout: float = 0.3,
        num_nodes: int | None = None,
        n_lags: int = 150,
        n_levels: int = 10,
        add_lag_feature: bool = False,
        cnn_kernel: int = 3,
        conv_type: str = "gcn",
        num_heads: int = 2,
        **_kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.lag_len = n_lags + 1
        self.n_groups = 2 * n_levels
        self.num_nodes = num_nodes if num_nodes is not None else self.n_groups * self.lag_len

        self._init_pos_features(num_nodes, n_lags, add_lag_feature)
        feat_in = in_channels + self.n_extra
        self.node_encoder = nn.Linear(feat_in, hidden_channels)

        # Select the spatial operator inside each temporal/graph block.
        # GAT requires flattened PyG batches.
        self.conv_type = conv_type
        pad = cnn_kernel // 2
        self.temporal1 = nn.ModuleList()
        self.gconvs = nn.ModuleList()
        self.temporal2 = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.temporal1.append(nn.Conv1d(hidden_channels, hidden_channels, cnn_kernel, padding=pad))
            self.gconvs.append(make_graph_conv(conv_type, hidden_channels, hidden_channels, num_heads))
            self.temporal2.append(nn.Conv1d(hidden_channels, hidden_channels, cnn_kernel, padding=pad))
            self.norms.append(nn.LayerNorm(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def _tconv(self, x: Tensor, conv: nn.Conv1d) -> Tensor:
        """Apply a 1D conv over the lag axis per group. x: [B, N, H] → [B, N, H]."""
        b = x.size(0)
        g, lag = self.n_groups, self.lag_len
        h = x.view(b, g, lag, -1).permute(0, 1, 3, 2).reshape(b * g, -1, lag)  # [B*G, H, L]
        h = conv(h)
        c = h.size(1)
        return h.reshape(b, g, c, lag).permute(0, 1, 3, 2).reshape(b, g * lag, c)  # [B, N, H]

    def forward_static(self, x_batch: Tensor, edge_index: Tensor) -> Tensor:
        """Static-graph path: x_batch [B, N, Fin], single-graph edge_index."""
        x = self._cat_pos_static(x_batch)
        x = F.relu(self.node_encoder(x))                       # [B, N, H]
        for t1, gc, t2, norm in zip(self.temporal1, self.gconvs, self.temporal2, self.norms):
            res = x
            x = F.relu(self._tconv(x, t1))
            x = F.relu(gc(x, edge_index))                      # static GCNConv: [B, N, H]
            x = self._tconv(x, t2)
            x = norm(x + res)
            x = F.dropout(x, p=self.dropout, training=self.training)
        graph_emb = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.head(graph_emb)

    def forward(self, data: Data) -> Tensor:
        """PyG path: data.x [B*N, Fin], batched edge_index."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self._cat_pos_flat(x)
        x = F.relu(self.node_encoder(x))                       # [B*N, H]
        b = x.size(0) // self.num_nodes
        x = x.view(b, self.num_nodes, -1)                      # [B, N, H]
        for t1, gc, t2, norm in zip(self.temporal1, self.gconvs, self.temporal2, self.norms):
            res = x
            x = F.relu(self._tconv(x, t1))
            x_flat = x.reshape(b * self.num_nodes, -1)
            x_flat = F.relu(gc(x_flat, edge_index))            # PyG GCNConv: [B*N, H]
            x = x_flat.view(b, self.num_nodes, -1)
            x = self._tconv(x, t2)
            x = norm(x + res)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_flat = x.reshape(b * self.num_nodes, -1)
        pooled = torch.cat(
            [global_mean_pool(x_flat, batch), global_max_pool(x_flat, batch)], dim=1
        )
        return self.head(pooled)
