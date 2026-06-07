import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_max_pool, global_mean_pool

from src.models.base import GNNClassifier
from src.models.bin import BiN


class CGNN(GNNClassifier):
    """Spatio-temporal model: temporal 1D-CNN over lags + spatial GNN over TMFG.

    The 3020 nodes are ordered (group, lag) with group = side*n_levels + level
    (20 groups) and lag = 0..n_lags (151 steps). Pipeline:

      1. x [B, N, Fin] → reshape [B, groups, lag_len, Fin]
      2. 1D-CNN convolves over the LAG axis (per group, weights shared across
         groups) → temporal features that capture how values change over time
         (order-flow dynamics the static GNN washes out). Length is preserved
         so all N nodes survive.
      3. reshape back to [B, N, H] node features
      4. GCN message passing over the TMFG edges + LayerNorm  (spatial reasoning)
      5. mean+max pool + MLP head

    The CNN supplies the temporal dimension the plain GNN lacks; the GNN keeps
    the cross-level/cross-side spatial structure of the order book.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 128,
        num_layers: int = 3,
        num_classes: int = 3,
        dropout: float = 0.3,
        num_nodes: int | None = None,
        n_lags: int = 150,
        n_levels: int = 10,
        add_lag_feature: bool = False,
        cnn_channels: int = 64,
        cnn_kernel: int = 5,
        use_bin: bool = False,
        **_kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.n_lags = n_lags
        self.n_levels = n_levels
        self.lag_len = n_lags + 1          # 151 steps per (side, level)
        self.n_groups = 2 * n_levels       # 20 = 2 sides × 10 levels
        self.num_nodes = num_nodes if num_nodes is not None else self.n_groups * self.lag_len

        # Learned bilinear input normalization (over raw price/volume channels)
        self.bin = BiN(self.n_groups, self.lag_len, in_channels) if use_bin else None

        self._init_pos_features(num_nodes, n_lags, add_lag_feature)
        feat_in = in_channels + self.n_extra

        # ── Temporal CNN over the lag axis (shared across all 20 groups) ──
        pad = cnn_kernel // 2
        self.tcnn = nn.Sequential(
            nn.Conv1d(feat_in, cnn_channels, kernel_size=cnn_kernel, padding=pad),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, hidden_channels, kernel_size=cnn_kernel, padding=pad),
            nn.ReLU(),
        )

        # ── Spatial GNN over the TMFG graph ──
        dims = [hidden_channels] * (num_layers + 1)
        self.convs = nn.ModuleList(GCNConv(dims[i], dims[i + 1]) for i in range(num_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers))
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, num_classes),
        )

    def _temporal(self, x_bnf: Tensor) -> Tensor:
        """[B, N, Fin] → temporal CNN over lag → [B, N, H]."""
        b = x_bnf.size(0)
        g, lag = self.n_groups, self.lag_len
        x = x_bnf.view(b, g, lag, -1)                 # [B, G, L, Fin]
        fin = x.size(-1)
        x = x.permute(0, 1, 3, 2).reshape(b * g, fin, lag)   # [B*G, Fin, L]
        x = self.tcnn(x)                               # [B*G, H, L]
        h = x.size(1)
        x = x.reshape(b, g, h, lag).permute(0, 1, 3, 2)      # [B, G, L, H]
        return x.reshape(b, g * lag, h)                # [B, N, H]

    def _gnn(self, x: Tensor, edge_index: Tensor) -> Tensor:
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def forward_static(self, x_batch: Tensor, edge_index: Tensor) -> Tensor:
        """Static-graph path: x_batch [B, N, Fin]."""
        if self.bin is not None:
            x_batch = self.bin(x_batch)
        x = self._cat_pos_static(x_batch)
        x = self._temporal(x)                          # [B, N, H]
        x = self._gnn(x, edge_index)                   # [B, N, H]
        graph_emb = torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=1)
        return self.head(graph_emb)

    def forward(self, data: Data) -> Tensor:
        """PyG path: data.x [B*N, Fin]."""
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.bin is not None:
            b = x.size(0) // self.num_nodes
            x = self.bin(x.view(b, self.num_nodes, -1)).reshape(x.size(0), -1)
        x = self._cat_pos_flat(x)
        b = x.size(0) // self.num_nodes
        x = self._temporal(x.view(b, self.num_nodes, -1))    # [B, N, H]
        x = x.reshape(b * self.num_nodes, -1)                # back to flat [B*N, H]
        x = self._gnn(x, edge_index)
        pooled = torch.cat(
            [global_mean_pool(x, batch), global_max_pool(x, batch)], dim=1
        )
        return self.head(pooled)
