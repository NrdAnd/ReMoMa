import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class BiN(nn.Module):
    """Bilinear Normalization (Tran et al.) for LOB tensors.

    Learns an adaptive input normalization along BOTH axes — spatial (the
    side/level groups) and temporal (the lags) — and combines them with
    learnable weights. Each feature channel (price, volume) is normalized
    separately, since they live on very different scales.

    Input/Output: [B, N, C] with N = n_groups * lag_len (node ordering
    (group, lag)) and C feature channels.

    For each channel:
      mode-1 (temporal): normalize each group over its lags  → affine γ1,β1 (per group)
      mode-2 (spatial):  normalize each lag over the groups   → affine γ2,β2 (per lag)
      out = λ1·mode1 + λ2·mode2   (λ ≥ 0, learnable)
    """

    def __init__(self, n_groups: int, lag_len: int, n_channels: int):
        super().__init__()
        self.g, self.l, self.c = n_groups, lag_len, n_channels
        self.gamma1 = nn.Parameter(torch.ones(n_channels, n_groups))
        self.beta1 = nn.Parameter(torch.zeros(n_channels, n_groups))
        self.gamma2 = nn.Parameter(torch.ones(n_channels, lag_len))
        self.beta2 = nn.Parameter(torch.zeros(n_channels, lag_len))
        self.lam1 = nn.Parameter(torch.tensor(0.5))
        self.lam2 = nn.Parameter(torch.tensor(0.5))

    def forward(self, x: Tensor) -> Tensor:
        b = x.size(0)
        eps = 1e-5
        # [B, N, C] → [B, C, G, L]
        x = x.view(b, self.g, self.l, self.c).permute(0, 3, 1, 2)

        # mode-1: normalize over lags (dim 3), per group
        mu_t = x.mean(dim=3, keepdim=True)
        std_t = x.std(dim=3, keepdim=True) + eps
        y1 = (x - mu_t) / std_t
        y1 = self.gamma1.view(1, self.c, self.g, 1) * y1 + self.beta1.view(1, self.c, self.g, 1)

        # mode-2: normalize over groups (dim 2), per lag
        mu_d = x.mean(dim=2, keepdim=True)
        std_d = x.std(dim=2, keepdim=True) + eps
        y2 = (x - mu_d) / std_d
        y2 = self.gamma2.view(1, self.c, 1, self.l) * y2 + self.beta2.view(1, self.c, 1, self.l)

        out = F.relu(self.lam1) * y1 + F.relu(self.lam2) * y2     # [B, C, G, L]
        return out.permute(0, 2, 3, 1).reshape(b, self.g * self.l, self.c)
