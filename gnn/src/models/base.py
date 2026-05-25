from abc import ABC, abstractmethod

import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data


class GNNClassifier(nn.Module, ABC):
    """Abstract base for graph-level GNN classifiers."""

    @abstractmethod
    def forward(self, data: Data) -> Tensor:
        """Returns logits of shape [batch_size, num_classes]."""

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
