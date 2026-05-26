import numpy as np
import pandas as pd
import torch
from pathlib import Path


def load_tmfg_edge_index(
    csv_path: str | Path,
    cache_path: str | Path | None = None,
) -> torch.Tensor:
    """Load TMFG adjacency matrix and return undirected edge_index in COO format.

    The CSV has node names as both row index and column header.
    Node ordering matches the GNN feature matrix:
        ask_i_lag_k  →  i*151 + k          (i=0..9, k=0..150)
        bid_i_lag_k  →  1510 + i*151 + k   (i=0..9, k=0..150)

    Args:
        csv_path:   Path to the TMFG binary adjacency matrix CSV.
        cache_path: Optional path to cache the edge_index tensor (.pt file).

    Returns:
        torch.Tensor: edge_index of shape [2, E], dtype long.
    """
    cache_path = Path(cache_path) if cache_path else None

    if cache_path is not None and cache_path.exists():
        return torch.load(cache_path, weights_only=True)

    adj = pd.read_csv(csv_path, index_col=0).values.astype(np.float32)
    adj_t = torch.from_numpy(adj)
    edge_index = adj_t.nonzero(as_tuple=False).t().contiguous().long()

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(edge_index, cache_path)

    return edge_index
