from src.models.base import GNNClassifier
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE

_REGISTRY = {"gcn": GCN, "gat": GAT, "sage": GraphSAGE}


def build_model(cfg: dict) -> GNNClassifier:
    model_type = cfg["model"]["type"].lower()
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY)}")

    data_cfg = cfg.get("data", {})
    n_lags = int(data_cfg.get("n_lags", 150))
    n_levels = int(data_cfg.get("n_levels", 10))
    num_nodes = 2 * n_levels * (n_lags + 1)

    params = {
        "in_channels":     2,
        "hidden_channels": cfg["model"]["hidden_channels"],
        "num_layers":      cfg["model"]["num_layers"],
        "num_classes":     3,
        "dropout":         cfg["model"]["dropout"],
        "num_heads":       cfg["model"].get("num_heads", 4),
        "num_nodes":       num_nodes,
        "n_lags":          n_lags,
        "add_lag_feature": cfg["model"].get("add_lag_feature", False),
    }
    return _REGISTRY[model_type](**params)
