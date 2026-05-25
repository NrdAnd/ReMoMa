from src.models.base import GNNClassifier
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE

_REGISTRY = {"gcn": GCN, "gat": GAT, "sage": GraphSAGE}


def build_model(cfg: dict) -> GNNClassifier:
    model_type = cfg["model"]["type"].lower()
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY)}")

    params = {
        "in_channels":     2,
        "hidden_channels": cfg["model"]["hidden_channels"],
        "num_layers":      cfg["model"]["num_layers"],
        "num_classes":     3,
        "dropout":         cfg["model"]["dropout"],
        "num_heads":       cfg["model"].get("num_heads", 4),
    }
    return _REGISTRY[model_type](**params)
