from functools import partial

from src.models.base import GNNClassifier
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE
from src.models.cgnn import CGNN
from src.models.stgcn import STGCN

_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "sage": GraphSAGE,
    "cgnn": CGNN,
    "cgnn_sage": partial(CGNN, conv_type="sage"),
    "stgcn": STGCN,
}


def build_model(cfg: dict) -> GNNClassifier:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY)}")

    data_cfg = cfg.get("data", {})
    n_lags = int(data_cfg.get("n_lags", 150))
    n_levels = int(data_cfg.get("n_levels", 10))
    num_nodes = 2 * n_levels * (n_lags + 1)

    # Per-architecture presets under model.overrides.<type> take precedence
    # over the shared model.<param> values. So choosing the architecture
    # automatically selects its tuned hyper-parameters.
    ov = (model_cfg.get("overrides") or {}).get(model_type, {}) or {}

    def pick(key, default=None):
        return ov.get(key, model_cfg.get(key, default))

    params = {
        "in_channels":     2,
        "hidden_channels": pick("hidden_channels"),
        "num_layers":      pick("num_layers"),
        "num_classes":     3,
        "dropout":         pick("dropout"),
        "num_heads":       pick("num_heads", 4),
        "num_nodes":       num_nodes,
        "n_lags":          n_lags,
        "n_levels":        n_levels,
        "add_lag_feature": model_cfg.get("add_lag_feature", False),
        "cnn_channels":    pick("cnn_channels", 64),
        "cnn_kernel":      pick("cnn_kernel", 5),
        "use_bin":         pick("use_bin", False),
    }
    return _REGISTRY[model_type](**params)
