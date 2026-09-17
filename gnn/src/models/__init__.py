from functools import partial

from src.models.base import GNNClassifier
from src.models.gcn import GCN
from src.models.gat import GAT
from src.models.sage import GraphSAGE
from src.models.cgnn import CGNN
from src.models.stgcn import STGCN
from src.models.recurrent_sparse_sthnn import RecurrentSparseSTHNNClassifier
from src.graph.adjacency import load_labeled_adjacency

_RECURRENT_TYPES = frozenset({"recurrent_sparse_sthnn"})

_REGISTRY = {
    "gcn": GCN,
    "gat": GAT,
    "sage": GraphSAGE,
    "cgnn": CGNN,
    "cgnn_sage": partial(CGNN, conv_type="sage"),
    "cgnn_gat": partial(CGNN, conv_type="gat"),
    "stgcn": STGCN,
    "stgcn_sage": partial(STGCN, conv_type="sage"),
    "stgcn_gat": partial(STGCN, conv_type="gat"),
    "recurrent_sparse_sthnn": RecurrentSparseSTHNNClassifier,
}


def is_recurrent_sparse_model(model_type: str) -> bool:
    return model_type.lower() in _RECURRENT_TYPES


def build_model(cfg: dict) -> GNNClassifier:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY)}")

    data_cfg = cfg.get("data", {})
    n_lags = int(data_cfg.get("n_lags", 150))
    n_levels = int(data_cfg.get("n_levels", 10))
    num_nodes = 2 * n_levels * (n_lags + 1)
    extra_node_features = data_cfg.get("extra_node_features") or []
    if isinstance(extra_node_features, str):
        extra_node_features = [
            part.strip() for part in extra_node_features.split(",") if part.strip()
        ]
    node_feature_dim = int(
        data_cfg.get("node_feature_dim", 2 + len(extra_node_features))
    )

    # Per-architecture presets under model.overrides.<type> take precedence
    # over the shared model.<param> values. So choosing the architecture
    # automatically selects its tuned hyper-parameters.
    ov = (model_cfg.get("overrides") or {}).get(model_type, {}) or {}

    def pick(key, default=None):
        return ov.get(key, model_cfg.get(key, default))

    params = {
        "in_channels":     node_feature_dim,
        "hidden_channels": pick("hidden_channels"),
        "hidden_dim":      pick("hidden_dim"),
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
    if is_recurrent_sparse_model(model_type):
        adj_path = pick("adjacency_path", data_cfg.get("adj_matrix_path"))
        if adj_path is None:
            raise ValueError(
                "model.type='recurrent_sparse_sthnn' requires data.adj_matrix_path "
                "or model.adjacency_path."
            )
        params.update(
            {
                "adjacency": load_labeled_adjacency(adj_path),
                "message_iterations": pick("message_iterations", 1),
                "readout_mode": pick("readout_mode", "all_lags"),
                "readout_dropout": pick("readout_dropout", 0.15),
                "add_self_lag_edges": pick("add_self_lag_edges", False),
                "self_lag_edge_dropout": pick("self_lag_edge_dropout", 0.15),
                "use_edge_weights": pick("use_edge_weights", False),
                "edge_weight_normalization": pick("edge_weight_normalization", "mean"),
            }
        )
    return _REGISTRY[model_type](**params)
