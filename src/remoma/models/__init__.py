"""Common model factory for the independently maintained model families."""
from remoma.models.base import GNNClassifier
from remoma.models.gnn import REGISTRY as GNN_REGISTRY, STATIC_TYPES
from remoma.models.recurrent import REGISTRY as RECURRENT_REGISTRY
from remoma.graph.adjacency import load_labeled_adjacency
from remoma.dataset.preprocessing import normalize_extra_node_features

_REGISTRY = {**GNN_REGISTRY, **RECURRENT_REGISTRY}


def model_names() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def model_family(model_type: str) -> str:
    name = model_type.lower()
    if name in GNN_REGISTRY:
        return "gnn"
    if name in RECURRENT_REGISTRY:
        return "recurrent"
    raise ValueError(f"Unknown model type {model_type!r}. Choose from {model_names()}.")


def is_recurrent_sparse_model(model_type: str) -> bool:
    return model_type.lower() in RECURRENT_REGISTRY


def supports_static_batching(model_type: str) -> bool:
    return model_type.lower() in STATIC_TYPES or is_recurrent_sparse_model(model_type)


def build_model(cfg: dict) -> GNNClassifier:
    model_cfg = cfg["model"]
    model_type = model_cfg["type"].lower()
    if model_type not in _REGISTRY:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(_REGISTRY)}")

    family = model_family(model_type)
    if model_cfg.get("family", family) != family:
        raise ValueError(f"model.family must be {family!r} for {model_type!r}.")

    data_cfg = cfg.get("data", {})
    n_lags = int(data_cfg.get("n_lags", 150))
    n_levels = int(data_cfg.get("n_levels", 10))
    num_nodes = 2 * n_levels * (n_lags + 1)
    extra_node_features = normalize_extra_node_features(data_cfg.get("extra_node_features"))
    node_feature_dim = 2 + len(extra_node_features)
    if int(data_cfg.get("node_feature_dim", node_feature_dim)) != node_feature_dim:
        raise ValueError("data.node_feature_dim disagrees with extra_node_features.")

    # Per-architecture presets under model.overrides.<type> take precedence
    # over the shared model.<param> values. So choosing the architecture
    # automatically selects its tuned hyper-parameters.
    ov = (model_cfg.get("overrides") or {}).get(model_type, {}) or {}

    def pick(key, default=None):
        return ov.get(key, model_cfg.get(key, default))

    params = {
        "in_channels":     node_feature_dim,
        "hidden_channels": pick("hidden_channels", 64),
        "hidden_dim":      pick("hidden_dim"),
        "num_layers":      pick("num_layers", 2),
        "num_classes":     3,
        "dropout":         pick("dropout", 0.15),
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
    if model_type.startswith(("cgnn", "stgcn")) and (int(params["cnn_kernel"]) < 1 or int(params["cnn_kernel"]) % 2 == 0):
        raise ValueError("cnn_kernel must be a positive odd integer to preserve lag dimensions.")
    return _REGISTRY[model_type](**params)
