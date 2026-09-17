from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base import GNNClassifier


EdgeKey = tuple[str, str, int]
READOUT_MODE_LAG0 = "lag0"
READOUT_MODE_ALL_LAGS = "all_lags"
SUPPORTED_READOUT_MODES = frozenset({READOUT_MODE_LAG0, READOUT_MODE_ALL_LAGS})
_LAGGED_LABEL_PATTERN = re.compile(r"^(.+)_lag(\d+)$")


@dataclass(frozen=True)
class RecurrentSparseNode:
    label: str
    feature: str
    lag: int


@dataclass(frozen=True)
class MessageEdge:
    source_idx: int
    target_idx: int
    block_idx: int
    weight: float = 1.0


@dataclass(frozen=True)
class MessageGraph:
    labels: list[str]
    nodes: list[RecurrentSparseNode]
    edge_keys: list[EdgeKey]
    temporal_edges: list[MessageEdge]
    same_lag_edges: list[MessageEdge]
    synthetic_temporal_edges: list[MessageEdge]


class EdgeMessageDropout(nn.Module):
    def __init__(self, dropout_p: float) -> None:
        super().__init__()
        self.dropout_p = float(dropout_p)

    def forward(self, edge_messages: Tensor) -> Tensor:
        if not self.training or self.dropout_p == 0:
            return edge_messages
        keep_probability = 1.0 - self.dropout_p
        mask = edge_messages.new_empty(
            edge_messages.shape[0],
            edge_messages.shape[1],
            1,
        )
        mask.bernoulli_(keep_probability).div_(keep_probability)
        return edge_messages * mask


class LagZeroReadout(nn.Module):
    def __init__(
        self,
        lag0_indices: Sequence[int],
        hidden_dim: int,
        out_dim: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        if not lag0_indices:
            raise ValueError("RecurrentSparseSTHNN requires at least one lag 0 node.")

        self.register_buffer(
            "lag0_idx",
            torch.tensor(list(lag0_indices), dtype=torch.long),
            persistent=False,
        )
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(len(lag0_indices) * hidden_dim, out_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        lag0_hidden = hidden.index_select(1, self.lag0_idx)
        flattened = lag0_hidden.reshape(hidden.shape[0], -1)
        return self.classifier(self.dropout(flattened))


class AllLagsReadout(nn.Module):
    def __init__(
        self,
        node_count: int,
        hidden_dim: int,
        out_dim: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        self.classifier = nn.Linear(node_count * hidden_dim, out_dim)

    def forward(self, hidden: Tensor) -> Tensor:
        flattened = hidden.reshape(hidden.shape[0], -1)
        return self.classifier(self.dropout(flattened))


def make_recurrent_readout(
    mode: str,
    nodes: Sequence[RecurrentSparseNode],
    hidden_dim: int,
    out_dim: int,
    dropout_p: float,
) -> nn.Module:
    if mode == READOUT_MODE_ALL_LAGS:
        return AllLagsReadout(len(nodes), hidden_dim, out_dim, dropout_p)
    if mode == READOUT_MODE_LAG0:
        lag0_indices = [idx for idx, node in enumerate(nodes) if node.lag == 0]
        return LagZeroReadout(lag0_indices, hidden_dim, out_dim, dropout_p)
    raise ValueError(
        "recurrent_readout_mode must be one of "
        f"{sorted(SUPPORTED_READOUT_MODES)}, got {mode!r}"
    )


class BaseRecurrentSparseSTHNN(nn.Module):
    """Block-sparse recurrent lag-distance network over labeled feature-lag nodes."""

    def __init__(
        self,
        graph: MessageGraph,
        *,
        hidden_dim: int = 8,
        message_iterations: int = 1,
        readout_mode: str = READOUT_MODE_ALL_LAGS,
        readout_dropout: float = 0.15,
        synthetic_edge_dropout: float = 0.0,
        in_channels: int = 2,
        use_edge_weights: bool = False,
        edge_weight_normalization: str = "mean",
        out_dim: int = 3,
    ) -> None:
        super().__init__()
        self._validate_settings(
            hidden_dim,
            message_iterations,
            readout_mode,
            readout_dropout,
            synthetic_edge_dropout,
            in_channels,
        )

        self.node_labels = graph.labels
        self.nodes = graph.nodes
        self.edge_keys = graph.edge_keys
        self.hidden_dim = int(hidden_dim)
        self.message_iterations = int(message_iterations)
        self.in_channels = int(in_channels)
        self.use_edge_weights = bool(use_edge_weights)
        self.edge_weight_normalization = str(edge_weight_normalization)
        self.out_dim = int(out_dim)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.in_channels, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.edge_blocks = nn.Parameter(
            torch.empty(len(self.edge_keys), self.hidden_dim, self.hidden_dim)
        )
        self.gru_cell = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        self.synthetic_edge_dropout = EdgeMessageDropout(synthetic_edge_dropout)
        self.readout = make_recurrent_readout(
            readout_mode,
            self.nodes,
            self.hidden_dim,
            self.out_dim,
            readout_dropout,
        )

        self._register_edges("temporal", graph.temporal_edges)
        self._register_edges("same_lag", graph.same_lag_edges)
        self._register_edges("synthetic_temporal", graph.synthetic_temporal_edges)
        self.reset_parameters()

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def reset_parameters(self) -> None:
        if self.edge_blocks.numel() > 0:
            for block in self.edge_blocks:
                nn.init.xavier_uniform_(block)

    def forward(self, x: Tensor) -> Tensor:
        node_features = self._node_features_from_input(x)
        hidden = self.node_encoder(node_features)

        first_messages = (
            self._temporal_messages(hidden)
            + self._synthetic_temporal_messages(hidden)
            + self._same_lag_messages(hidden)
        )
        hidden = self._update_hidden(first_messages, hidden)

        for _ in range(self.message_iterations - 1):
            same_lag_messages = self._same_lag_messages(hidden)
            hidden = self._update_hidden(same_lag_messages, hidden)

        logits = self.readout(hidden)
        if self.out_dim == 1:
            return logits.view(-1)
        return logits

    def _node_features_from_input(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(
                "RecurrentSparseSTHNN expects input rank 4 after temporal wrapping, "
                f"got {tuple(x.shape)}"
            )

        batch_size, channels, _, _ = x.shape
        if channels != 1:
            raise ValueError(
                "RecurrentSparseSTHNN expects a singleton input channel, "
                f"got shape {tuple(x.shape)}"
            )

        flat_features = x.squeeze(1).reshape(batch_size, -1)
        if flat_features.shape[1] % self.in_channels != 0:
            raise ValueError(
                "RecurrentSparseSTHNN input feature count is not divisible by "
                f"in_channels={self.in_channels}; got flattened feature count "
                f"{flat_features.shape[1]}."
            )

        input_nodes = flat_features.shape[1] // self.in_channels
        if input_nodes != self.node_count:
            raise ValueError(
                "RecurrentSparseSTHNN input/graph mismatch: input contains "
                f"{input_nodes} nodes, but the graph has "
                f"{self.node_count} nodes."
            )

        return flat_features.reshape(batch_size, self.node_count, self.in_channels)

    def _temporal_messages(self, hidden: Tensor) -> Tensor:
        return self._messages(
            hidden,
            self.temporal_src_idx,
            self.temporal_dst_idx,
            self.temporal_block_idx,
            self.temporal_weight,
        )

    def _same_lag_messages(self, hidden: Tensor) -> Tensor:
        return self._messages(
            hidden,
            self.same_lag_src_idx,
            self.same_lag_dst_idx,
            self.same_lag_block_idx,
            self.same_lag_weight,
        )

    def _synthetic_temporal_messages(self, hidden: Tensor) -> Tensor:
        return self._messages(
            hidden,
            self.synthetic_temporal_src_idx,
            self.synthetic_temporal_dst_idx,
            self.synthetic_temporal_block_idx,
            self.synthetic_temporal_weight,
            edge_dropout=self.synthetic_edge_dropout,
        )

    def _messages(
        self,
        hidden: Tensor,
        source_idx: Tensor,
        target_idx: Tensor,
        block_idx: Tensor,
        edge_weight: Tensor,
        edge_dropout: EdgeMessageDropout | None = None,
    ) -> Tensor:
        messages = hidden.new_zeros(hidden.shape)
        if source_idx.numel() == 0:
            return messages

        source_hidden = hidden.index_select(1, source_idx)
        edge_blocks = self.edge_blocks.index_select(0, block_idx)
        edge_messages = torch.einsum("bed,eod->beo", source_hidden, edge_blocks)
        edge_messages = edge_messages * edge_weight.view(1, -1, 1)
        if edge_dropout is not None:
            edge_messages = edge_dropout(edge_messages)
        scatter_idx = target_idx.view(1, -1, 1).expand(
            hidden.shape[0],
            -1,
            self.hidden_dim,
        )
        messages.scatter_add_(1, scatter_idx, edge_messages)
        return messages

    def _update_hidden(self, messages: Tensor, hidden: Tensor) -> Tensor:
        batch_size = hidden.shape[0]
        updated = self.gru_cell(
            messages.reshape(batch_size * self.node_count, self.hidden_dim),
            hidden.reshape(batch_size * self.node_count, self.hidden_dim),
        )
        return updated.reshape(batch_size, self.node_count, self.hidden_dim)

    def _register_edges(self, prefix: str, edges: Sequence[MessageEdge]) -> None:
        source_idx, target_idx, block_idx, weight = _edge_tensors(edges)
        self.register_buffer(f"{prefix}_src_idx", source_idx, persistent=False)
        self.register_buffer(f"{prefix}_dst_idx", target_idx, persistent=False)
        self.register_buffer(f"{prefix}_block_idx", block_idx, persistent=False)
        self.register_buffer(f"{prefix}_weight", weight, persistent=False)

    @staticmethod
    def _validate_settings(
        hidden_dim: int,
        message_iterations: int,
        readout_mode: str,
        readout_dropout: float,
        synthetic_edge_dropout: float,
        in_channels: int,
    ) -> None:
        if in_channels < 1:
            raise ValueError(f"recurrent in_channels must be >= 1, got {in_channels}")
        if hidden_dim < 1:
            raise ValueError(f"recurrent_hidden_dim must be >= 1, got {hidden_dim}")
        if message_iterations < 1:
            raise ValueError(
                "recurrent_message_iterations must be >= 1, "
                f"got {message_iterations}"
            )
        if readout_mode not in SUPPORTED_READOUT_MODES:
            raise ValueError(
                "recurrent_readout_mode must be one of "
                f"{sorted(SUPPORTED_READOUT_MODES)}, got {readout_mode!r}"
            )
        if readout_dropout < 0 or readout_dropout >= 1:
            raise ValueError(
                "recurrent_readout_dropout must be in [0, 1), "
                f"got {readout_dropout}"
            )
        if synthetic_edge_dropout < 0 or synthetic_edge_dropout >= 1:
            raise ValueError(
                "recurrent_self_lag_edge_dropout must be in [0, 1), "
                f"got {synthetic_edge_dropout}"
            )


class RecurrentSparseSTHNN(BaseRecurrentSparseSTHNN):
    def __init__(
        self,
        adjacency: object,
        *,
        hidden_dim: int = 8,
        message_iterations: int = 1,
        readout_mode: str = READOUT_MODE_ALL_LAGS,
        readout_dropout: float = 0.15,
        in_channels: int = 2,
        use_edge_weights: bool = False,
        edge_weight_normalization: str = "mean",
        out_dim: int = 3,
    ) -> None:
        super().__init__(
            build_message_graph(
                adjacency,
                use_edge_weights=use_edge_weights,
                edge_weight_normalization=edge_weight_normalization,
            ),
            hidden_dim=hidden_dim,
            message_iterations=message_iterations,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            in_channels=in_channels,
            use_edge_weights=use_edge_weights,
            edge_weight_normalization=edge_weight_normalization,
            out_dim=out_dim,
        )


class SelfLagRecurrentSparseSTHNN(BaseRecurrentSparseSTHNN):
    def __init__(
        self,
        adjacency: object,
        *,
        hidden_dim: int = 8,
        message_iterations: int = 1,
        readout_mode: str = READOUT_MODE_ALL_LAGS,
        readout_dropout: float = 0.15,
        add_self_lag_edges: bool = True,
        self_lag_edge_dropout: float = 0.15,
        in_channels: int = 2,
        use_edge_weights: bool = False,
        edge_weight_normalization: str = "mean",
        out_dim: int = 3,
    ) -> None:
        base_graph = build_message_graph(
            adjacency,
            use_edge_weights=use_edge_weights,
            edge_weight_normalization=edge_weight_normalization,
        )
        graph = add_missing_self_lag_edges(base_graph) if add_self_lag_edges else base_graph
        super().__init__(
            graph,
            hidden_dim=hidden_dim,
            message_iterations=message_iterations,
            readout_mode=readout_mode,
            readout_dropout=readout_dropout,
            synthetic_edge_dropout=self_lag_edge_dropout if add_self_lag_edges else 0.0,
            in_channels=in_channels,
            use_edge_weights=use_edge_weights,
            edge_weight_normalization=edge_weight_normalization,
            out_dim=out_dim,
        )


class RecurrentSparseSTHNNClassifier(GNNClassifier):
    """Pipeline adapter for RecurrentSparseSTHNN.

    Processed LOB tensors are stored in the project's historical node order:
    all ask levels by lag, then all bid levels by lag. The recurrent adjacency
    file defines its own labeled order. This adapter performs an explicit
    label-based reorder before invoking the recurrent core.
    """

    def __init__(
        self,
        adjacency: object,
        in_channels: int = 2,
        hidden_channels: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        num_classes: int = 3,
        dropout: float | None = None,
        num_nodes: int | None = None,
        n_lags: int = 100,
        n_levels: int = 10,
        message_iterations: int = 1,
        readout_mode: str = READOUT_MODE_ALL_LAGS,
        readout_dropout: float = 0.15,
        add_self_lag_edges: bool = False,
        self_lag_edge_dropout: float = 0.15,
        use_edge_weights: bool = False,
        edge_weight_normalization: str = "mean",
        **_kwargs,
    ) -> None:
        super().__init__()
        del dropout, num_layers
        in_channels = int(in_channels)
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got {in_channels}")

        recurrent_hidden_dim = int(
            hidden_dim
            if hidden_dim is not None
            else hidden_channels
            if hidden_channels is not None
            else 5
        )
        if add_self_lag_edges:
            self.core = SelfLagRecurrentSparseSTHNN(
                adjacency,
                hidden_dim=recurrent_hidden_dim,
                message_iterations=message_iterations,
                readout_mode=readout_mode,
                readout_dropout=readout_dropout,
                add_self_lag_edges=add_self_lag_edges,
                self_lag_edge_dropout=self_lag_edge_dropout,
                in_channels=in_channels,
                use_edge_weights=use_edge_weights,
                edge_weight_normalization=edge_weight_normalization,
                out_dim=num_classes,
            )
        else:
            self.core = RecurrentSparseSTHNN(
                adjacency,
                hidden_dim=recurrent_hidden_dim,
                message_iterations=message_iterations,
                readout_mode=readout_mode,
                readout_dropout=readout_dropout,
                in_channels=in_channels,
                use_edge_weights=use_edge_weights,
                edge_weight_normalization=edge_weight_normalization,
                out_dim=num_classes,
            )

        expected_nodes = 2 * int(n_levels) * (int(n_lags) + 1)
        if num_nodes is not None and int(num_nodes) != expected_nodes:
            raise ValueError(
                f"num_nodes={num_nodes} disagrees with n_levels={n_levels}, "
                f"n_lags={n_lags} -> {expected_nodes} nodes."
            )
        if self.core.node_count != expected_nodes:
            raise ValueError(
                "Adjacency node count does not match processed tensor shape: "
                f"adjacency has {self.core.node_count} nodes, config implies "
                f"{expected_nodes}. Check data.n_lags/data.n_levels and "
                "data.adj_matrix_path."
            )

        reorder_idx = _preprocessed_lob_to_adjacency_order(
            self.core.node_labels,
            n_lags=int(n_lags),
            n_levels=int(n_levels),
        )
        self.register_buffer(
            "node_reorder_idx",
            torch.tensor(reorder_idx, dtype=torch.long),
            persistent=False,
        )
        self.num_nodes = expected_nodes
        self.n_lags = int(n_lags)
        self.n_levels = int(n_levels)
        self.in_channels = in_channels
        self.use_edge_weights = bool(use_edge_weights)
        self.edge_weight_normalization = str(edge_weight_normalization)

        # Useful metadata for logging/debugging.
        self.recurrent_edge_key_count = len(self.core.edge_keys)
        self.recurrent_temporal_edge_count = int(self.core.temporal_src_idx.numel())
        self.recurrent_same_lag_edge_count = int(self.core.same_lag_src_idx.numel())
        self.recurrent_synthetic_edge_count = int(
            self.core.synthetic_temporal_src_idx.numel()
        )

    def forward_static(self, x_batch: Tensor, edge_index: Tensor | None = None) -> Tensor:
        del edge_index
        if x_batch.ndim != 3:
            raise ValueError(
                "RecurrentSparseSTHNNClassifier.forward_static expects "
                f"[B, N, F], got {tuple(x_batch.shape)}."
            )
        if x_batch.shape[1] != self.num_nodes or x_batch.shape[2] != self.in_channels:
            raise ValueError(
                "RecurrentSparseSTHNNClassifier input must be [B, "
                f"{self.num_nodes}, {self.in_channels}], got "
                f"{tuple(x_batch.shape)}."
            )
        x_ordered = x_batch.index_select(1, self.node_reorder_idx)
        return self.core(x_ordered.unsqueeze(1))

    def forward(self, data) -> Tensor:
        x = data.x
        num_graphs = int(data.num_graphs)
        x_batch = x.reshape(num_graphs, self.num_nodes, -1)
        return self.forward_static(x_batch)


def build_message_graph(
    adjacency: object,
    *,
    use_edge_weights: bool = False,
    edge_weight_normalization: str = "mean",
) -> MessageGraph:
    labels = _node_labels_from_adjacency(adjacency)
    nodes = _parse_lagged_nodes(labels)
    node_index = {(node.feature, node.lag): idx for idx, node in enumerate(nodes)}
    adjacency_matrix = _adjacency_matrix(adjacency, expected_size=len(labels))
    if not use_edge_weights:
        adjacency_matrix = (adjacency_matrix != 0).astype(float)
    elif edge_weight_normalization not in {"none", "mean", "max"}:
        raise ValueError(
            "edge_weight_normalization must be one of: none, mean, max; "
            f"got {edge_weight_normalization!r}"
        )

    edge_keys: list[EdgeKey] = []
    key_index: dict[EdgeKey, int] = {}
    temporal_edges: list[MessageEdge] = []
    same_lag_edges: list[MessageEdge] = []
    temporal_seen: set[tuple[int, int, int]] = set()
    same_lag_seen: set[tuple[int, int, int]] = set()
    lags = sorted({node.lag for node in nodes})

    for row_idx in range(len(nodes)):
        for col_idx in range(row_idx + 1, len(nodes)):
            forward_weight = adjacency_matrix[row_idx, col_idx]
            reverse_weight = adjacency_matrix[col_idx, row_idx]
            if forward_weight == 0 and reverse_weight == 0:
                continue
            edge_weight = _undirected_edge_weight(forward_weight, reverse_weight)

            left = nodes[row_idx]
            right = nodes[col_idx]
            if left.lag == right.lag:
                _add_same_lag_edges(
                    left.feature,
                    right.feature,
                    lags,
                    node_index,
                    edge_keys,
                    key_index,
                    same_lag_edges,
                    same_lag_seen,
                    edge_weight,
                )
            else:
                target, source = _temporal_target_source(left, right)
                _add_temporal_edges(
                    target.feature,
                    source.feature,
                    source.lag - target.lag,
                    lags,
                    node_index,
                    edge_keys,
                    key_index,
                    temporal_edges,
                    temporal_seen,
                    edge_weight,
                )

    if use_edge_weights:
        temporal_edges = _normalize_edge_weights(
            temporal_edges,
            edge_weight_normalization,
        )
        same_lag_edges = _normalize_edge_weights(
            same_lag_edges,
            edge_weight_normalization,
        )

    return MessageGraph(
        labels=labels,
        nodes=nodes,
        edge_keys=edge_keys,
        temporal_edges=temporal_edges,
        same_lag_edges=same_lag_edges,
        synthetic_temporal_edges=[],
    )


def add_missing_self_lag_edges(graph: MessageGraph) -> MessageGraph:
    edge_keys = list(graph.edge_keys)
    key_index = {key: idx for idx, key in enumerate(edge_keys)}
    node_index = {(node.feature, node.lag): idx for idx, node in enumerate(graph.nodes)}
    regular_temporal_edges = {
        (edge.source_idx, edge.target_idx) for edge in graph.temporal_edges
    }
    synthetic_edges: list[MessageEdge] = []
    synthetic_seen: set[tuple[int, int, int]] = set()
    lags = sorted({node.lag for node in graph.nodes})
    features = sorted({node.feature for node in graph.nodes})

    for feature in features:
        block_idx: int | None = None
        for target_lag in lags:
            source_lag = target_lag + 1
            target_idx = node_index.get((feature, target_lag))
            source_idx = node_index.get((feature, source_lag))
            if target_idx is None or source_idx is None:
                continue
            if (source_idx, target_idx) in regular_temporal_edges:
                continue
            if block_idx is None:
                block_idx = _edge_key_index(
                    (feature, feature, 1),
                    edge_keys,
                    key_index,
                )
            _append_edge(
                synthetic_edges,
                synthetic_seen,
                source_idx,
                target_idx,
                block_idx,
            )

    return MessageGraph(
        labels=graph.labels,
        nodes=graph.nodes,
        edge_keys=edge_keys,
        temporal_edges=graph.temporal_edges,
        same_lag_edges=graph.same_lag_edges,
        synthetic_temporal_edges=synthetic_edges,
    )


def _node_labels_from_adjacency(adjacency: object) -> list[str]:
    index = getattr(adjacency, "index", None)
    columns = getattr(adjacency, "columns", None)
    if index is None or columns is None:
        raise ValueError(
            "RecurrentSparseSTHNN requires a labeled adjacency DataFrame with "
            "matching index and column labels like 'ASKs1_lag5'."
        )

    labels = [str(label) for label in index]
    column_labels = [str(label) for label in columns]
    if labels != column_labels:
        raise ValueError("Adjacency index and columns must use the same node order.")
    return labels


def _parse_lagged_nodes(labels: Sequence[str]) -> list[RecurrentSparseNode]:
    nodes: list[RecurrentSparseNode] = []
    seen: set[tuple[str, int]] = set()
    for label in labels:
        match = _LAGGED_LABEL_PATTERN.fullmatch(label)
        if match is None:
            raise ValueError(
                "RecurrentSparseSTHNN requires node labels ending in '_lagN'; "
                f"got {label!r}"
            )

        feature = match.group(1)
        lag = int(match.group(2))
        identity = (feature, lag)
        if identity in seen:
            raise ValueError(f"Duplicate lagged node label identity: {label!r}")
        seen.add(identity)
        nodes.append(RecurrentSparseNode(label=label, feature=feature, lag=lag))
    return nodes


def _adjacency_matrix(adjacency: object, expected_size: int) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=float)
    if matrix.ndim != 2 or matrix.shape != (expected_size, expected_size):
        raise ValueError(
            "Adjacency must be square and match labeled node count; "
            f"got shape {matrix.shape} for {expected_size} labels"
        )
    return matrix


def _undirected_edge_weight(forward_weight: float, reverse_weight: float) -> float:
    """Collapse a possibly asymmetric adjacency entry into one scalar weight."""
    weights = [abs(float(w)) for w in (forward_weight, reverse_weight) if w != 0]
    return max(weights) if weights else 1.0


def _normalize_edge_weights(
    edges: Sequence[MessageEdge],
    mode: str,
) -> list[MessageEdge]:
    if mode == "none" or not edges:
        return list(edges)
    weights = np.asarray([edge.weight for edge in edges], dtype=np.float32)
    if mode == "mean":
        scale = float(weights.mean())
    elif mode == "max":
        scale = float(weights.max())
    else:
        raise ValueError(
            "edge_weight_normalization must be one of: none, mean, max; "
            f"got {mode!r}"
        )
    if scale <= 0 or not np.isfinite(scale):
        return list(edges)
    return [
        MessageEdge(
            source_idx=edge.source_idx,
            target_idx=edge.target_idx,
            block_idx=edge.block_idx,
            weight=float(edge.weight / scale),
        )
        for edge in edges
    ]


def _add_same_lag_edges(
    first_feature: str,
    second_feature: str,
    lags: Sequence[int],
    node_index: dict[tuple[str, int], int],
    edge_keys: list[EdgeKey],
    key_index: dict[EdgeKey, int],
    edges: list[MessageEdge],
    seen: set[tuple[int, int, int]],
    weight: float,
) -> None:
    for target_feature, source_feature in (
        (first_feature, second_feature),
        (second_feature, first_feature),
    ):
        block_idx = _edge_key_index(
            (target_feature, source_feature, 0),
            edge_keys,
            key_index,
        )
        for lag in lags:
            target_idx = node_index.get((target_feature, lag))
            source_idx = node_index.get((source_feature, lag))
            if target_idx is None or source_idx is None:
                continue
            _append_edge(edges, seen, source_idx, target_idx, block_idx, weight)


def _add_temporal_edges(
    target_feature: str,
    source_feature: str,
    lag_distance: int,
    lags: Sequence[int],
    node_index: dict[tuple[str, int], int],
    edge_keys: list[EdgeKey],
    key_index: dict[EdgeKey, int],
    edges: list[MessageEdge],
    seen: set[tuple[int, int, int]],
    weight: float,
) -> None:
    block_idx = _edge_key_index(
        (target_feature, source_feature, lag_distance),
        edge_keys,
        key_index,
    )
    for target_lag in lags:
        source_lag = target_lag + lag_distance
        target_idx = node_index.get((target_feature, target_lag))
        source_idx = node_index.get((source_feature, source_lag))
        if target_idx is None or source_idx is None:
            continue
        _append_edge(edges, seen, source_idx, target_idx, block_idx, weight)


def _edge_key_index(
    key: EdgeKey,
    edge_keys: list[EdgeKey],
    key_index: dict[EdgeKey, int],
) -> int:
    if key not in key_index:
        key_index[key] = len(edge_keys)
        edge_keys.append(key)
    return key_index[key]


def _append_edge(
    edges: list[MessageEdge],
    seen: set[tuple[int, int, int]],
    source_idx: int,
    target_idx: int,
    block_idx: int,
    weight: float = 1.0,
) -> None:
    identity = (source_idx, target_idx, block_idx)
    if identity in seen:
        return
    seen.add(identity)
    edges.append(
        MessageEdge(
            source_idx=source_idx,
            target_idx=target_idx,
            block_idx=block_idx,
            weight=float(weight),
        )
    )


def _temporal_target_source(
    left: RecurrentSparseNode,
    right: RecurrentSparseNode,
) -> tuple[RecurrentSparseNode, RecurrentSparseNode]:
    if left.lag < right.lag:
        return left, right
    return right, left


def _edge_tensors(
    edges: Sequence[MessageEdge],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if not edges:
        empty = torch.empty(0, dtype=torch.long)
        empty_weight = torch.empty(0, dtype=torch.float32)
        return empty, empty, empty, empty_weight
    return (
        torch.tensor([edge.source_idx for edge in edges], dtype=torch.long),
        torch.tensor([edge.target_idx for edge in edges], dtype=torch.long),
        torch.tensor([edge.block_idx for edge in edges], dtype=torch.long),
        torch.tensor([edge.weight for edge in edges], dtype=torch.float32),
    )


def _preprocessed_lob_to_adjacency_order(
    adjacency_labels: Sequence[str],
    *,
    n_lags: int,
    n_levels: int,
) -> list[int]:
    source_labels = _preprocessed_lob_labels(n_lags=n_lags, n_levels=n_levels)
    source_index = {label: idx for idx, label in enumerate(source_labels)}

    missing = [label for label in adjacency_labels if label not in source_index]
    extras = [label for label in source_labels if label not in set(adjacency_labels)]
    if missing or extras:
        raise ValueError(
            "Adjacency labels do not match the processed LOB tensor labels. "
            f"Missing from processed tensors: {missing[:5]}; "
            f"extra processed labels: {extras[:5]}."
        )
    return [source_index[label] for label in adjacency_labels]


def _preprocessed_lob_labels(*, n_lags: int, n_levels: int) -> list[str]:
    labels: list[str] = []
    for side in ("ASKs", "BIDs"):
        for level in range(1, n_levels + 1):
            for lag in range(n_lags + 1):
                labels.append(f"{side}{level}_lag{lag}")
    return labels
