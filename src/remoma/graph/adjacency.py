"""Validated graph loading with canonical LOB node order and content-aware caches."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_LEGACY_LABEL = re.compile(r"^(ask|bid)_(\d+)_lag_(\d+)$")
_LABEL = re.compile(r"^(ASKs|BIDs)(\d+)_lag(\d+)$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(label: object) -> str:
    value = str(label)
    match = _LEGACY_LABEL.fullmatch(value)
    if match:
        side, level, lag = match.groups()
        return f"{'ASKs' if side == 'ask' else 'BIDs'}{int(level) + 1}_lag{int(lag)}"
    if not _LABEL.fullmatch(value):
        raise ValueError(f"Unsupported LOB graph label: {value!r}.")
    return value


def load_labeled_adjacency(path: str | Path) -> pd.DataFrame:
    """Load symmetric, finite, nonnegative CSV/TSV weights with unique LOB labels.

    Legacy labels use zero-based levels (ask_0_lag_0); canonical labels use
    one-based levels (ASKs1_lag0). Matrix order is retained for recurrent models.
    """
    path = Path(path)
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() == ".tsv" else ",", index_col=0)
    if frame.empty or frame.shape[0] != frame.shape[1]:
        raise ValueError(f"Adjacency must be a nonempty square matrix ({path}).")
    rows = [canonical_label(label) for label in frame.index]
    columns = [canonical_label(label) for label in frame.columns]
    if rows != columns or len(set(rows)) != len(rows):
        raise ValueError("Adjacency axes must have identical, unique labels in the same order.")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Adjacency weights must be finite and nonnegative.")
    if not np.allclose(values, values.T, rtol=1e-5, atol=1e-8):
        raise ValueError("Adjacency must be symmetric.")
    frame.index, frame.columns = rows, columns
    return frame


def load_tmfg_edge_index(
    csv_path: str | Path,
    cache_path: str | Path | None = None,
    *,
    n_lags: int | None = None,
    n_levels: int = 10,
) -> torch.Tensor:
    """Return binary directed COO edges in side/level/lag tensor order.

    GNNs use nonzero connectivity; recurrent models may additionally use weights.
    Old bare-tensor caches are rebuilt. The cache records the source SHA-256 and
    requested dimensions, preventing reuse across different graphs.
    """
    key = {"sha256": sha256_file(csv_path), "n_lags": n_lags, "n_levels": n_levels, "schema": 2}
    cache = Path(cache_path) if cache_path else None
    if cache is not None and cache.exists():
        try:
            saved = torch.load(cache, weights_only=True, map_location="cpu")
            if isinstance(saved, dict) and saved.get("key") == key:
                return saved["edge_index"]
        except (OSError, RuntimeError, EOFError):
            pass
    frame = load_labeled_adjacency(csv_path)
    if n_lags is None:
        n_lags = max(int(_LABEL.fullmatch(label)[3]) for label in frame.index)
    expected = [f"{side}{level}_lag{lag}" for side in ("ASKs", "BIDs")
                for level in range(1, n_levels + 1) for lag in range(n_lags + 1)]
    if set(frame.index) != set(expected):
        raise ValueError("Adjacency labels do not match data.n_levels/data.n_lags.")
    values = frame.loc[expected, expected].to_numpy(dtype=np.float32)
    edges = torch.from_numpy(values).nonzero(as_tuple=False).t().contiguous().long()
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_suffix(".tmp.pt")
        torch.save({"key": key, "edge_index": edges}, temporary)
        temporary.replace(cache)
    return edges
