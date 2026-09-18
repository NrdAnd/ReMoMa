from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap

from remoma.dataset.binning import VolumeBinner
from remoma.graph.adjacency import sha256_file
from remoma.dataset.labeling import compute_labels
from remoma.dataset.order_flow import load_message_csv, per_event_order_flow
from remoma.utils.io import configured_orderbooks, discover_message_files, load_lobster_csv, cached_orderbook

# LOBSTER 40-column layout (10 levels, ask/bid, price/volume)
_ASK_P_COLS = np.arange(0, 40, 4)
_BID_P_COLS = np.arange(2, 40, 4)
_ASK_V_COLS = np.arange(1, 40, 4)
_BID_V_COLS = np.arange(3, 40, 4)
_EXTRA_NODE_FEATURES = frozenset(
    {
        "spread",
        "level_imbalance",
        "depth_imbalance",
        "microprice",
        "order_cancel_flow",
        "order_limit_flow",
        "order_trade_flow",
        "side",
    }
)
_ORDER_FLOW_FEATURE_TO_COL = {
    "order_trade_flow": 0,
    "order_limit_flow": 1,
    "order_cancel_flow": 2,
}


def normalize_extra_node_features(value: Any) -> list[str]:
    """Normalize and validate optional per-node engineered feature names."""
    if value is None:
        return []
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",")]
    else:
        raw = [str(part).strip() for part in value]
    features = [part.lower() for part in raw if part]
    unknown = sorted(set(features) - _EXTRA_NODE_FEATURES)
    if unknown:
        raise ValueError(
            "Unknown extra_node_features values "
            f"{unknown}. Supported: {sorted(_EXTRA_NODE_FEATURES)}"
        )
    if len(set(features)) != len(features):
        raise ValueError("extra_node_features must not contain duplicates.")
    return features


def uses_order_flow_features(extra_node_features: list[str]) -> bool:
    return any(name in _ORDER_FLOW_FEATURE_TO_COL for name in extra_node_features)


def _make_samples(
    file_indices: list[int],
    all_data: list[np.ndarray],
    n_lags: int,
    k: int,
) -> np.ndarray:
    """Vectorized construction of (file_idx, t) sample pairs."""
    parts = []
    for fi in file_indices:
        t_start = n_lags
        t_end = len(all_data[fi]) - 1 - k  # inclusive
        n = t_end - t_start + 1
        if n <= 0:
            continue
        parts.append(
            np.stack(
                [
                    np.full(n, fi, dtype=np.int32),
                    np.arange(t_start, t_end + 1, dtype=np.int32),
                ],
                axis=1,
            )
        )
    if not parts:
        return np.empty((0, 2), dtype=np.int32)
    return np.concatenate(parts, axis=0)


def split_by_file(
    all_data: list[np.ndarray],
    cfg: dict[str, Any],
    n_lags: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split chronologically using disjoint, explicit file index groups."""
    groups = [cfg["data"][key] for key in ("train_files", "val_files", "test_files")]
    flat = [index for group in groups for index in group]
    if any(not group for group in groups) or len(flat) != len(set(flat)):
        raise ValueError("File splits must be nonempty and disjoint, without duplicate indices.")
    if any(not isinstance(index, int) or index < 0 or index >= len(all_data) for index in flat):
        raise ValueError("File split index is outside the discovered file list.")
    if max(groups[0]) >= min(groups[1]) or max(groups[1]) >= min(groups[2]):
        raise ValueError("File splits must be chronological: train before validation before test.")
    return (
        _make_samples(cfg["data"]["train_files"], all_data, n_lags, k),
        _make_samples(cfg["data"]["val_files"], all_data, n_lags, k),
        _make_samples(cfg["data"]["test_files"], all_data, n_lags, k),
    )


def split_by_lag(
    all_data: list[np.ndarray],
    cfg: dict[str, Any],
    n_lags: int,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split each file chronologically, purging overlapping inputs and targets.

    A sample covers raw rows [t-n_lags, t+k]. Removing n_lags+k samples
    before each boundary makes adjacent splits disjoint in raw row space.
    """
    r_train = cfg["data"]["train_ratio"]
    r_val = cfg["data"]["val_ratio"]
    train_parts, val_parts, test_parts = [], [], []

    for fi, data in enumerate(all_data):
        t_start = n_lags
        t_end = len(data) - 1 - k
        n = t_end - t_start + 1
        if n <= 0:
            continue
        t1 = t_start + int(r_train * n)
        t2 = t_start + int((r_train + r_val) * n)

        def _block(a: int, b: int) -> np.ndarray:
            m = b - a
            if m <= 0:
                return np.empty((0, 2), dtype=np.int32)
            return np.stack(
                [np.full(m, fi, dtype=np.int32), np.arange(a, b, dtype=np.int32)],
                axis=1,
            )

        train_parts.append(_block(t_start, t1 - n_lags - k))
        val_parts.append(_block(t1, t2 - n_lags - k))
        test_parts.append(_block(t2, t_end + 1))

    def _cat(parts: list[np.ndarray]) -> np.ndarray:
        parts = [p for p in parts if len(p) > 0]
        if not parts:
            return np.empty((0, 2), dtype=np.int32)
        return np.concatenate(parts, axis=0)

    return _cat(train_parts), _cat(val_parts), _cat(test_parts)


def _subsample(samples: np.ndarray, max_n: int | None, seed: int) -> np.ndarray:
    """Randomly subsample rows, preserving order."""
    if max_n is None or len(samples) <= max_n:
        return samples
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(samples), size=max_n, replace=False)
    return samples[np.sort(idx)]


def _balance_samples(
    samples: np.ndarray,
    all_labels: list[np.ndarray],
    max_per_class: int | None,
    seed: int,
) -> np.ndarray:
    """Subsample train_s to have equal class representation."""
    labels = np.array([all_labels[int(fi)][int(t)] for fi, t in samples])
    counts = np.bincount(labels, minlength=3)
    n = int(counts.min())
    if max_per_class is not None:
        n = min(n, int(max_per_class))
    rng = np.random.default_rng(seed)
    chosen = np.concatenate([
        rng.choice(np.where(labels == c)[0], size=n, replace=False)
        for c in range(3)
    ])
    rng.shuffle(chosen)
    return samples[chosen]


def build_price_stats(train_files: list[int], all_data: list[np.ndarray]) -> dict[str, np.ndarray]:
    """Legacy artifact kept for compatibility with older checkpoints/scripts."""
    data = np.concatenate([all_data[i] for i in train_files], axis=0)
    return {
        "ask_mean": data[:, _ASK_P_COLS].mean(0).astype(np.float32),
        "ask_std": data[:, _ASK_P_COLS].std(0).astype(np.float32),
        "bid_mean": data[:, _BID_P_COLS].mean(0).astype(np.float32),
        "bid_std": data[:, _BID_P_COLS].std(0).astype(np.float32),
    }


def build_processed_paths(processed_dir: str | Path) -> dict[str, Path]:
    root = Path(processed_dir)
    return {
        "dir": root,
        "X_train": root / "X_train.npy",
        "y_train": root / "y_train.npy",
        "X_val": root / "X_val.npy",
        "y_val": root / "y_val.npy",
        "X_test": root / "X_test.npy",
        "y_test": root / "y_test.npy",
        "binner": root / "binner.pkl",
        "price_stats": root / "price_stats.npy",
        "meta": root / "meta.json",
    }


def preprocess_signature(cfg: dict[str, Any]) -> dict[str, Any]:
    """Configuration subset that defines processed-dataset semantics."""
    data_cfg = cfg["data"]
    split = data_cfg["split_strategy"]
    extra_node_features = normalize_extra_node_features(
        data_cfg.get("extra_node_features")
    )
    sig: dict[str, Any] = {
        "schema_version": 3,
        "storage_mode": data_cfg.get("storage_mode", "materialized"),
        "subsample_seed": int(data_cfg.get("subsample_seed", 42)),
        "n_lags": int(data_cfg["n_lags"]),
        "n_levels": int(data_cfg["n_levels"]),
        "n_volume_bins": int(data_cfg["n_volume_bins"]),
        "prediction_horizon": int(data_cfg.get("prediction_horizon", 1)),
        "threshold": float(data_cfg["threshold"]),
        "price_type": str(data_cfg.get("price_type", "mid")),
        "label_mode": str(data_cfg.get("label_mode", "pct")),
        "split_strategy": split,
        "normalize_prices": bool(data_cfg.get("normalize_prices", True)),
        "feature_dtype": str(data_cfg.get("feature_dtype", "float16")),
        "extra_node_features": extra_node_features,
        "node_feature_dim": 2 + len(extra_node_features),
    }
    if uses_order_flow_features(extra_node_features):
        sig["message_dir"] = str(data_cfg.get("message_dir", data_cfg["raw_dir"]))
        sig["message_pattern"] = str(data_cfg.get("message_pattern", "*_message_10.csv"))
    if split == "by_file":
        sig["train_files"] = [int(x) for x in data_cfg["train_files"]]
        sig["val_files"] = [int(x) for x in data_cfg["val_files"]]
        sig["test_files"] = [int(x) for x in data_cfg["test_files"]]
    elif split == "by_lag":
        sig["train_ratio"] = float(data_cfg["train_ratio"])
        sig["val_ratio"] = float(data_cfg["val_ratio"])
    else:
        raise ValueError(f"Unknown split_strategy '{split}'")
    mpc = data_cfg.get("max_samples_per_class")
    sig["max_samples_per_class"] = int(mpc) if mpc is not None else None
    mvs = data_cfg.get("max_val_samples")
    sig["max_val_samples"] = int(mvs) if mvs is not None else None
    mts = data_cfg.get("max_test_samples")
    sig["max_test_samples"] = int(mts) if mts is not None else None
    return sig


def file_manifest(paths: list[Path]) -> list[dict[str, Any]]:
    """Record content hashes, allowing detection of changes under the same name."""
    return [{"name": path.name, "sha256": sha256_file(path)} for path in paths]


def has_compatible_processed_dataset(cfg: dict[str, Any]) -> bool:
    """Return True if all processed artifacts exist and match config signature."""
    paths = build_processed_paths(cfg["data"]["processed_dir"])
    needed = [
        paths["X_train"],
        paths["y_train"],
        paths["X_val"],
        paths["y_val"],
        paths["X_test"],
        paths["y_test"],
        paths["binner"],
        paths["meta"],
    ]
    if not all(p.exists() for p in needed):
        return False

    try:
        with open(paths["meta"], "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    if meta.get("signature") != preprocess_signature(cfg):
        return False

    try:
        raw_paths = configured_orderbooks(cfg["data"])
    except FileNotFoundError:
        if not cfg["data"].get("allow_missing_raw", False):
            return False
        raw_paths = []
    if not raw_paths:
        if not cfg["data"].get("allow_missing_raw", False):
            return False
    else:
        if meta.get("raw_manifest") != file_manifest(raw_paths):
            return False
        extras = normalize_extra_node_features(cfg["data"].get("extra_node_features"))
        if uses_order_flow_features(extras):
            try:
                message_paths = discover_message_files(
                    cfg["data"].get("message_dir", cfg["data"]["raw_dir"]), raw_paths,
                    cfg["data"].get("message_pattern", "*_message_10.csv"),
                )
            except FileNotFoundError:
                return False
            if meta.get("message_manifest") != file_manifest(message_paths):
                return False
    try:
        for split in ("train", "val", "test"):
            x = np.load(paths[f"X_{split}"], mmap_mode="r", allow_pickle=False)
            y = np.load(paths[f"y_{split}"], mmap_mode="r", allow_pickle=False)
            expected = ((meta["counts"][split], 2) if meta.get("storage_mode") == "indexed"
                        else (meta["counts"][split], meta["n_nodes"], meta["n_features"]))
            if x.shape != expected or y.shape != (expected[0],) or expected[0] == 0:
                return False
            x_dtype = np.dtype("int32" if meta.get("storage_mode") == "indexed" else meta["feature_dtype"])
            if x.dtype != x_dtype or y.dtype != np.int64:
                return False
        for artifact in meta.get("indexed_artifacts", []):
            if sha256_file(artifact["path"]) != artifact["sha256"]:
                return False
        for name, digest in meta.get("artifact_hashes", {}).items():
            if sha256_file(paths["dir"] / name) != digest:
                return False
    except (OSError, ValueError, KeyError):
        return False

    return True


def _extract_chunk_features(
    data: np.ndarray,
    t_idx: np.ndarray,
    n_lags: int,
    binner: VolumeBinner,
    normalize_prices: bool,
    order_flow: np.ndarray | None = None,
) -> np.ndarray:
    """Build feature blocks for a chunk of timestamps with vectorized indexing.

    Returns a tensor shaped [chunk_size, n_nodes, n_features] with node ordering
    ask levels by lag, then bid levels by lag.
    """
    return _extract_chunk_features_with_extras(
        data=data,
        t_idx=t_idx,
        n_lags=n_lags,
        binner=binner,
        normalize_prices=normalize_prices,
        extra_node_features=[],
        order_flow=order_flow,
    )


def _extract_chunk_features_with_extras(
    data: np.ndarray,
    t_idx: np.ndarray,
    n_lags: int,
    binner: VolumeBinner,
    normalize_prices: bool,
    extra_node_features: list[str],
    order_flow: np.ndarray | None = None,
    binned_volumes: np.ndarray | None = None,
) -> np.ndarray:
    """Build base price/volume channels plus optional engineered node channels."""
    # idx[b, k] = timestamp for lag k (k=0 => t, k=n_lags => oldest)
    lags = np.arange(n_lags + 1, dtype=np.intp)[None, :]
    idx = t_idx[:, None].astype(np.intp) - lags
    windows = data[idx]  # [B, n_lags+1, 40]

    ask_prices_raw = (
        windows[:, :, _ASK_P_COLS].transpose(0, 2, 1).astype(np.float32, copy=False)
    )
    bid_prices_raw = (
        windows[:, :, _BID_P_COLS].transpose(0, 2, 1).astype(np.float32, copy=False)
    )
    ask_vols = (
        windows[:, :, _ASK_V_COLS].transpose(0, 2, 1).astype(np.float32, copy=False)
    )
    bid_vols = (
        windows[:, :, _BID_V_COLS].transpose(0, 2, 1).astype(np.float32, copy=False)
    )

    current_mid = (ask_prices_raw[:, 0, 0] + bid_prices_raw[:, 0, 0]) * 0.5
    np.maximum(current_mid, 1.0, out=current_mid)

    if normalize_prices:
        mid = current_mid[:, None, None]
        ask_prices = (ask_prices_raw - mid) / mid
        bid_prices = (bid_prices_raw - mid) / mid
    else:
        ask_prices = ask_prices_raw
        bid_prices = bid_prices_raw

    if binned_volumes is None:
        ask_vols_b, bid_vols_b = binner.transform_window(ask_vols, bid_vols)
    else:
        # Bins are frozen per training split and computed once per raw row.
        volume_window = binned_volumes[idx]
        ask_vols_b = volume_window[:, :, :10].transpose(0, 2, 1)
        bid_vols_b = volume_window[:, :, 10:].transpose(0, 2, 1)
    order_flow_windows = None
    if uses_order_flow_features(extra_node_features):
        if order_flow is None:
            raise ValueError(
                "Order-flow node features require aligned LOBSTER message files."
            )
        order_flow_windows = order_flow[idx].transpose(0, 2, 1).astype(
            np.float32,
            copy=False,
        )

    b = len(t_idx)
    n_levels = ask_prices.shape[1]
    lag_len = n_lags + 1
    half = n_levels * lag_len
    n_nodes = 2 * half
    n_features = 2 + len(extra_node_features)

    out = np.empty((b, n_nodes, n_features), dtype=np.float32)
    out[:, :half, 0] = ask_prices.reshape(b, -1)
    out[:, :half, 1] = ask_vols_b.reshape(b, -1)
    out[:, half:, 0] = bid_prices.reshape(b, -1)
    out[:, half:, 1] = bid_vols_b.reshape(b, -1)

    def lag_feature(values: np.ndarray) -> np.ndarray:
        return np.repeat(values[:, None, :], n_levels, axis=1).reshape(b, -1)

    channel = 2
    for name in extra_node_features:
        if name == "spread":
            spread = (
                ask_prices_raw[:, 0, :] - bid_prices_raw[:, 0, :]
            ) / current_mid[:, None]
            values = lag_feature(spread)
            out[:, :half, channel] = values
            out[:, half:, channel] = values
        elif name == "level_imbalance":
            level_imbalance = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-9)
            values = level_imbalance.reshape(b, -1)
            out[:, :half, channel] = values
            out[:, half:, channel] = values
        elif name == "depth_imbalance":
            bid_depth = bid_vols.sum(axis=1)
            ask_depth = ask_vols.sum(axis=1)
            depth_imbalance = (bid_depth - ask_depth) / (bid_depth + ask_depth + 1e-9)
            values = lag_feature(depth_imbalance)
            out[:, :half, channel] = values
            out[:, half:, channel] = values
        elif name == "microprice":
            best_ask = ask_prices_raw[:, 0, :]
            best_bid = bid_prices_raw[:, 0, :]
            best_ask_vol = ask_vols[:, 0, :]
            best_bid_vol = bid_vols[:, 0, :]
            microprice = (
                best_ask * best_bid_vol + best_bid * best_ask_vol
            ) / (best_ask_vol + best_bid_vol + 1e-9)
            values = lag_feature(
                (microprice - current_mid[:, None]) / current_mid[:, None]
            )
            out[:, :half, channel] = values
            out[:, half:, channel] = values
        elif name in _ORDER_FLOW_FEATURE_TO_COL:
            assert order_flow_windows is not None
            flow_col = _ORDER_FLOW_FEATURE_TO_COL[name]
            depth = ask_vols.sum(axis=1) + bid_vols.sum(axis=1)
            flow = order_flow_windows[:, flow_col, :] / (depth + 1e-9)
            values = lag_feature(np.clip(flow, -5.0, 5.0))
            out[:, :half, channel] = values
            out[:, half:, channel] = values
        elif name == "side":
            out[:, :half, channel] = -1.0
            out[:, half:, channel] = 1.0
        else:
            raise AssertionError(f"Unhandled extra node feature: {name}")
        channel += 1

    if not np.isfinite(out).all():
        raise ValueError("Non-finite features; check raw data and normalization.")
    return out


def _write_split_memmap(
    split_name: str,
    samples: np.ndarray,
    all_data: list[np.ndarray],
    all_labels: list[np.ndarray],
    all_order_flow: list[np.ndarray] | None,
    binner: VolumeBinner,
    out_x_path: Path,
    out_y_path: Path,
    n_lags: int,
    n_levels: int,
    normalize_prices: bool,
    extra_node_features: list[str],
    feature_dtype: np.dtype,
    chunk_size: int,
    verbose: bool,
) -> None:
    """Materialize one split to disk-backed .npy arrays."""
    n_samples = len(samples)
    n_nodes = 2 * n_levels * (n_lags + 1)
    n_features = 2 + len(extra_node_features)

    x_mm = open_memmap(
        out_x_path,
        mode="w+",
        dtype=feature_dtype,
        shape=(n_samples, n_nodes, n_features),
    )
    y_mm = open_memmap(out_y_path, mode="w+", dtype=np.int64, shape=(n_samples,))

    cursor = 0
    if n_samples > 0:
        _, first_idx = np.unique(samples[:, 0], return_index=True)
        unique_files = samples[np.sort(first_idx), 0]
    else:
        unique_files = np.array([], dtype=np.int32)
    for fi in unique_files:
        file_mask = samples[:, 0] == fi
        t_all = samples[file_mask, 1].astype(np.intp, copy=False)
        labels = all_labels[int(fi)][t_all]
        data = all_data[int(fi)]
        order_flow = None if all_order_flow is None else all_order_flow[int(fi)]

        y_mm[cursor:cursor + len(t_all)] = labels

        for s in range(0, len(t_all), chunk_size):
            e = min(s + chunk_size, len(t_all))
            t_chunk = t_all[s:e]
            features = _extract_chunk_features_with_extras(
                data=data,
                t_idx=t_chunk,
                n_lags=n_lags,
                binner=binner,
                normalize_prices=normalize_prices,
                extra_node_features=extra_node_features,
                order_flow=order_flow,
            )

            out = x_mm[cursor + s:cursor + e]
            if not np.isfinite(features).all():
                raise ValueError("Non-finite node features; inspect raw prices and message data.")
            out[:] = features
            if not np.isfinite(out).all():
                raise ValueError("Feature overflow on disk; use normalized prices or float32.")

        cursor += len(t_all)
        if verbose:
            print(f"  {split_name}: file {int(fi)} -> {len(t_all):,} samples")

    x_mm.flush()
    y_mm.flush()
    del x_mm, y_mm


def preprocess_to_disk(
    cfg: dict[str, Any],
    *,
    force: bool = False,
    chunk_size: int | None = None,
    verbose: bool = True,
) -> dict[str, Path]:
    """Build mmap-backed processed splits once and store them on disk.

    This function performs all expensive feature assembly offline:
      - lag-window extraction
      - micro/mid label generation
      - volume binning
      - node feature tensor assembly [3020, 2]

    The resulting .npy arrays are memory-mappable and can be consumed directly
    during training with near-zero CPU preprocessing overhead.
    """
    data_cfg = cfg["data"]
    paths = build_processed_paths(data_cfg["processed_dir"])
    paths["dir"].mkdir(parents=True, exist_ok=True)

    if not force and has_compatible_processed_dataset(cfg):
        if verbose:
            print("Using existing compatible processed dataset.")
        return paths

    raw_files = configured_orderbooks(data_cfg)
    if not raw_files:
        raise FileNotFoundError(f"No LOBSTER files found in '{data_cfg['raw_dir']}'")

    paths["meta"].unlink(missing_ok=True)

    if verbose:
        print("Loading raw LOBSTER files...")
    storage_mode = data_cfg.get("storage_mode", "materialized")
    cached_paths = []
    if storage_mode == "indexed" or data_cfg.get("raw_cache_dir"):
        cache = Path(data_cfg.get("raw_cache_dir", paths["dir"] / "raw_cache"))
        loaded = [cached_orderbook(f, cache) for f in raw_files]
        all_data = [pair[0] for pair in loaded]
        cached_paths = [pair[1] for pair in loaded]
    else:
        all_data = [load_lobster_csv(f) for f in raw_files]
    if verbose:
        for f, d in zip(raw_files, all_data):
            print(f"  {f.name}: {d.shape[0]:,} ticks")

    n_lags = int(data_cfg["n_lags"])
    n_levels = int(data_cfg["n_levels"])
    normalize_prices = bool(data_cfg.get("normalize_prices", True))
    feature_dtype = np.dtype(data_cfg.get("feature_dtype", "float16"))
    extra_node_features = normalize_extra_node_features(
        data_cfg.get("extra_node_features")
    )
    n_node_features = 2 + len(extra_node_features)

    all_order_flow: list[np.ndarray] | None = None
    message_files: list[Path] = []
    if uses_order_flow_features(extra_node_features):
        message_dir = data_cfg.get("message_dir", data_cfg["raw_dir"])
        message_pattern = str(data_cfg.get("message_pattern", "*_message_10.csv"))
        message_files = discover_message_files(message_dir, raw_files, message_pattern)
        if verbose:
            print("Loading raw LOBSTER message files for order-flow features...")
        all_order_flow = []
        for orderbook_path, message_path, orderbook_data in zip(
            raw_files,
            message_files,
            all_data,
        ):
            msg = load_message_csv(str(message_path))
            if len(msg) != len(orderbook_data):
                raise ValueError(
                    "LOBSTER message/orderbook row mismatch: "
                    f"{message_path.name} has {len(msg):,} rows, "
                    f"{orderbook_path.name} has {len(orderbook_data):,} rows."
                )
            all_order_flow.append(per_event_order_flow(msg))
            if verbose:
                print(f"  {message_path.name}: {msg.shape[0]:,} events")

    if n_levels != len(_ASK_P_COLS):
        raise ValueError(
            f"n_levels={n_levels} is not supported by this 40-column loader; "
            f"expected {len(_ASK_P_COLS)}."
        )
    k = int(data_cfg.get("prediction_horizon", 1))
    threshold = float(data_cfg["threshold"])
    price_type = str(data_cfg.get("price_type", "mid"))
    chunk = int(chunk_size or data_cfg.get("preprocess_chunk_size", 2048))

    if verbose:
        print("Computing labels...")
    label_mode = str(data_cfg.get("label_mode", "pct"))
    all_labels = [compute_labels(d, threshold, k, price_type, label_mode) for d in all_data]

    strategy = data_cfg["split_strategy"]
    if strategy == "by_file":
        train_s, val_s, test_s = split_by_file(all_data, cfg, n_lags, k)
    elif strategy == "by_lag":
        train_s, val_s, test_s = split_by_lag(all_data, cfg, n_lags, k)
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'")

    if verbose:
        print(
            f"Split ({strategy}): train={len(train_s):,} | "
            f"val={len(val_s):,} | test={len(test_s):,}"
        )

    if any(len(samples) == 0 for samples in (train_s, val_s, test_s)):
        raise ValueError("Every split must contain samples after temporal purging.")
    fit_data = []
    for file_index in sorted(set(train_s[:, 0])):
        times = train_s[train_s[:, 0] == file_index, 1]
        fit_data.append(all_data[int(file_index)][int(times.min()) - n_lags:int(times.max()) + 1])

    # fixed data seed: the val/test subsample must NOT move with training.seed,
    # otherwise multi-seed runs are evaluated on different test subsamples.
    seed = cfg.get("data", {}).get("subsample_seed", 42)
    max_per_class = data_cfg.get("max_samples_per_class")
    if max_per_class is not None:
        train_s = _balance_samples(train_s, all_labels, max_per_class, seed)
        if verbose:
            print(f"Balanced train: {len(train_s):,} samples ({len(train_s)//3:,}/class, cap={max_per_class:,})")
    max_val = data_cfg.get("max_val_samples")
    if max_val is not None:
        val_s = _subsample(val_s, max_val, seed)
        if verbose:
            print(f"Subsampled val:  {len(val_s):,} samples")
    max_test = data_cfg.get("max_test_samples")
    if max_test is not None:
        test_s = _subsample(test_s, max_test, seed)
        if verbose:
            print(f"Subsampled test: {len(test_s):,} samples")

    train_file_indices = sorted({int(x) for x in train_s[:, 0]}) if len(train_s) > 0 else []
    train_data_list = [all_data[i] for i in train_file_indices]
    if not train_data_list:
        raise ValueError("Training split is empty. Adjust split configuration before preprocessing.")
    binner = VolumeBinner(n_bins=int(data_cfg["n_volume_bins"])).fit(fit_data)
    binner.save(paths["binner"])

    # Kept for backward compatibility with existing evaluation/checkpoint flows.
    if normalize_prices and train_file_indices and storage_mode == "materialized":
        np.save(paths["price_stats"], build_price_stats(list(range(len(fit_data))), fit_data))

    if verbose:
        print("Writing processed splits (mmap .npy)...")
        print(
            "Node features: "
            f"{n_node_features} "
            f"([price, volume]"
            + (
                f" + {extra_node_features}"
                if extra_node_features
                else ""
            )
            + ")"
        )

    indexed_metadata = {}
    if storage_mode == "indexed":
        from remoma.dataset.indexed import write_indexed_splits
        indexed_metadata = write_indexed_splits(
            paths, {"train": train_s, "val": val_s, "test": test_s}, all_data,
            all_labels, all_order_flow, binner, cached_paths, chunk,
        )
    else:
        _write_materialized_splits(
            paths, train_s, val_s, test_s, all_data, all_labels, all_order_flow,
            binner, n_lags, n_levels, normalize_prices, extra_node_features, feature_dtype, chunk, verbose,
        )

    meta = {
        "signature": preprocess_signature(cfg),
        "storage_mode": storage_mode,
        "raw_files": [f.name for f in raw_files],
        "raw_manifest": file_manifest(raw_files),
        "message_manifest": file_manifest(message_files),
        "message_files": [f.name for f in message_files],
        "counts": {"train": int(len(train_s)), "val": int(len(val_s)), "test": int(len(test_s))},
        "n_nodes": int(2 * n_levels * (n_lags + 1)),
        "n_features": int(n_node_features),
        "extra_node_features": extra_node_features,
        "feature_dtype": str(feature_dtype), "chunk_size": chunk,
        **indexed_metadata,
    }
    if storage_mode == "indexed":
        meta["artifact_hashes"] = {paths[name].name: sha256_file(paths[name])
                                  for name in ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test", "binner")}
    from remoma.utils.artifacts import write_json
    write_json(paths["meta"], meta)
    if verbose:
        print(f"Processed dataset ready ({storage_mode}).")
    return paths


def _write_materialized_splits(paths, train_s, val_s, test_s, all_data, all_labels, all_order_flow,
                               binner, n_lags, n_levels, normalize_prices, extra_node_features,
                               feature_dtype, chunk, verbose):
    _write_split_memmap(
        split_name="train",
        samples=train_s,
        all_data=all_data,
        all_labels=all_labels,
        all_order_flow=all_order_flow,
        binner=binner,
        out_x_path=paths["X_train"],
        out_y_path=paths["y_train"],
        n_lags=n_lags,
        n_levels=n_levels,
        normalize_prices=normalize_prices,
        extra_node_features=extra_node_features,
        feature_dtype=feature_dtype,
        chunk_size=chunk,
        verbose=verbose,
    )
    _write_split_memmap(
        split_name="val",
        samples=val_s,
        all_data=all_data,
        all_labels=all_labels,
        all_order_flow=all_order_flow,
        binner=binner,
        out_x_path=paths["X_val"],
        out_y_path=paths["y_val"],
        n_lags=n_lags,
        n_levels=n_levels,
        normalize_prices=normalize_prices,
        extra_node_features=extra_node_features,
        feature_dtype=feature_dtype,
        chunk_size=chunk,
        verbose=verbose,
    )
    _write_split_memmap(
        split_name="test",
        samples=test_s,
        all_data=all_data,
        all_labels=all_labels,
        all_order_flow=all_order_flow,
        binner=binner,
        out_x_path=paths["X_test"],
        out_y_path=paths["y_test"],
        n_lags=n_lags,
        n_levels=n_levels,
        normalize_prices=normalize_prices,
        extra_node_features=extra_node_features,
        feature_dtype=feature_dtype,
        chunk_size=chunk,
        verbose=verbose,
    )
