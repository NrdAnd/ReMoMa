"""Daily full or relative-lag NMI without materializing a float lag tensor.

Relative mode stores M[delta, i, j] for delta >= 0. Negative lags follow
M[-delta, i, j] = M[delta, j, i]. Each representative uses exactly N-L rows,
and quantile bins are fitted separately on each shifted column, as in the
research notebook. Entropies and mutual information accumulate in float64.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import tempfile
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from remoma.graph.adjacency import sha256_file
from remoma.utils.artifacts import exclusive_lock, identity, read_json, write_json

SCHEMA = 1


@dataclass(frozen=True)
class NMISettings:
    method: str = "relative"
    max_lag: int = 100
    n_bins: int = 2000
    n_levels: int = 10
    backend: str = "auto"

    def validate(self):
        if self.method not in {"relative", "full"}:
            raise ValueError("NMI method must be relative or full.")
        if self.backend not in {"auto", "cpu", "cuda"}:
            raise ValueError("NMI backend must be auto, cpu, or cuda.")
        if (any(type(value) is not int for value in (self.max_lag, self.n_bins, self.n_levels))
                or self.max_lag < 0 or self.n_bins < 2 or not 1 <= self.n_levels <= 10):
            raise ValueError("Require max_lag >= 0, n_bins >= 2 and 1 <= n_levels <= 10.")
        if self.n_bins > 65535:
            raise ValueError("At most 65535 NMI bins are supported (uint16 discrete storage).")


def array_backend(requested: str):
    if requested != "cpu":
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() > 0:
                return cp, {"name": "cuda", "library": "cupy", "version": cp.__version__}
        except Exception as exc:
            if requested == "cuda":
                raise RuntimeError("CUDA NMI requires a working CuPy/CUDA installation.") from exc
        if requested == "cuda":
            raise RuntimeError("No CUDA device is available for NMI.")
    return np, {"name": "cpu", "library": "numpy", "version": np.__version__}


def _host(array, xp):
    return np.asarray(array) if xp is np else xp.asnumpy(array)


def discretize(values, n_bins: int, xp=np):
    """Exact linear quantiles; duplicate edges removed, constants kept constant."""
    values = xp.asarray(values, dtype=xp.float64)
    edges = xp.unique(xp.quantile(values, xp.linspace(0, 1, n_bins + 1, dtype=xp.float64)))
    if len(edges) == 1:
        return xp.zeros(len(values), dtype=xp.uint16)
    if len(edges) == 2:
        # Preserve genuinely binary columns; the old notebook collapsed these.
        return (values > edges[0]).astype(xp.uint16)
    return xp.searchsorted(edges[1:-1], values, side="right").astype(xp.uint16)


def _marginal(codes, n_bins, xp):
    counts = xp.bincount(codes.astype(xp.int64), minlength=n_bins)
    p = counts[counts > 0].astype(xp.float64) / len(codes)
    entropy = float(-xp.sum(p * xp.log(p)))
    return counts, entropy


def discrete_nmi(x, y, n_bins: int, xp=np, marginal_x=None, marginal_y=None) -> float:
    """Arithmetic-normalized empirical MI using integer joint counts.

    Dense counting is used when economical; otherwise only observed joint
    categories are counted. Both paths compute the same statistic.
    """
    if len(x) != len(y) or not len(x):
        raise ValueError("NMI requires nonempty, aligned columns.")
    cx, hx = marginal_x if marginal_x is not None else _marginal(x, n_bins, xp)
    cy, hy = marginal_y if marginal_y is not None else _marginal(y, n_bins, xp)
    if hx == 0 or hy == 0:
        return 1.0 if hx == hy else 0.0
    joint = x.astype(xp.int64) * n_bins + y
    if n_bins * n_bins <= max(4096, 4 * len(x)):
        counts = xp.bincount(joint, minlength=n_bins * n_bins)
        keys = xp.flatnonzero(counts)
        counts = counts[keys]
    else:
        keys, counts = xp.unique(joint, return_counts=True)
    counts = counts.astype(xp.float64)
    denom = cx[keys // n_bins].astype(xp.float64) * cy[keys % n_bins]
    mi = xp.sum((counts / len(x)) * xp.log(counts * len(x) / denom))
    value = float(mi) / ((hx + hy) / 2)
    if not np.isfinite(value) or value < -1e-10 or value > 1 + 1e-10:
        raise ValueError(f"Invalid NMI estimate: {value}.")
    return float(np.clip(value, 0, 1))


def read_volumes(path: Path, n_levels: int) -> np.ndarray:
    """Read only one day's volumes into memory, validating all 40 CSV columns."""
    columns = [4 * level + offset for offset in (1, 3) for level in range(n_levels)]
    parts = []
    for frame in pd.read_csv(path, header=None, dtype=np.float64, chunksize=65536):
        if frame.shape[1] != 40:
            raise ValueError(f"Expected 40 columns in {path}.")
        sizes = frame.iloc[:, columns].to_numpy(copy=True)
        if not np.isfinite(sizes).all() or (sizes < 0).any():
            raise ValueError(f"Volumes must be finite and nonnegative: {path}.")
        parts.append(sizes)
    if not parts:
        raise ValueError(f"Empty orderbook: {path}.")
    return np.concatenate(parts)


def compute_daily_nmi(volumes: np.ndarray, settings: NMISettings, *, work_dir=None,
                      progress: bool = True) -> np.ndarray:
    """Return compact relative values [L+1,B,B] or a full [B(L+1),B(L+1)] matrix."""
    settings.validate()
    xp, _ = array_backend(settings.backend)
    lag, bins = settings.max_lag, settings.n_bins
    volumes = np.asarray(volumes, dtype=np.float64)
    n, bases = volumes.shape
    if bases != settings.n_levels * 2 or n <= lag:
        raise ValueError("Volume dimensions must match n_levels and contain more rows than max_lag.")
    if not np.isfinite(volumes).all() or (volumes < 0).any():
        raise ValueError("Volumes must be finite and nonnegative.")
    rows = n - lag

    def column(base, shift):
        return discretize(volumes[lag - shift:n - shift, base], bins, xp)

    if settings.method == "relative":
        # Only B anchor columns and one shifted column reside on the GPU.
        anchors = [column(base, 0) for base in range(bases)]
        marginals = [_marginal(col, bins, xp) for col in anchors]
        result = np.empty((lag + 1, bases, bases), dtype=np.float64)
        result[0] = np.eye(bases)
        for i in range(bases):
            for j in range(i + 1, bases):
                value = discrete_nmi(anchors[i], anchors[j], bins, xp, marginals[i], marginals[j])
                result[0, i, j] = result[0, j, i] = value
        for shift in tqdm(range(1, lag + 1), desc="Relative NMI lags", disable=not progress):
            for i in range(bases):
                shifted = column(i, shift)
                marginal = _marginal(shifted, bins, xp)
                for j in range(bases):
                    result[shift, i, j] = discrete_nmi(
                        shifted, anchors[j], bins, xp, marginal, marginals[j])
        return result

    # Full mode stores discrete uint16 columns in a temporary disk array, never
    # a float64 lag tensor. GPU residence remains two columns plus histograms.
    features = bases * (lag + 1)
    with tempfile.TemporaryDirectory(prefix="nmi-codes-", dir=work_dir) as scratch:
        codes = np.lib.format.open_memmap(Path(scratch) / "codes.npy", mode="w+",
                                        dtype=np.uint16, shape=(features, rows))
        marginals = []
        for i in tqdm(range(features), desc="NMI quantiles", disable=not progress):
            col = column(i // (lag + 1), i % (lag + 1))
            codes[i] = _host(col, xp)
            counts, entropy = _marginal(col, bins, xp)
            marginals.append((_host(counts, xp), entropy))
        result = np.eye(features, dtype=np.float64)
        for i in tqdm(range(features), desc="Full NMI rows", disable=not progress):
            x = xp.asarray(codes[i])
            mx = (xp.asarray(marginals[i][0]), marginals[i][1])
            for j in range(i + 1, features):
                my = (xp.asarray(marginals[j][0]), marginals[j][1])
                result[i, j] = result[j, i] = discrete_nmi(x, xp.asarray(codes[j]), bins, xp, mx, my)
        del codes
    return result


def expand_relative(values: np.ndarray) -> np.ndarray:
    """Reconstruct side/level/lag order while preserving the signed lag direction."""
    length, bases, _ = values.shape
    delta = np.arange(length)[:, None] - np.arange(length)[None, :]
    result = np.empty((bases * length, bases * length), dtype=np.float64)
    for i in range(bases):
        for j in range(bases):
            result[i * length:(i + 1) * length, j * length:(j + 1) * length] = np.where(
                delta >= 0, values[np.abs(delta), i, j], values[np.abs(delta), j, i])
    np.fill_diagonal(result, 1.0)
    return result


def daily_artifact(path: Path, settings: NMISettings, cache: Path, *, progress=True) -> tuple[np.ndarray, dict]:
    """Validate the input content on every invocation; reuse only matching artifacts."""
    settings.validate()
    xp, backend = array_backend(settings.backend)
    source_hash = sha256_file(path)
    effective_settings = asdict(settings) | {"backend": backend["name"]}
    key = {"schema": SCHEMA, "source_sha256": source_hash, "settings": effective_settings,
           "backend": backend, "implementation_sha256": sha256_file(Path(__file__))}
    digest = identity(key)
    root = cache / "nmi" / digest
    manifest_path = root / "manifest.json"
    array_path = root / "values.npy"

    def load_existing():
        manifest = read_json(manifest_path)
        if manifest["key"] != key or sha256_file(array_path) != manifest["values_sha256"]:
            raise ValueError(f"NMI cache integrity failure: {root}.")
        return np.load(array_path, mmap_mode="r", allow_pickle=False), manifest

    if manifest_path.exists():
        print(f"Reusing {settings.method} NMI: {path.name}", flush=True)
        return load_existing()
    with exclusive_lock(cache / "locks" / (digest + ".nmi.lock")):
        if manifest_path.exists():
            return load_existing()
        root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        volumes = read_volumes(path, settings.n_levels)
        try:
            values = compute_daily_nmi(volumes, settings, work_dir=root, progress=progress)
        finally:
            # The model trainer runs in a separate process. Return CuPy's cached
            # allocations before that process needs the GPU for model batches.
            if xp is not np:
                xp.get_default_memory_pool().free_all_blocks()
                xp.get_default_pinned_memory_pool().free_all_blocks()
        if sha256_file(path) != source_hash:
            raise ValueError(f"Input changed while computing NMI: {path}.")
        temporary = root / "values.partial.npy"
        np.save(temporary, values, allow_pickle=False)
        temporary.replace(array_path)
        manifest = {"key": key, "id": digest, "source_name": path.name, "rows": len(volumes),
                    "sample_rows": len(volumes) - settings.max_lag, "shape": list(values.shape),
                    "values_sha256": sha256_file(array_path), "seconds": time.perf_counter() - started}
        write_json(manifest_path, manifest)
        return values, manifest
