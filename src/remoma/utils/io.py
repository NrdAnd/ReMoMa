from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path


def load_lobster_csv(path: str | Path) -> np.ndarray:
    """Load a LOBSTER orderbook CSV (no header, 40 columns).

    Column layout per level i (0-indexed, 10 levels):
        4*i   → AskPrice_L(i+1)
        4*i+1 → AskVol_L(i+1)
        4*i+2 → BidPrice_L(i+1)
        4*i+3 → BidVol_L(i+1)

    Returns:
        np.ndarray: shape [N, 40], dtype float32.
    """
    values = pd.read_csv(path, header=None, dtype=np.float32, on_bad_lines="error").values
    if values.ndim != 2 or values.shape[1] != 40:
        raise ValueError(f"Expected exactly 40 LOBSTER orderbook columns: {path}.")
    if not np.isfinite(values[:, 0::2]).all() or (values[:, 0::2] <= 0).any():
        raise ValueError(f"Orderbook prices must be finite and positive: {path}.")
    if not np.isfinite(values[:, 1::2]).all() or (values[:, 1::2] < 0).any():
        raise ValueError(f"Orderbook volumes must be finite and nonnegative: {path}.")
    return values


def discover_files(raw_dir: str | Path, pattern: str = "*_orderbook_10.csv") -> list[Path]:
    """Return sorted list of LOBSTER CSV files matching pattern."""
    return sorted(Path(raw_dir).glob(pattern))


def configured_orderbooks(data_cfg: dict) -> list[Path]:
    """Honor an explicit frozen file list instead of discovering additional days."""
    if data_cfg.get("raw_files") is None:
        return discover_files(data_cfg["raw_dir"])
    paths = [Path(p) if Path(p).is_absolute() else Path(data_cfg["raw_dir"]) / p
             for p in data_cfg["raw_files"]]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("data.raw_files must be nonempty and unique.")
    if [p.name for p in paths] != sorted(p.name for p in paths):
        raise ValueError("data.raw_files must be in chronological filename order.")
    if any(not p.is_file() for p in paths):
        raise FileNotFoundError("A file listed in data.raw_files is missing.")
    return paths


def cached_orderbook(path: Path, cache: Path) -> tuple[np.ndarray, Path]:
    """Parse a CSV once; share read-only float32 memmaps across folds and workers."""
    from remoma.graph.adjacency import sha256_file
    from remoma.utils.artifacts import exclusive_lock, read_json, write_json
    digest = sha256_file(path)
    root = Path(cache) / "raw_v1" / digest
    target, metadata = root / "orderbook.npy", root / "manifest.json"
    if not metadata.exists():
        with exclusive_lock(Path(cache) / "locks" / (digest + ".raw.lock")):
            if not metadata.exists():
                root.mkdir(parents=True, exist_ok=True)
                values = load_lobster_csv(path)
                if sha256_file(path) != digest:
                    raise ValueError(f"Input changed during parsing: {path}.")
                temporary = root / "orderbook.partial.npy"
                np.save(temporary, values, allow_pickle=False)
                temporary.replace(target)
                write_json(metadata, {"source_sha256": digest, "array_sha256": sha256_file(target)})
    record = read_json(metadata)
    if record["source_sha256"] != digest or sha256_file(target) != record["array_sha256"]:
        raise ValueError(f"Raw array cache integrity failure: {target}.")
    return np.load(target, mmap_mode="r", allow_pickle=False), target


def discover_message_files(
    raw_dir: str | Path,
    orderbook_files: list[Path],
    pattern: str = "*_message_10.csv",
) -> list[Path]:
    """Return message files aligned to sorted LOBSTER orderbook files.

    LOBSTER orderbook and message files share the same prefix, differing only
    by ``orderbook`` vs ``message`` in the filename. We match by that expected
    name exactly. Equal file counts do not establish row or day alignment.
    """
    root = Path(raw_dir)
    expected = [
        root / path.name.replace("_orderbook_10.csv", "_message_10.csv")
        for path in orderbook_files
    ]
    if all(path.exists() for path in expected):
        return expected

    found = sorted(root.glob(pattern))

    missing = [path.name for path in expected if not path.exists()]
    raise FileNotFoundError(
        "Could not align LOBSTER message files with orderbook files. "
        f"Missing expected files: {missing[:5]}; found {len(found)} files "
        f"with pattern {pattern!r} in {root}."
    )
