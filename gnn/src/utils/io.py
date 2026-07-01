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
    return pd.read_csv(path, header=None, dtype=np.float32, on_bad_lines="skip").values


def discover_files(raw_dir: str | Path, pattern: str = "*_orderbook_10.csv") -> list[Path]:
    """Return sorted list of LOBSTER CSV files matching pattern."""
    return sorted(Path(raw_dir).glob(pattern))


def discover_message_files(
    raw_dir: str | Path,
    orderbook_files: list[Path],
    pattern: str = "*_message_10.csv",
) -> list[Path]:
    """Return message files aligned to sorted LOBSTER orderbook files.

    LOBSTER orderbook and message files share the same prefix, differing only
    by ``orderbook`` vs ``message`` in the filename. We match by that expected
    name first and fall back to a sorted-pattern count check for custom names.
    """
    root = Path(raw_dir)
    expected = [
        root / path.name.replace("_orderbook_10.csv", "_message_10.csv")
        for path in orderbook_files
    ]
    if all(path.exists() for path in expected):
        return expected

    found = sorted(root.glob(pattern))
    if len(found) == len(orderbook_files):
        return found

    missing = [path.name for path in expected if not path.exists()]
    raise FileNotFoundError(
        "Could not align LOBSTER message files with orderbook files. "
        f"Missing expected files: {missing[:5]}; found {len(found)} files "
        f"with pattern {pattern!r} in {root}."
    )
