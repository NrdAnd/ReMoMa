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
