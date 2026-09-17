#!/usr/bin/env python3
"""Pinpoint where the volume feature becomes all-zeros.

Inspects three stages of the volume pipeline:
  1. The fitted VolumeBinner edges (binner.pkl).
  2. Raw volume columns from one LOBSTER file.
  3. The stored X_train.npy volume channel.

Usage:
    python scripts/inspect_volume.py --config configs/gnn/default.yaml
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config

from remoma.dataset.preprocessing import build_processed_paths
from remoma.utils.io import discover_files

_ASK_V_COLS = np.arange(1, 40, 4)
_BID_V_COLS = np.arange(3, 40, 4)


def main(config_path: str) -> None:
    import yaml
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    paths = build_processed_paths(data_cfg["processed_dir"])

    print("=" * 60)
    print("STAGE 1 — fitted binner edges (binner.pkl)")
    print("=" * 60)
    with open(paths["binner"], "rb") as f:
        state = pickle.load(f)
    edges = state["edges"]
    print(f"  n_bins (config)   : {state['n_bins']}")
    print(f"  len(edges)        : {len(edges)}")
    print(f"  edges[:5]         : {np.asarray(edges[:5])}")
    print(f"  edges[-5:]        : {np.asarray(edges[-5:])}")
    print(f"  edges[1:-1] empty?: {len(edges) <= 2}  "
          f"(if True -> binner outputs all zeros)")

    print("\n" + "=" * 60)
    print("STAGE 2 — raw volume columns (first LOBSTER file)")
    print("=" * 60)
    files = discover_files(data_cfg["raw_dir"])
    raw = pd.read_csv(files[0], header=None, on_bad_lines="skip").values
    vol_cols = np.concatenate([_ASK_V_COLS, _BID_V_COLS])
    vols = raw[:, vol_cols].astype(np.float64)
    print(f"  raw shape         : {raw.shape}")
    print(f"  volume min/max    : {vols.min():.1f} / {vols.max():.1f}")
    print(f"  volume mean/std   : {vols.mean():.1f} / {vols.std():.1f}")
    print(f"  unique vol values : {len(np.unique(vols)):,}")
    print(f"  pct of zeros      : {100.0 * (vols == 0).mean():.2f}%")

    print("\n" + "=" * 60)
    print("STAGE 3 — stored X_train.npy volume channel")
    print("=" * 60)
    X = np.load(paths["X_train"], mmap_mode="r")
    sample = np.asarray(X[:200], dtype=np.float32)  # [200, N, 2]
    price_ch = sample[..., 0]
    vol_ch = sample[..., 1]
    print(f"  X_train shape     : {X.shape}  dtype={X.dtype}")
    print(f"  price ch min/max  : {price_ch.min():+.3e} / {price_ch.max():+.3e}")
    print(f"  volume ch min/max : {vol_ch.min():+.3e} / {vol_ch.max():+.3e}")
    print(f"  volume ch std     : {vol_ch.std():.3e}")
    print(f"  volume unique vals: {len(np.unique(vol_ch))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn/default.yaml")
    args = parser.parse_args()
    main(args.config)
