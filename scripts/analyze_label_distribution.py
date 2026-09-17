#!/usr/bin/env python3
"""Analyze class balance across label-generation hyperparameters.

Example:
    python scripts/analyze_label_distribution.py --config configs/gnn/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config

from remoma.dataset.labeling import compute_labels
from remoma.utils.io import discover_files, load_lobster_csv


def _parse_int_list(value: str) -> list[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_list(value: str) -> list[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _format_pct(v: float) -> str:
    return f"{v:6.2f}%"


def main(config_path: str, ks: list[int], thresholds: list[float], price_types: list[str]) -> None:
    cfg = load_config(config_path)

    raw_dir = cfg["data"]["raw_dir"]
    files = discover_files(raw_dir)
    if not files:
        raise FileNotFoundError(f"No LOBSTER files found in '{raw_dir}'")

    print(f"Loading {len(files)} files from {raw_dir}...")
    all_data = [load_lobster_csv(f) for f in files]
    for f, d in zip(files, all_data):
        print(f"  {f.name}: {d.shape[0]:,} ticks")

    print("\nLabel distribution grid:")
    print("  class mapping: 0=DOWN, 1=FLAT, 2=UP\n")

    for price_type in price_types:
        for k in ks:
            for threshold in thresholds:
                counts = np.zeros(3, dtype=np.int64)
                for data in all_data:
                    labels = compute_labels(data, threshold=threshold, k=k, price_type=price_type)
                    counts += np.bincount(labels, minlength=3)

                total = int(counts.sum())
                pct = (counts / max(total, 1)) * 100.0
                print(
                    f"k={k:<3} threshold={threshold:g} price_type={price_type:<5} "
                    f"DOWN: {_format_pct(pct[0])}  FLAT: {_format_pct(pct[1])}  UP: {_format_pct(pct[2])}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn/default.yaml")
    parser.add_argument("--ks", default="1,5,10,20,50,100,200")
    parser.add_argument("--thresholds", default="0,1e-5,5e-5,1e-4")
    parser.add_argument("--price-types", default="mid,micro")
    args = parser.parse_args()

    main(
        config_path=args.config,
        ks=_parse_int_list(args.ks),
        thresholds=_parse_float_list(args.thresholds),
        price_types=_parse_str_list(args.price_types),
    )
