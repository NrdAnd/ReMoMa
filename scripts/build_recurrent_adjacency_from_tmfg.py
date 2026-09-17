#!/usr/bin/env python3
"""Convert the project TMFG adjacency to RecurrentSparseSTHNN format.

This uses the already-built TMFG support used by the original GNN pipeline and
exports the lag-0..100 induced subgraph with labels/order expected by the
RecurrentSparseSTHNN implementation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/graphs/tmfg_lag150_bins2000.csv",
        help="Existing TMFG adjacency CSV, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output",
        default="data/graphs/recurrent_sparse_tmfg_lag100_bins2000_from_full_tmfg_adjacency.tsv",
        help="Output TSV path, relative to the repository root by default.",
    )
    parser.add_argument("--max-lag", type=int, default=100)
    parser.add_argument("--n-levels", type=int, default=10)
    return parser.parse_args()


def project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def source_label(side: str, level: int, lag: int) -> str:
    source_side = "ask" if side == "ASKs" else "bid"
    return f"{source_side}_{level - 1}_lag_{lag}"


def recurrent_label(side: str, level: int, lag: int) -> str:
    return f"{side}{level}_lag{lag}"


def recurrent_order(max_lag: int, n_levels: int) -> list[tuple[str, int, int]]:
    return [
        (side, level, lag)
        for lag in range(max_lag, -1, -1)
        for level in range(1, n_levels + 1)
        for side in ("ASKs", "BIDs")
    ]


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_path = project_path(args.output)

    print(f"Loading TMFG adjacency: {input_path}")
    adjacency = pd.read_csv(input_path, index_col=0)
    labels = [str(label) for label in adjacency.index]
    if labels != [str(label) for label in adjacency.columns]:
        raise ValueError("Input adjacency matrix must have matching index/columns.")

    order = recurrent_order(args.max_lag, args.n_levels)
    source_labels = [source_label(side, level, lag) for side, level, lag in order]
    target_labels = [recurrent_label(side, level, lag) for side, level, lag in order]

    missing = [label for label in source_labels if label not in adjacency.index]
    if missing:
        raise ValueError(f"Missing required TMFG labels: {missing[:5]}")

    recurrent_adj = adjacency.loc[source_labels, source_labels].copy()
    recurrent_adj.index = target_labels
    recurrent_adj.columns = target_labels
    recurrent_adj = (recurrent_adj != 0).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    recurrent_adj.to_csv(output_path, sep="\t")

    directed_edges = int(recurrent_adj.to_numpy().sum())
    print(f"Saved: {output_path}")
    print(f"Shape: {recurrent_adj.shape} | directed nonzero entries: {directed_edges:,}")


if __name__ == "__main__":
    main()
