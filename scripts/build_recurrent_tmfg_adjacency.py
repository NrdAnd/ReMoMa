#!/usr/bin/env python3
"""Build the RecurrentSparseSTHNN adjacency from this project's TMFG pipeline.

The colleague's `example.tsv` is only a format reference. This script builds a
project-owned adjacency by:
  1. loading the LOB NMI similarity matrix,
  2. keeping consecutive lags 0..max_lag,
  3. running TMFG on that filtered similarity matrix,
  4. exporting a labeled TSV in the order expected by RecurrentSparseSTHNN.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from remoma.graph.tmfg import TMFG, OutputMode  # noqa: E402


_NMI_LABEL_RE = re.compile(r"^(ask|bid)_(\d+)_lag_(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/similarity/lob_similarity_nmi_rellag_lag150_bins2000_mean.csv",
        help="NMI/similarity CSV path, relative to the repository root by default.",
    )
    parser.add_argument(
        "--output",
        default="data/graphs/recurrent_sparse_tmfg_lag100_bins2000_from_nmi_mean.tsv",
        help="Output TSV path, relative to the repository root by default.",
    )
    parser.add_argument("--max-lag", type=int, default=100)
    parser.add_argument("--n-levels", type=int, default=10)
    parser.add_argument(
        "--weighted",
        action="store_true",
        help="Keep selected NMI/TMFG edge weights instead of binarizing to 0/1.",
    )
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


def validate_source_labels(similarity: pd.DataFrame) -> None:
    labels = [str(label) for label in similarity.index]
    if labels != [str(label) for label in similarity.columns]:
        raise ValueError("Input similarity matrix must have matching index/columns.")
    bad = [label for label in labels if _NMI_LABEL_RE.fullmatch(label) is None]
    if bad:
        raise ValueError(f"Unexpected source labels, e.g. {bad[:5]}")


def main() -> None:
    args = parse_args()
    input_path = project_path(args.input)
    output_path = project_path(args.output)
    if not input_path.exists():
        available = sorted((ROOT / "data" / "similarity").glob("*.csv"))
        available_list = "\n".join(f"  - {path}" for path in available) or "  <none>"
        raise FileNotFoundError(
            f"Input similarity matrix not found: {input_path}\n"
            "The recurrent TMFG adjacency should be rebuilt from the same NMI "
            "source used by the project TMFG experiment. Available local CSVs:\n"
            f"{available_list}\n"
            "Use --input only if you intentionally want a different source."
        )

    print(f"Loading similarity matrix: {input_path}")
    similarity = pd.read_csv(input_path, index_col=0)
    validate_source_labels(similarity)

    order = recurrent_order(args.max_lag, args.n_levels)
    source_labels = [source_label(side, level, lag) for side, level, lag in order]
    target_labels = [recurrent_label(side, level, lag) for side, level, lag in order]

    missing = [label for label in source_labels if label not in similarity.index]
    if missing:
        raise ValueError(f"Missing required lag labels in similarity matrix: {missing[:5]}")

    filtered = similarity.loc[source_labels, source_labels].copy()
    filtered.index = target_labels
    filtered.columns = target_labels

    print(
        "Running TMFG: "
        f"{filtered.shape[0]} nodes | lags=0..{args.max_lag} | levels={args.n_levels}"
    )
    tmfg = TMFG()
    output_mode = (
        OutputMode.WEIGHTED_SPARSE_W_MATRIX
        if args.weighted
        else OutputMode.UNWEIGHTED_SPARSE_W_MATRIX
    )
    _, _, adjacency = tmfg.fit_transform(
        filtered,
        output=output_mode.value,
    )

    adj_df = pd.DataFrame(adjacency, index=target_labels, columns=target_labels)
    if not args.weighted:
        adj_df = (adj_df != 0).astype(int)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    adj_df.to_csv(output_path, sep="\t")

    directed_edges = int(adj_df.to_numpy().sum())
    print(f"Saved: {output_path}")
    if args.weighted:
        nonzero = int((adj_df.to_numpy() != 0).sum())
        print(
            f"Shape: {adj_df.shape} | directed nonzero entries: {nonzero:,} | "
            "weighted=true"
        )
    else:
        print(f"Shape: {adj_df.shape} | directed nonzero entries: {directed_edges:,}")


if __name__ == "__main__":
    main()
