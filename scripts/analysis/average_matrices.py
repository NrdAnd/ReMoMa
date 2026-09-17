#!/usr/bin/env python3
"""Average identically labeled similarity matrices without implicit alignment."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def average_matrices(files: list[str | Path], output: str | Path) -> pd.DataFrame:
    if not files:
        raise ValueError("Provide at least one input matrix.")
    total = None
    for path in files:
        frame = pd.read_csv(path, index_col=0)
        if frame.shape[0] != frame.shape[1] or list(frame.index) != list(frame.columns):
            raise ValueError(f"Matrix must be square with matching labeled axes: {path}.")
        if not frame.index.is_unique or not np.isfinite(frame.to_numpy()).all():
            raise ValueError(f"Matrix labels must be unique and values finite: {path}.")
        if total is None:
            total = frame.copy()
        else:
            if not total.index.equals(frame.index) or not total.columns.equals(frame.columns):
                raise ValueError(f"Matrix labels/order differ: {path}.")
            total += frame
    result = total / len(files)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    average_matrices(args.inputs, args.output)
    print(f"Saved mean similarity matrix to {args.output}")
