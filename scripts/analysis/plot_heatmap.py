#!/usr/bin/env python3
"""Save a heatmap from a labeled similarity matrix."""
import argparse
from pathlib import Path


def plot_heatmap(frame, title, output, cmap="YlGnBu", vmin=0.0, vmax=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axis = plt.subplots(figsize=(12, 10))
    image = axis.imshow(frame.to_numpy(dtype=np.float32), cmap=cmap, vmin=vmin,
                        vmax=vmax, interpolation="nearest", aspect="equal")
    figure.colorbar(image, ax=axis, label="Normalized mutual information")
    ticks = np.unique(np.linspace(0, len(frame) - 1, min(20, len(frame)), dtype=int))
    axis.set_xticks(ticks, [str(frame.columns[i]) for i in ticks])
    axis.set_yticks(ticks, [str(frame.index[i]) for i in ticks])
    axis.set_title(title)
    axis.tick_params(axis="x", rotation=90, labelsize=8)
    axis.tick_params(axis="y", labelsize=8)
    figure.tight_layout()
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Mean NMI — Limit Order Book")
    args = parser.parse_args()
    import pandas as pd
    plot_heatmap(pd.read_csv(args.input, index_col=0), args.title, args.output)
    print(f"Saved heatmap to {args.output}")
