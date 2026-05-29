#!/usr/bin/env python3
"""Offline preprocessing for the LOB-GNN pipeline.

Builds disk-backed split tensors once:
  - X_train.npy, y_train.npy
  - X_val.npy,   y_val.npy
  - X_test.npy,  y_test.npy
  - binner.pkl, meta.json, edge_index.pt
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.preprocessing import preprocess_to_disk
from src.graph.adjacency import load_tmfg_edge_index


def main(config_path: str, force: bool, chunk_size: int | None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    paths = preprocess_to_disk(cfg, force=force, chunk_size=chunk_size, verbose=True)
    data_cfg = cfg["data"]
    edge_index = load_tmfg_edge_index(
        data_cfg["adj_matrix_path"],
        cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
    )

    print("\nPreprocessing complete.")
    print(f"Processed dir: {paths['dir']}")
    print(f"Cached edge_index: {Path(data_cfg['processed_dir']) / 'edge_index.pt'}")
    print(f"Edges: {edge_index.shape[1]:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild even if compatible artifacts exist.")
    parser.add_argument("--chunk-size", type=int, default=None, help="Override preprocess chunk size.")
    args = parser.parse_args()
    main(args.config, args.force, args.chunk_size)
