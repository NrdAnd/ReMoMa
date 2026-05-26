#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the test set.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.binning import VolumeBinner
from src.dataset.labeling import compute_labels
from src.dataset.lob_dataset import LOBDataset
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model
from src.training.metrics import compute_metrics, print_report
from src.training.trainer import Trainer
from src.utils.io import discover_files, load_lobster_csv


def main(config_path: str, checkpoint_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_lags = cfg["data"]["n_lags"]
    k      = cfg["data"].get("prediction_horizon", 1)
    ckpt_dir = Path(checkpoint_path).parent

    print("Loading LOB files...")
    files = discover_files(cfg["data"]["raw_dir"])
    all_data = [load_lobster_csv(f) for f in files]
    all_labels = [compute_labels(d, cfg["data"]["threshold"], k, cfg["data"].get("price_type", "mid")) for d in all_data]

    test_file_indices = cfg["data"]["test_files"]
    test_s = np.concatenate([
        np.stack([
            np.full(len(all_data[fi]) - n_lags - k, fi, dtype=np.int32),
            np.arange(n_lags, len(all_data[fi]) - k, dtype=np.int32),
        ], axis=1)
        for fi in test_file_indices
    ])

    binner = VolumeBinner.load(ckpt_dir / "binner.pkl")

    price_stats = None
    ps_path = ckpt_dir / "price_stats.npy"
    if ps_path.exists():
        price_stats = np.load(str(ps_path), allow_pickle=True).item()

    edge_index = load_tmfg_edge_index(
        cfg["data"]["adj_matrix_path"],
        cache_path=Path(cfg["data"]["processed_dir"]) / "edge_index.pt",
    )

    test_ds = LOBDataset(
        test_s, all_data, all_labels, edge_index, binner, price_stats, n_lags
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"].get("num_workers", 0),
    )

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))

    trainer = Trainer(
        model=model,
        optimizer=None, scheduler=None, criterion=None,
        device=device,
        checkpoint_dir=ckpt_dir,
    )
    preds, labels = trainer.predict(test_loader)
    metrics = compute_metrics(preds, labels)

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    print()
    print_report(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
