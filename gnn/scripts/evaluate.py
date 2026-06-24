#!/usr/bin/env python3
"""Evaluate a saved checkpoint on the processed test split.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --config config/default.yaml
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from src.dataset.preprocessing import (
    build_processed_paths,
    has_compatible_processed_dataset,
    preprocess_to_disk,
)
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model, is_recurrent_sparse_model
from src.training.metrics import compute_metrics, print_report
from src.training.threshold import apply_thresholds, decision_metrics
from src.training.trainer import Trainer


def _load_thresholds(path: str | None, t_down: float | None, t_up: float | None) -> tuple[float, float] | None:
    if path is None and t_down is None and t_up is None:
        return None
    if path is not None:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        thresholds = payload["thresholds"]
        return float(thresholds["down"]), float(thresholds["up"])
    if t_down is None or t_up is None:
        raise ValueError("Pass both --t-down and --t-up, or pass --thresholds.")
    return float(t_down), float(t_up)


def _print_signal_summary(metrics: dict[str, float]) -> None:
    print(
        "Signals:     "
        f"precision={metrics['signal_precision']:.4f}  "
        f"recall={metrics['signal_recall']:.4f}  "
        f"f1={metrics['signal_f1']:.4f}  "
        f"rate={metrics['signal_rate']:.4f}  "
        f"flat→signal={metrics['flat_to_signal_rate']:.4f}"
    )


def main(
    config_path: str,
    checkpoint_path: str,
    thresholds_path: str | None = None,
    t_down: float | None = None,
    t_up: float | None = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]

    if not has_compatible_processed_dataset(cfg):
        print("Processed tensors not found or incompatible. Building them now...")
        preprocess_to_disk(cfg, force=False, verbose=True)

    paths = build_processed_paths(data_cfg["processed_dir"])
    model_type = cfg["model"]["type"].lower()
    recurrent_sparse = is_recurrent_sparse_model(model_type)
    if recurrent_sparse:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        test_ds = StaticLOBTensorDataset(str(paths["X_test"]), str(paths["y_test"]))
        test_loader = TorchDataLoader(
            test_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"].get("num_workers", 0),
        )
    else:
        edge_index = load_tmfg_edge_index(
            data_cfg["adj_matrix_path"],
            cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
        )
        test_ds = FastLOBDataset(str(paths["X_test"]), str(paths["y_test"]), edge_index)
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
        optimizer=None,
        scheduler=None,
        criterion=None,
        device=device,
        checkpoint_dir=Path(checkpoint_path).parent,
        static_graph_batching=recurrent_sparse,
        static_edge_index=edge_index.to(device) if recurrent_sparse else None,
    )
    thresholds = _load_thresholds(thresholds_path, t_down, t_up)
    if thresholds is None:
        preds, labels = trainer.predict(test_loader)
    else:
        td, tu = thresholds
        probs, labels = trainer.predict_proba(test_loader)
        preds = apply_thresholds(probs, td, tu)
        print(f"Decision thresholds: down>={td:.2f} up>={tu:.2f}\n")

    metrics = compute_metrics(preds, labels)
    signal_metrics = decision_metrics(preds, labels)

    print(f"Accuracy:    {metrics['accuracy']:.4f}")
    print(f"F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"F1 Weighted: {metrics['f1_weighted']:.4f}")
    _print_signal_summary(signal_metrics)
    print()
    print_report(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--thresholds", default=None, help="JSON produced by scripts/tune_threshold.py")
    parser.add_argument("--t-down", type=float, default=None)
    parser.add_argument("--t-up", type=float, default=None)
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.thresholds, args.t_down, args.t_up)
