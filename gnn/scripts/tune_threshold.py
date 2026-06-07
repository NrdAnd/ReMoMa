#!/usr/bin/env python3
"""Post-hoc decision-threshold tuning for imbalanced F1 macro.

A model trained on balanced classes over-predicts the rare classes when the
eval set is imbalanced (here ~90% flat). This script keeps the trained model
fixed and only changes the DECISION RULE: predict down/up only when the model
is confident enough (prob >= threshold), else flat. The thresholds are tuned on
the VALIDATION set to maximize F1 macro, then applied to the TEST set.

Usage:
    python scripts/tune_threshold.py --config config/default.yaml
    python scripts/tune_threshold.py --checkpoint checkpoints/best.pt
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from src.dataset.preprocessing import build_processed_paths
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model
from src.training.threshold import apply_thresholds, tune_thresholds


@torch.no_grad()
def get_probs(model, loader, edge_index, device, static_mode):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        if static_mode:
            x, y = batch
            out = model.forward_static(x.to(device), edge_index)
        else:
            batch = batch.to(device)
            out = model(batch)
            y = batch.y
        probs.append(torch.softmax(out, dim=1).cpu().numpy())
        labels.append(np.asarray(y.cpu()))
    return np.concatenate(probs), np.concatenate(labels)


def main(config_path: str, checkpoint: str, model: str | None = None) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if model is not None:
        cfg["model"]["type"] = model          # must match the trained checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_cfg = cfg["data"]
    paths = build_processed_paths(data_cfg["processed_dir"])

    edge_index = load_tmfg_edge_index(
        data_cfg["adj_matrix_path"],
        cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
    )

    model_type = cfg["model"]["type"].lower()
    static_mode = bool(cfg["training"].get("static_graph_batching", False)) and model_type in ("gcn", "cgnn", "stgcn")
    static_edge = edge_index.to(device) if static_mode else None

    bs = int(cfg["training"]["batch_size"])
    nw = cfg["training"].get("num_workers", 0)

    def make_loader(xp, yp):
        if static_mode:
            ds = StaticLOBTensorDataset(str(xp), str(yp))
            return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw)
        from scripts.train import make_collate_fn  # only needed for PyG models
        ds = FastLOBDataset(str(xp), str(yp), edge_index)
        n_nodes = 2 * int(data_cfg["n_levels"]) * (int(data_cfg["n_lags"]) + 1)
        cf = make_collate_fn(edge_index, bs, n_nodes)
        return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=cf)

    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    print(f"Loaded {checkpoint} | model={model_type} | static={static_mode}")

    print("Computing probabilities on val and test...")
    val_p, val_y = get_probs(model, make_loader(paths["X_val"], paths["y_val"]), static_edge, device, static_mode)
    test_p, test_y = get_probs(model, make_loader(paths["X_test"], paths["y_test"]), static_edge, device, static_mode)

    # ── Baseline: plain argmax ──
    base_f1 = f1_score(test_y, test_p.argmax(1), average="macro", zero_division=0)

    # ── Tune (t_down, t_up) on VAL, apply to TEST (shared logic) ──
    td, tu, val_tuned = tune_thresholds(val_p, val_y)
    tuned_test = apply_thresholds(test_p, td, tu)
    tuned_f1 = f1_score(test_y, tuned_test, average="macro", zero_division=0)
    val_base = f1_score(val_y, val_p.argmax(1), average="macro", zero_division=0)

    print("\n" + "=" * 56)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 56)
    print(f"  best thresholds (tuned on val): down>={td:.2f}  up>={tu:.2f}")
    print(f"  VAL  F1 macro : {val_base:.4f}  →  {val_tuned:.4f}")
    print(f"  TEST F1 macro : {base_f1:.4f}  →  {tuned_f1:.4f}   (Δ {tuned_f1 - base_f1:+.4f})")
    print("\n── TEST report (tuned) ─────────────────────────────")
    print(classification_report(test_y, tuned_test, target_names=["down", "flat", "up"], digits=4, zero_division=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "cgnn", "stgcn"], default=None,
                        help="model type of the checkpoint (es. cgnn)")
    args = parser.parse_args()
    main(args.config, args.checkpoint, args.model)
