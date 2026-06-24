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
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from src.dataset.preprocessing import (
    build_processed_paths,
    has_compatible_processed_dataset,
    preprocess_to_disk,
)
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model, is_recurrent_sparse_model
from src.training.threshold import apply_thresholds, decision_metrics, search_thresholds


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


def _print_decision_summary(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<12} "
        f"acc={metrics['accuracy']:.4f}  "
        f"macro_f1={metrics['f1_macro']:.4f}  "
        f"weighted_f1={metrics['f1_weighted']:.4f}  "
        f"signal_prec={metrics['signal_precision']:.4f}  "
        f"signal_rec={metrics['signal_recall']:.4f}  "
        f"signal_rate={metrics['signal_rate']:.4f}  "
        f"flat→signal={metrics['flat_to_signal_rate']:.4f}"
    )


def main(
    config_path: str,
    checkpoint: str,
    model: str | None = None,
    lo: float = 0.33,
    hi: float = 0.95,
    step: float = 0.01,
    objective: str = "macro_f1",
    flat_fp_penalty: float = 0.0,
    min_signal_precision: float | None = None,
    min_signal_recall: float | None = None,
    max_signal_rate: float | None = None,
    max_flat_to_signal_rate: float | None = None,
    save_thresholds: str | None = None,
) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if model is not None:
        cfg["model"]["type"] = model          # must match the trained checkpoint
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
    else:
        edge_index = load_tmfg_edge_index(
            data_cfg["adj_matrix_path"],
            cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
        )

    static_mode = recurrent_sparse or (
        bool(cfg["training"].get("static_graph_batching", False))
        and model_type in ("gcn", "cgnn", "stgcn")
    )
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
    val_argmax = val_p.argmax(1)
    test_argmax = test_p.argmax(1)
    val_base_metrics = decision_metrics(val_argmax, val_y)
    test_base_metrics = decision_metrics(test_argmax, test_y)

    # ── Tune (t_down, t_up) on VAL, apply to TEST (shared logic) ──
    search = search_thresholds(
        val_p,
        val_y,
        lo=lo,
        hi=hi,
        step=step,
        objective=objective,
        flat_fp_penalty=flat_fp_penalty,
        min_signal_precision=min_signal_precision,
        min_signal_recall=min_signal_recall,
        max_signal_rate=max_signal_rate,
        max_flat_to_signal_rate=max_flat_to_signal_rate,
    )
    td, tu = search["t_down"], search["t_up"]
    tuned_test = apply_thresholds(test_p, td, tu)
    test_tuned_metrics = decision_metrics(tuned_test, test_y)

    print("\n" + "=" * 56)
    print("THRESHOLD TUNING RESULTS")
    print("=" * 56)
    print(f"  objective: {objective}")
    print(f"  best thresholds (tuned on val): down>={td:.2f}  up>={tu:.2f}")
    print(f"  best validation score: {search['score']:.4f}")
    print()
    _print_decision_summary("VAL argmax", val_base_metrics)
    _print_decision_summary("VAL tuned", search["metrics"])
    _print_decision_summary("TEST argmax", test_base_metrics)
    _print_decision_summary("TEST tuned", test_tuned_metrics)
    print("\n── TEST report (tuned) ─────────────────────────────")
    print(classification_report(test_y, tuned_test, target_names=["down", "flat", "up"], digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(test_y, tuned_test, labels=[0, 1, 2]))

    save_path = Path(save_thresholds) if save_thresholds else Path(checkpoint).parent / "thresholds.json"
    payload = {
        "checkpoint": str(checkpoint),
        "config": str(config_path),
        "model_type": model_type,
        "thresholds": {"down": td, "up": tu},
        "validation_argmax_metrics": val_base_metrics,
        "validation_tuned_metrics": search["metrics"],
        "test_argmax_metrics": test_base_metrics,
        "test_tuned_metrics": test_tuned_metrics,
        "search": search["search"] | {"objective": objective, "score": search["score"]},
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved thresholds to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--model", choices=["gcn", "gat", "sage", "cgnn", "stgcn", "recurrent_sparse_sthnn"], default=None,
                        help="model type of the checkpoint (es. cgnn)")
    parser.add_argument("--lo", type=float, default=0.33)
    parser.add_argument("--hi", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=["macro_f1", "macro_f1_penalized", "signal_f1", "signal_precision"],
        default="macro_f1",
    )
    parser.add_argument("--flat-fp-penalty", type=float, default=0.0)
    parser.add_argument("--min-signal-precision", type=float, default=None)
    parser.add_argument("--min-signal-recall", type=float, default=None)
    parser.add_argument("--max-signal-rate", type=float, default=None)
    parser.add_argument("--max-flat-to-signal-rate", type=float, default=None)
    parser.add_argument("--save-thresholds", default=None)
    args = parser.parse_args()
    main(
        args.config,
        args.checkpoint,
        args.model,
        lo=args.lo,
        hi=args.hi,
        step=args.step,
        objective=args.objective,
        flat_fp_penalty=args.flat_fp_penalty,
        min_signal_precision=args.min_signal_precision,
        min_signal_recall=args.min_signal_recall,
        max_signal_rate=args.max_signal_rate,
        max_flat_to_signal_rate=args.max_flat_to_signal_rate,
        save_thresholds=args.save_thresholds,
    )
