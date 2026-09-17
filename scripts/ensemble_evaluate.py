#!/usr/bin/env python3
"""Evaluate a probability-averaged ensemble of trained LOB-GNN checkpoints.

The script keeps each trained model fixed, averages their class probabilities,
tunes the final down/up decision thresholds on the validation split, and applies
the same thresholds to the test split.

By default the threshold objective is plain macro F1, with no flat false-positive
penalty. This is intentional for the first bagging experiment: it isolates the
effect of probability averaging from the more conservative penalized rule.

Usage:
    python scripts/ensemble_evaluate.py \
      --config configs/recurrent/recurrent_sparse_sthnn.yaml \
      --checkpoints checkpoints/seed42/best.pt checkpoints/seed123/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config
from remoma.utils.checkpoints import load_checkpoint_state, checkpoint_config

from remoma.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from remoma.dataset.preprocessing import (
    build_processed_paths,
    has_compatible_processed_dataset,
    preprocess_to_disk,
)
from remoma.graph.adjacency import load_tmfg_edge_index
from remoma.models import build_model, is_recurrent_sparse_model
from remoma.training.threshold import apply_thresholds, decision_metrics, search_thresholds


def _print_decision_summary(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:<18} "
        f"acc={metrics['accuracy']:.4f}  "
        f"macro_f1={metrics['f1_macro']:.4f}  "
        f"weighted_f1={metrics['f1_weighted']:.4f}  "
        f"signal_prec={metrics['signal_precision']:.4f}  "
        f"signal_rec={metrics['signal_recall']:.4f}  "
        f"signal_rate={metrics['signal_rate']:.4f}  "
        f"flat_to_signal={metrics['flat_to_signal_rate']:.4f}"
    )


def _build_loaders(cfg: dict, device: torch.device):
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
            n_lags=int(data_cfg["n_lags"]), n_levels=int(data_cfg["n_levels"]),
            cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
        )

    static_mode = recurrent_sparse or (
        bool(cfg["training"].get("static_graph_batching", False))
        and model_type in ("gcn", "cgnn", "stgcn")
    )
    static_edge = edge_index.to(device) if static_mode else None

    batch_size = int(cfg["training"]["batch_size"])
    num_workers = int(cfg["training"].get("num_workers", 0))

    def make_loader(x_path, y_path):
        if static_mode:
            ds = StaticLOBTensorDataset(str(x_path), str(y_path))
            return DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

        from scripts.train import make_collate_fn

        ds = FastLOBDataset(str(x_path), str(y_path), edge_index)
        n_nodes = 2 * int(data_cfg["n_levels"]) * (int(data_cfg["n_lags"]) + 1)
        collate_fn = make_collate_fn(edge_index, batch_size, n_nodes)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    val_loader = make_loader(paths["X_val"], paths["y_val"])
    test_loader = make_loader(paths["X_test"], paths["y_test"])
    return val_loader, test_loader, static_edge, static_mode


@torch.no_grad()
def _predict_probs(model, loader, edge_index, device: torch.device, static_mode: bool):
    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

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


def _load_checkpoint_probs(
    cfg: dict,
    checkpoint: Path,
    val_loader,
    test_loader,
    edge_index,
    device: torch.device,
    static_mode: bool,
):
    model = build_model(cfg).to(device)
    load_checkpoint_state(model, checkpoint, cfg, device)

    val_probs, val_labels = _predict_probs(model, val_loader, edge_index, device, static_mode)
    test_probs, test_labels = _predict_probs(model, test_loader, edge_index, device, static_mode)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_probs, val_labels, test_probs, test_labels


def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)

    checkpoints = [Path(path) for path in args.checkpoints]
    missing = [path for path in checkpoints if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_type = cfg["model"]["type"].lower()
    print(f"Device: {device}")
    print(f"Model: {model_type}")
    print(f"Checkpoints: {len(checkpoints)}")

    val_loader, test_loader, edge_index, static_mode = _build_loaders(cfg, device)

    val_sum = test_sum = None
    val_labels_ref = test_labels_ref = None
    members = []

    for idx, checkpoint in enumerate(checkpoints, start=1):
        print(f"\n[{idx}/{len(checkpoints)}] Loading {checkpoint}")
        val_probs, val_labels, test_probs, test_labels = _load_checkpoint_probs(
            cfg,
            checkpoint,
            val_loader,
            test_loader,
            edge_index,
            device,
            static_mode,
        )

        if val_labels_ref is None:
            val_labels_ref = val_labels
            test_labels_ref = test_labels
            val_sum = np.zeros_like(val_probs, dtype=np.float64)
            test_sum = np.zeros_like(test_probs, dtype=np.float64)
        else:
            if not np.array_equal(val_labels_ref, val_labels):
                raise ValueError(f"Validation labels differ for checkpoint {checkpoint}")
            if not np.array_equal(test_labels_ref, test_labels):
                raise ValueError(f"Test labels differ for checkpoint {checkpoint}")

        val_sum += val_probs
        test_sum += test_probs

        member_metrics = decision_metrics(test_probs.argmax(1), test_labels)
        members.append({"checkpoint": str(checkpoint), "test_argmax_metrics": member_metrics})
        _print_decision_summary("member argmax", member_metrics)

    assert val_sum is not None and test_sum is not None
    assert val_labels_ref is not None and test_labels_ref is not None

    val_probs = val_sum / len(checkpoints)
    test_probs = test_sum / len(checkpoints)
    val_labels = val_labels_ref
    test_labels = test_labels_ref

    val_argmax = val_probs.argmax(1)
    test_argmax = test_probs.argmax(1)
    val_argmax_metrics = decision_metrics(val_argmax, val_labels)
    test_argmax_metrics = decision_metrics(test_argmax, test_labels)

    search = search_thresholds(
        val_probs,
        val_labels,
        lo=args.lo,
        hi=args.hi,
        step=args.step,
        objective=args.objective,
        flat_fp_penalty=args.flat_fp_penalty,
        min_signal_precision=args.min_signal_precision,
        min_signal_recall=args.min_signal_recall,
        max_signal_rate=args.max_signal_rate,
        max_flat_to_signal_rate=args.max_flat_to_signal_rate,
    )
    td, tu = search["t_down"], search["t_up"]
    tuned_test = apply_thresholds(test_probs, td, tu)
    test_tuned_metrics = decision_metrics(tuned_test, test_labels)

    print("\n" + "=" * 64)
    print("ENSEMBLE RESULTS")
    print("=" * 64)
    print(f"objective: {args.objective}")
    print(f"flat_fp_penalty: {args.flat_fp_penalty:.4f}")
    print(f"best thresholds tuned on val: down>={td:.2f}  up>={tu:.2f}")
    print(f"best validation score: {search['score']:.4f}")
    print()
    _print_decision_summary("VAL ens argmax", val_argmax_metrics)
    _print_decision_summary("VAL ens tuned", search["metrics"])
    _print_decision_summary("TEST ens argmax", test_argmax_metrics)
    _print_decision_summary("TEST ens tuned", test_tuned_metrics)

    print("\nTEST report (ensemble tuned)")
    print(classification_report(test_labels, tuned_test, target_names=["down", "flat", "up"], digits=4, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, tuned_test, labels=[0, 1, 2]))

    save_path = (
        Path(args.save_thresholds)
        if args.save_thresholds is not None
        else Path(cfg["paths"]["checkpoints"]) / "ensemble_thresholds.json"
    )
    payload = {
        "config": str(args.config),
        "model_type": model_type,
        "checkpoints": [str(path) for path in checkpoints],
        "thresholds": {"down": td, "up": tu},
        "members": members,
        "validation_ensemble_argmax_metrics": val_argmax_metrics,
        "validation_ensemble_tuned_metrics": search["metrics"],
        "test_ensemble_argmax_metrics": test_argmax_metrics,
        "test_ensemble_tuned_metrics": test_tuned_metrics,
        "search": search["search"] | {"objective": args.objective, "score": search["score"]},
    }
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved ensemble thresholds to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/recurrent/recurrent_sparse_sthnn.yaml")
    parser.add_argument("--checkpoints", nargs="+", required=True)
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
    main(parser.parse_args())
