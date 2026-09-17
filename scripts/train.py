#!/usr/bin/env python3
"""Training entry point for LOB-GNN.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/gnn/default.yaml
"""

import argparse
import csv
import json
import random
import sys
import time
from uuid import uuid4
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config

from remoma.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from remoma.dataset.preprocessing import (
    build_processed_paths,
    has_compatible_processed_dataset,
    preprocess_to_disk,
)
from remoma.graph.adjacency import load_tmfg_edge_index
from remoma.models import build_model, is_recurrent_sparse_model, model_names, model_family, supports_static_batching
from remoma.config import project_path
from remoma.utils.checkpoints import save_run_configuration
from remoma.training.metrics import compute_metrics, format_report
from remoma.training.trainer import Trainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LOBBatch:
    """Lightweight batch container with cached batched edge_index."""

    __slots__ = ("x", "edge_index", "batch", "y", "num_graphs")

    def __init__(self, x, edge_index, batch, y, num_graphs):
        self.x = x
        self.edge_index = edge_index
        self.batch = batch
        self.y = y
        self.num_graphs = num_graphs

    def to(self, device):
        return LOBBatch(
            x=self.x.to(device, non_blocking=True),
            edge_index=self.edge_index.to(device, non_blocking=True),
            batch=self.batch.to(device, non_blocking=True),
            y=self.y.to(device, non_blocking=True),
            num_graphs=self.num_graphs,
        )


class LOBCollator:
    """Picklable graph collator with cached edges for a full batch."""
    def __init__(self, edge_index, batch_size, num_nodes):
        self.edge_index = edge_index
        self.batch_size = batch_size
        self.num_nodes = num_nodes
        self.full_edges, self.full_batch = self._graph(batch_size)

    def _graph(self, size):
        offsets = torch.arange(size, dtype=torch.long) * self.num_nodes
        edges = (self.edge_index.unsqueeze(2) + offsets.view(1, 1, size)).permute(0, 2, 1).reshape(2, -1).contiguous()
        batch = torch.arange(size, dtype=torch.long).repeat_interleave(self.num_nodes)
        return edges, batch

    def __call__(self, data_list):
        size = len(data_list)
        edges, batch = (self.full_edges, self.full_batch) if size == self.batch_size else self._graph(size)
        return LOBBatch(torch.cat([item.x for item in data_list]), edges, batch,
                        torch.stack([item.y for item in data_list]), size)


def make_collate_fn(edge_index, batch_size, num_nodes):
    return LOBCollator(edge_index, batch_size, num_nodes)


def build_class_weights_from_y(y_path: str | Path, device: torch.device) -> torch.Tensor:
    y = np.load(y_path, mmap_mode="r")
    counts = np.bincount(y, minlength=3)
    weights = 1.0 / (counts.astype(np.float32) + 1e-6)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32, device=device)


SUMMARY_HEADER = [
    "timestamp", "model", "split", "rule", "f1_macro", "mcc", "accuracy",
    "f1_weighted", "f1_down", "f1_flat", "f1_up", "thr_down", "thr_up", "params",
]


def _summary_row(stamp, model, split, rule, m, td, tu, params):
    fc = m.get("f1_per_class", [0.0, 0.0, 0.0])
    thr_d = "" if td == "" else f"{td:.2f}"
    thr_u = "" if tu == "" else f"{tu:.2f}"
    return [
        stamp, model, split, rule,
        f"{m['f1_macro']:.4f}", f"{m['mcc']:.4f}", f"{m['accuracy']:.4f}",
        f"{m['f1_weighted']:.4f}", f"{fc[0]:.4f}", f"{fc[1]:.4f}", f"{fc[2]:.4f}",
        thr_d, thr_u, params,
    ]


def _append_summary(path: Path, rows: list[list]) -> None:
    exists = path.exists()
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(SUMMARY_HEADER)
        w.writerows(rows)


def main(config_path: str, overrides: dict | None = None) -> None:
    cfg = load_config(config_path)

    # CLI overrides (so you can switch architecture without editing the yaml)
    overrides = overrides or {}
    if overrides.get("model") is not None:
        cfg["model"]["type"] = overrides["model"]
        cfg["model"]["family"] = model_family(overrides["model"])
    if overrides.get("hidden") is not None:
        # write into the selected arch's preset so it wins over the default
        mt = cfg["model"]["type"].lower()
        key = "hidden_dim" if is_recurrent_sparse_model(mt) else "hidden_channels"
        cfg["model"].setdefault("overrides", {}).setdefault(mt, {})[key] = overrides["hidden"]
    if overrides.get("cnn_channels") is not None:
        # scale the temporal-CNN intermediate width (cgnn* only) for capacity tests
        mt = cfg["model"]["type"].lower()
        cfg["model"].setdefault("overrides", {}).setdefault(mt, {})["cnn_channels"] = overrides["cnn_channels"]
    if overrides.get("lr") is not None:
        cfg["training"]["lr"] = float(overrides["lr"])
    if overrides.get("epochs") is not None:
        cfg["training"]["epochs"] = int(overrides["epochs"])
    if overrides.get("split") is not None:
        cfg["data"]["split_strategy"] = overrides["split"]
    if overrides.get("seed") is not None:
        cfg["training"]["seed"] = int(overrides["seed"])
    if overrides.get("checkpoint_dir") is not None:
        cfg["paths"]["checkpoints"] = str(overrides["checkpoint_dir"])
    applied = {k: v for k, v in overrides.items() if v is not None and k != "config"}
    if applied:
        print(f"CLI overrides: {applied}\n")

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Build / reuse processed splits ─────────────────────────
    data_cfg = cfg["data"]
    use_precomputed = data_cfg.get("use_precomputed", True)
    if not use_precomputed:
        raise ValueError("This training script now requires data.use_precomputed=true.")

    force_preprocess = bool(data_cfg.get("force_preprocess", False))

    print("[1/5] Preparing processed dataset...")
    if force_preprocess or not has_compatible_processed_dataset(cfg):
        preprocess_to_disk(cfg, force=force_preprocess, verbose=True)
    else:
        print(f"Using cached processed tensors in '{data_cfg['processed_dir']}'.")

    paths = build_processed_paths(data_cfg["processed_dir"])

    # ── Graph structure ────────────────────────────────────────
    processed_dir = Path(data_cfg["processed_dir"])
    model_type = cfg["model"]["type"].lower()
    recurrent_sparse = is_recurrent_sparse_model(model_type)
    num_nodes = 2 * data_cfg["n_levels"] * (data_cfg["n_lags"] + 1)

    if recurrent_sparse:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        print(
            f"\n[2/5] Recurrent sparse graph: {num_nodes} nodes | "
            f"adjacency={data_cfg['adj_matrix_path']}"
        )
    else:
        edge_index = load_tmfg_edge_index(
            data_cfg["adj_matrix_path"],
            n_lags=int(data_cfg["n_lags"]), n_levels=int(data_cfg["n_levels"]),
            cache_path=processed_dir / "edge_index.pt",
        )
        print(f"\n[2/5] Graph: {num_nodes} nodes | {edge_index.shape[1]} directed edges")

    static_graph_batching = bool(cfg["training"].get("static_graph_batching", False))
    # GAT operators require flattened PyG batches; recurrent models use tensors.
    use_static_mode = recurrent_sparse or (
        static_graph_batching and supports_static_batching(model_type)
    )
    if static_graph_batching and not use_static_mode:
        print(
            f"[warn] static_graph_batching requested, but model.type='{model_type}' "
            "is not supported by the static path. Falling back to PyG batching."
        )

    # ── Datasets / loaders ─────────────────────────────────────
    if use_static_mode:
        train_ds = StaticLOBTensorDataset(str(paths["X_train"]), str(paths["y_train"]))
        val_ds = StaticLOBTensorDataset(str(paths["X_val"]), str(paths["y_val"]))
        test_ds = StaticLOBTensorDataset(str(paths["X_test"]), str(paths["y_test"]))
    else:
        train_ds = FastLOBDataset(str(paths["X_train"]), str(paths["y_train"]), edge_index)
        val_ds = FastLOBDataset(str(paths["X_val"]), str(paths["y_val"]), edge_index)
        test_ds = FastLOBDataset(str(paths["X_test"]), str(paths["y_test"]), edge_index)
    print(f"  Samples: train={len(train_ds):,} | val={len(val_ds):,} | test={len(test_ds):,}")

    loader_kw_base = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 0),
        persistent_workers=cfg["training"].get("num_workers", 0) > 0,
        pin_memory=(device.type == "cuda"),
    )
    if use_static_mode:
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw_base)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw_base)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw_base)
    else:
        n_lags = int(data_cfg["n_lags"])
        n_levels = int(data_cfg["n_levels"])
        collate_fn = make_collate_fn(edge_index, cfg["training"]["batch_size"], num_nodes)
        loader_kw = dict(loader_kw_base)
        loader_kw["collate_fn"] = collate_fn
        train_loader = DataLoader(train_ds, shuffle=True, drop_last=False, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw)

    # ── Model ──────────────────────────────────────────────────
    print(f"\n[3/5] Initializing model ({cfg['model']['type'].upper()})...")
    model = build_model(cfg).to(device)
    print(f"  Trainable parameters: {model.count_parameters():,}")
    if recurrent_sparse:
        print(
            "  Recurrent blocks: "
            f"{model.recurrent_edge_key_count:,} | temporal edges: "
            f"{model.recurrent_temporal_edge_count:,} | same-lag edges: "
            f"{model.recurrent_same_lag_edge_count:,}"
        )
        print(
            "  Edge weights: "
            f"{getattr(model, 'use_edge_weights', False)}"
            + (
                f" ({getattr(model, 'edge_weight_normalization', 'none')})"
                if getattr(model, "use_edge_weights", False)
                else ""
            )
        )

    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        class_weights = build_class_weights_from_y(paths["y_train"], device)
        print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.90, 0.95),
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["training"].get("lr_scheduler_patience", 3),
        factor=cfg["training"].get("lr_scheduler_factor", 0.5),
    )

    if overrides.get("checkpoint_dir"):
        ckpt_dir = project_path(overrides["checkpoint_dir"])
    else:
        run_name = (f"{model_type}_{data_cfg['split_strategy']}_seed{cfg['training']['seed']}_"
                    f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}")
        ckpt_dir = Path(cfg["paths"]["checkpoints"]) / run_name
    if any((ckpt_dir / name).exists() for name in ("best.pt", "resolved_config.yaml")):
        raise FileExistsError(f"Run directory is already in use: {ckpt_dir}. Select a new --checkpoint-dir.")
    cfg = save_run_configuration(cfg, ckpt_dir)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=ckpt_dir,
        patience=cfg["training"]["early_stopping_patience"],
        grad_clip=cfg["training"].get("grad_clip", 1.0),
        static_graph_batching=use_static_mode,
        static_edge_index=edge_index.to(device, non_blocking=True) if use_static_mode else None,
    )

    # ── Training ───────────────────────────────────────────────
    print(f"\n[4/5] Training for up to {cfg['training']['epochs']} epochs...\n")
    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    # ── Test evaluation ────────────────────────────────────────
    print("\n[5/5] Evaluating best checkpoint on test split...")
    trainer.load_best()

    results_dir = ckpt_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    split = data_cfg["split_strategy"]
    model_name = cfg["model"]["type"].lower()
    n_params = model.count_parameters()
    stamp = time.strftime("%Y%m%d_%H%M%S")

    lines = [
        "── Test Results ─────────────────────────────────────",
        f"  Model: {model_name} | Split: {split} | Params: {n_params:,}",
    ]
    summary_rows: list[list] = []

    if cfg["training"].get("tune_threshold", False):
        from remoma.training.threshold import apply_thresholds, decision_metrics, search_thresholds

        val_probs, val_labels = trainer.predict_proba(val_loader)
        test_probs, test_labels = trainer.predict_proba(test_loader)
        search = search_thresholds(val_probs, val_labels)
        td, tu = search["t_down"], search["t_up"]

        argmax_preds = test_probs.argmax(1)
        tuned_preds = apply_thresholds(test_probs, td, tu)
        m_arg = compute_metrics(argmax_preds, test_labels)
        m_tuned = compute_metrics(tuned_preds, test_labels)
        signal_arg = decision_metrics(argmax_preds, test_labels)
        signal_tuned = decision_metrics(tuned_preds, test_labels)

        lines.append(f"  F1 Macro (argmax): {m_arg['f1_macro']:.4f}   MCC (argmax): {m_arg['mcc']:.4f}")
        lines.append(f"  F1 Macro (tuned):  {m_tuned['f1_macro']:.4f}   MCC (tuned):  {m_tuned['mcc']:.4f}   "
                     f"(down>={td:.2f} up>={tu:.2f} | val_f1={search['metrics']['f1_macro']:.3f})")
        lines.append(f"  Accuracy (tuned):  {m_tuned['accuracy']:.4f}")
        lines.append("")
        lines.append(format_report(tuned_preds, test_labels))
        summary_rows.append(_summary_row(stamp, model_name, split, "argmax", m_arg, "", "", n_params))
        summary_rows.append(_summary_row(stamp, model_name, split, "tuned", m_tuned, td, tu, n_params))
        lines.append(f"  Signal precision: {signal_arg['signal_precision']:.4f} -> {signal_tuned['signal_precision']:.4f}")
        lines.append(f"  Flat-to-signal rate: {signal_arg['flat_to_signal_rate']:.4f} -> {signal_tuned['flat_to_signal_rate']:.4f}")

        threshold_payload = {
            "checkpoint": str(ckpt_dir / "best.pt"),
            "config": str(config_path),
            "model_type": model_type,
            "thresholds": {"down": td, "up": tu},
            "validation_tuned_metrics": search["metrics"],
            "test_argmax_metrics": signal_arg,
            "test_tuned_metrics": signal_tuned,
            "search": search["search"] | {"objective": search["objective"], "score": search["score"]},
        }
        threshold_path = ckpt_dir / "thresholds.json"
        threshold_path.write_text(json.dumps(threshold_payload, indent=2), encoding="utf-8")
        print(f"\nThresholds saved to {threshold_path}")
    else:
        preds, labels = trainer.predict(test_loader)
        metrics = compute_metrics(preds, labels)
        lines.append(f"  Accuracy:    {metrics['accuracy']:.4f}")
        lines.append(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
        lines.append(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
        lines.append(f"  MCC:         {metrics['mcc']:.4f}")
        lines.append("")
        lines.append(format_report(preds, labels))
        summary_rows.append(_summary_row(stamp, model_name, split, "argmax", metrics, "", "", n_params))

    report_text = "\n".join(lines)
    print("\n" + report_text)

    txt_path = results_dir / f"{model_name}_{split}_{stamp}.txt"
    txt_path.write_text(report_text + "\n")
    _append_summary(results_dir / "summary.csv", summary_rows)
    print(f"\n[saved] full report → {txt_path}")
    print(f"[saved] summary → {results_dir / 'summary.csv'}")
    print(f"[saved] weights → {ckpt_dir / 'best.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn/default.yaml")
    parser.add_argument("--model",
                        choices=model_names(),
                        default=None,
                        help="override model.type (e.g. stgcn)")
    parser.add_argument("--hidden", type=int, default=None,
                        help="override hidden_channels for GNNs or hidden_dim for recurrent models")
    parser.add_argument("--cnn-channels", dest="cnn_channels", type=int, default=None,
                        help="override the temporal CNN width for CGNN capacity experiments")
    parser.add_argument("--lr", type=float, default=None,
                        help="override training.lr")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override training.epochs")
    parser.add_argument("--split", choices=["by_lag", "by_file"], default=None,
                        help="override the temporal split strategy")
    parser.add_argument("--seed", type=int, default=None,
                        help="override the model initialization seed (independent of data sampling)")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="override paths.checkpoints")
    args = parser.parse_args()
    main(args.config, vars(args))
