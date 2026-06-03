#!/usr/bin/env python3
"""Training entry point for LOB-GNN.

Usage:
    python scripts/train.py
    python scripts/train.py --config config/default.yaml
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
from src.dataset.preprocessing import (
    build_processed_paths,
    has_compatible_processed_dataset,
    preprocess_to_disk,
)
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model
from src.training.metrics import compute_metrics, print_report
from src.training.trainer import Trainer


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


def make_collate_fn(edge_index: torch.Tensor, batch_size: int, num_nodes: int):
    """Create a collate_fn with precomputed batched edge_index for full batches."""
    n = batch_size
    offsets = torch.arange(n, dtype=torch.long) * num_nodes
    batched_ei = (edge_index.unsqueeze(2) + offsets.view(1, 1, n)).permute(0, 2, 1).reshape(2, -1).contiguous()
    batch_vec = torch.arange(n, dtype=torch.long).repeat_interleave(num_nodes).contiguous()

    def _collate(data_list):
        m = len(data_list)
        x = torch.cat([d.x for d in data_list])
        y = torch.stack([d.y for d in data_list])

        if m == n:
            ei, bv = batched_ei, batch_vec
        else:
            offs = torch.arange(m, dtype=torch.long) * num_nodes
            ei = (edge_index.unsqueeze(2) + offs.view(1, 1, m)).permute(0, 2, 1).reshape(2, -1).contiguous()
            bv = torch.arange(m, dtype=torch.long).repeat_interleave(num_nodes).contiguous()

        return LOBBatch(x=x, edge_index=ei, batch=bv, y=y, num_graphs=m)

    return _collate


class BalancedSampler(torch.utils.data.Sampler):
    """Samples exactly n_per_class indices from each class per epoch.

    n_per_class = min(min_class_count, max_per_class).
    If max_per_class is None, uses the smallest class count (no upsampling).
    """

    def __init__(self, y_path: str | Path, max_per_class: int | None, seed: int):
        labels = np.load(y_path, mmap_mode="r")
        counts = np.bincount(labels, minlength=3)
        n = int(counts.min())
        if max_per_class is not None:
            n = min(n, int(max_per_class))
        self.n_per_class = n
        self.class_indices = [np.where(labels == c)[0] for c in range(3)]
        self.seed = seed
        self._epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        self._epoch += 1
        chosen = np.concatenate([
            rng.choice(idx, size=self.n_per_class, replace=False)
            for idx in self.class_indices
        ])
        rng.shuffle(chosen)
        return iter(chosen.tolist())

    def __len__(self) -> int:
        return self.n_per_class * 3


def build_class_weights_from_y(y_path: str | Path, device: torch.device) -> torch.Tensor:
    y = np.load(y_path, mmap_mode="r")
    counts = np.bincount(y, minlength=3)
    weights = 1.0 / (counts.astype(np.float32) + 1e-6)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

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

    # ── Edge index (static) ────────────────────────────────────
    processed_dir = Path(data_cfg["processed_dir"])
    edge_index = load_tmfg_edge_index(
        data_cfg["adj_matrix_path"],
        cache_path=processed_dir / "edge_index.pt",
    )
    print(f"\n[2/5] Graph: {2 * data_cfg['n_levels'] * (data_cfg['n_lags'] + 1)} nodes | {edge_index.shape[1]} directed edges")

    static_graph_batching = bool(cfg["training"].get("static_graph_batching", False))
    model_type = cfg["model"]["type"].lower()
    use_static_mode = static_graph_batching and model_type == "gcn"
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
    max_per_class = cfg["training"].get("max_samples_per_class")
    train_sampler = BalancedSampler(paths["y_train"], max_per_class, cfg["training"]["seed"])
    print(f"  Balanced sampler: {train_sampler.n_per_class:,} samples/class "
          f"→ {len(train_sampler):,} total/epoch"
          + (f" (cap: {max_per_class:,})" if max_per_class else " (min-class cap)"))

    if use_static_mode:
        train_loader = DataLoader(train_ds, sampler=train_sampler, drop_last=True, **loader_kw_base)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw_base)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw_base)
    else:
        n_lags = int(data_cfg["n_lags"])
        n_levels = int(data_cfg["n_levels"])
        num_nodes = 2 * n_levels * (n_lags + 1)
        collate_fn = make_collate_fn(edge_index, cfg["training"]["batch_size"], num_nodes)
        loader_kw = dict(loader_kw_base)
        loader_kw["collate_fn"] = collate_fn
        train_loader = DataLoader(train_ds, sampler=train_sampler, drop_last=True, **loader_kw)
        val_loader = DataLoader(val_ds, shuffle=False, drop_last=False, **loader_kw)
        test_loader = DataLoader(test_ds, shuffle=False, drop_last=False, **loader_kw)

    # ── Model ──────────────────────────────────────────────────
    print(f"\n[3/5] Initializing model ({cfg['model']['type'].upper()})...")
    model = build_model(cfg).to(device)
    print(f"  Trainable parameters: {model.count_parameters():,}")

    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        class_weights = build_class_weights_from_y(paths["y_train"], device)
        print(f"  Class weights: {class_weights.cpu().numpy().round(3)}")

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=cfg["training"]["early_stopping_patience"] // 2,
        factor=0.5,
    )

    ckpt_dir = Path(cfg["paths"]["checkpoints"])
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
    preds, labels = trainer.predict(test_loader)
    metrics = compute_metrics(preds, labels)

    print("\n── Test Results ─────────────────────────────────────")
    print(f"  Accuracy:    {metrics['accuracy']:.4f}")
    print(f"  F1 Macro:    {metrics['f1_macro']:.4f}")
    print(f"  F1 Weighted: {metrics['f1_weighted']:.4f}")
    print()
    print_report(preds, labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    main(args.config)
