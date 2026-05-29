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

from src.dataset.binning import VolumeBinner
from src.dataset.labeling import compute_labels
from src.dataset.lob_dataset import LOBDataset
from src.graph.adjacency import load_tmfg_edge_index
from src.models import build_model
from src.training.metrics import compute_metrics, print_report
from src.training.trainer import Trainer
from src.utils.io import discover_files, load_lobster_csv

_ASK_P_COLS = np.arange(0, 40, 4)
_BID_P_COLS = np.arange(2, 40, 4)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LOBBatch:
    """Batch container leggero: evita la ricalcolazione degli offset edge_index per ogni batch.

    Tutti i grafi LOB condividono la stessa struttura (stesso TMFG edge_index),
    quindi il batched edge_index è identico per ogni batch di uguali dimensioni.
    """
    __slots__ = ("x", "edge_index", "batch", "y", "num_graphs")

    def __init__(self, x, edge_index, batch, y, num_graphs):
        self.x          = x
        self.edge_index = edge_index
        self.batch      = batch
        self.y          = y
        self.num_graphs = num_graphs

    def to(self, device):
        return LOBBatch(
            x          = self.x.to(device, non_blocking=True),
            edge_index = self.edge_index.to(device, non_blocking=True),
            batch      = self.batch.to(device, non_blocking=True),
            y          = self.y.to(device, non_blocking=True),
            num_graphs = self.num_graphs,
        )


def make_collate_fn(edge_index: torch.Tensor, batch_size: int, num_nodes: int):
    """Crea una collate_fn con edge_index batched precomputato per batch interi.

    Risparmia ~O(batch_size × num_edges) operazioni di offset ad ogni batch.
    Per batch parziali (ultimo batch di val/test) ricade sul calcolo on-the-fly.
    """
    N = batch_size
    offsets = torch.arange(N, dtype=torch.long) * num_nodes          # [N]
    # [2, E, 1] + [1, 1, N] → [2, E, N] → [2, N, E] → [2, N*E]
    batched_ei = (
        edge_index.unsqueeze(2) + offsets.view(1, 1, N)
    ).permute(0, 2, 1).reshape(2, -1).contiguous()
    batch_vec = torch.arange(N, dtype=torch.long).repeat_interleave(num_nodes).contiguous()

    def _collate(data_list):
        n = len(data_list)
        x = torch.cat([d.x for d in data_list])
        y = torch.stack([d.y for d in data_list])

        if n == N:
            ei, bv = batched_ei, batch_vec
        else:
            # Ultimo batch parziale (val/test): calcolo on-the-fly
            offs = torch.arange(n, dtype=torch.long) * num_nodes
            ei   = (edge_index.unsqueeze(2) + offs.view(1, 1, n)).permute(0, 2, 1).reshape(2, -1).contiguous()
            bv   = torch.arange(n, dtype=torch.long).repeat_interleave(num_nodes).contiguous()

        return LOBBatch(x=x, edge_index=ei, batch=bv, y=y, num_graphs=n)

    return _collate


def _make_samples(file_indices: list[int], all_data: list[np.ndarray], n_lags: int, k: int) -> np.ndarray:
    """Vectorized construction of (file_idx, t) sample pairs."""
    parts = []
    for fi in file_indices:
        t_start = n_lags
        t_end = len(all_data[fi]) - 1 - k  # inclusive; need data[t+k] for label
        n = t_end - t_start + 1
        parts.append(np.stack([
            np.full(n, fi, dtype=np.int32),
            np.arange(t_start, t_end + 1, dtype=np.int32),
        ], axis=1))
    return np.concatenate(parts, axis=0)


def split_by_file(all_data, cfg, n_lags, k):
    return (
        _make_samples(cfg["data"]["train_files"], all_data, n_lags, k),
        _make_samples(cfg["data"]["val_files"],   all_data, n_lags, k),
        _make_samples(cfg["data"]["test_files"],  all_data, n_lags, k),
    )


def split_by_lag(all_data, cfg, n_lags, k):
    r_train = cfg["data"]["train_ratio"]
    r_val   = cfg["data"]["val_ratio"]
    train_parts, val_parts, test_parts = [], [], []

    for fi, data in enumerate(all_data):
        t_start = n_lags
        t_end = len(data) - 1 - k
        n = t_end - t_start + 1
        t1 = t_start + int(r_train * n)
        t2 = t_start + int((r_train + r_val) * n)

        def _block(a, b):
            k = b - a
            return np.stack([np.full(k, fi, dtype=np.int32), np.arange(a, b, dtype=np.int32)], axis=1)

        train_parts.append(_block(t_start, t1))
        val_parts.append(_block(t1, t2))
        test_parts.append(_block(t2, t_end + 1))

    return (
        np.concatenate(train_parts),
        np.concatenate(val_parts),
        np.concatenate(test_parts),
    )


def build_price_stats(train_files: list[int], all_data: list[np.ndarray]) -> dict:
    data = np.concatenate([all_data[i] for i in train_files], axis=0)
    return {
        "ask_mean": data[:, _ASK_P_COLS].mean(0).astype(np.float32),
        "ask_std":  data[:, _ASK_P_COLS].std(0).astype(np.float32),
        "bid_mean": data[:, _BID_P_COLS].mean(0).astype(np.float32),
        "bid_std":  data[:, _BID_P_COLS].std(0).astype(np.float32),
    }


def build_class_weights(
    train_samples: np.ndarray,
    all_labels: list[np.ndarray],
    device: torch.device,
) -> torch.Tensor:
    counts = np.zeros(3, dtype=np.int64)
    for fi in range(len(all_labels)):
        mask = train_samples[:, 0] == fi
        if not mask.any():
            continue
        counts += np.bincount(all_labels[fi][train_samples[mask, 1]], minlength=3)
    weights = 1.0 / (counts.astype(np.float32) + 1e-6)
    weights = weights / weights.sum() * 3
    return torch.tensor(weights, dtype=torch.float32, device=device)


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["training"]["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    n_lags = cfg["data"]["n_lags"]
    k      = cfg["data"].get("prediction_horizon", 1)

    # ── Load data ──────────────────────────────────────────────
    print("[1/5] Loading LOB files...")
    files = discover_files(cfg["data"]["raw_dir"])
    if not files:
        raise FileNotFoundError(f"No LOBSTER files found in '{cfg['data']['raw_dir']}'")
    all_data = [load_lobster_csv(f) for f in files]
    for f, d in zip(files, all_data):
        print(f"  {f.name}: {d.shape[0]:,} ticks")

    # ── Labels ─────────────────────────────────────────────────
    threshold   = cfg["data"]["threshold"]
    price_type  = cfg["data"].get("price_type", "mid")
    all_labels  = [compute_labels(d, threshold, k, price_type) for d in all_data]

    # ── Split ──────────────────────────────────────────────────
    strategy = cfg["data"]["split_strategy"]
    if strategy == "by_file":
        train_s, val_s, test_s = split_by_file(all_data, cfg, n_lags, k)
    elif strategy == "by_lag":
        train_s, val_s, test_s = split_by_lag(all_data, cfg, n_lags, k)
    else:
        raise ValueError(f"Unknown split_strategy '{strategy}'")

    print(f"\n[2/5] Split ({strategy}): "
          f"train={len(train_s):,} | val={len(val_s):,} | test={len(test_s):,}")

    # ── Preprocessing (fit on train only) ──────────────────────
    print("\n[3/5] Fitting preprocessors on training data...")
    train_file_indices = sorted(set(int(r[0]) for r in train_s))
    train_data_list = [all_data[i] for i in train_file_indices]

    binner = VolumeBinner(n_bins=cfg["data"]["n_volume_bins"]).fit(train_data_list)

    price_stats = None
    if cfg["data"].get("normalize_prices", True):
        price_stats = build_price_stats(train_file_indices, all_data)

    # Save artifacts for later evaluation
    ckpt_dir = Path(cfg["paths"]["checkpoints"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    binner.save(ckpt_dir / "binner.pkl")
    if price_stats is not None:
        np.save(ckpt_dir / "price_stats.npy", price_stats)

    # ── Edge index ─────────────────────────────────────────────
    processed_dir = Path(cfg["data"]["processed_dir"])
    edge_index = load_tmfg_edge_index(
        cfg["data"]["adj_matrix_path"],
        cache_path=processed_dir / "edge_index.pt",
    )
    print(f"  Graph: 3020 nodes | {edge_index.shape[1]} directed edges")

    # ── Datasets and loaders ───────────────────────────────────
    ds_kwargs = dict(
        file_data=all_data,
        file_labels=all_labels,
        edge_index=edge_index,
        binner=binner,
        price_stats=price_stats,
        n_lags=n_lags,
    )
    train_ds = LOBDataset(train_s, **ds_kwargs)
    val_ds   = LOBDataset(val_s,   **ds_kwargs)
    test_ds  = LOBDataset(test_s,  **ds_kwargs)

    num_nodes  = 2 * cfg["data"]["n_levels"] * (n_lags + 1)   # 2*10*151 = 3020
    collate_fn = make_collate_fn(edge_index, cfg["training"]["batch_size"], num_nodes)

    loader_kw = dict(
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"].get("num_workers", 0),
        persistent_workers=cfg["training"].get("num_workers", 0) > 0,
        collate_fn=collate_fn,
    )
    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kw)

    # ── Model ──────────────────────────────────────────────────
    print(f"\n[4/5] Initializing model ({cfg['model']['type'].upper()})...")
    model = build_model(cfg).to(device)
    print(f"  Trainable parameters: {model.count_parameters():,}")

    class_weights = None
    if cfg["training"].get("use_class_weights", True):
        class_weights = build_class_weights(train_s, all_labels, device)
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

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device=device,
        checkpoint_dir=ckpt_dir,
        patience=cfg["training"]["early_stopping_patience"],
        grad_clip=cfg["training"].get("grad_clip", 1.0),
    )

    # ── Training ───────────────────────────────────────────────
    print(f"\n[5/5] Training for up to {cfg['training']['epochs']} epochs...\n")
    trainer.fit(train_loader, val_loader, epochs=cfg["training"]["epochs"])

    # ── Test evaluation ────────────────────────────────────────
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
