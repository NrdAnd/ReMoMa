#!/usr/bin/env python3
"""Diagnostic harness to localize why the GNN is not learning.

Runs three independent tests on the precomputed dataset + model:

  A. forward() vs forward_static() equivalence on the same batch.
  B. Feature statistics (per-channel min/max/mean/std) on real samples.
  C. Single-batch overfit: can the model drive loss -> 0 on ~256 samples?

Usage:
    python scripts/diagnose.py --config configs/gnn/default.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch_geometric.data import Batch, Data

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config

from remoma.dataset.preprocessing import build_processed_paths
from remoma.dataset.indexed import FeatureReader
from remoma.graph.adjacency import load_tmfg_edge_index
from remoma.models import build_model, is_recurrent_sparse_model, supports_static_batching


def load_real_batch(paths, n: int, num_nodes: int, balanced: bool = True):
    """Load n real samples from the train split (optionally class-balanced)."""
    reader = FeatureReader(paths["X_train"])
    y = np.load(paths["y_train"], mmap_mode="r")
    y_full = np.asarray(y)

    if balanced:
        per = n // 3
        idx = []
        rng = np.random.default_rng(0)
        for c in range(3):
            pool = np.where(y_full == c)[0]
            idx.append(rng.choice(pool, size=min(per, len(pool)), replace=False))
        idx = np.sort(np.concatenate(idx))
    else:
        idx = np.arange(min(n, len(y)))

    x_np = reader.read(idx)
    y_np = y_full[idx].astype(np.int64)
    assert x_np.shape[1] == num_nodes, (x_np.shape, num_nodes)
    return torch.from_numpy(x_np), torch.from_numpy(y_np)


def test_a_equivalence(model, x, edge_index, device):
    print("\n" + "=" * 60)
    print("TEST A — forward() vs forward_static() equivalence")
    print("=" * 60)
    model.eval()  # deterministic: no dropout, BN uses running stats
    x = x.to(device)
    ei = edge_index.to(device)
    B, N, Fdim = x.shape

    with torch.no_grad():
        # static path
        out_static = model.forward_static(x, ei)

        # PyG path: build a real batched graph from the same x
        data_list = [Data(x=x[i], edge_index=ei, y=torch.tensor(0)) for i in range(B)]
        batch = Batch.from_data_list(data_list).to(device)
        out_pyg = model(batch)

    diff = (out_static - out_pyg).abs()
    print(f"  out_static shape : {tuple(out_static.shape)}")
    print(f"  out_pyg    shape : {tuple(out_pyg.shape)}")
    print(f"  max abs diff     : {diff.max().item():.3e}")
    print(f"  mean abs diff    : {diff.mean().item():.3e}")
    if diff.max().item() < 1e-3:
        print("  => EQUIVALENT. forward_static is NOT the bug.")
    else:
        print("  => DIVERGENT. forward_static has a real bug.")


def test_b_features(x, y):
    print("\n" + "=" * 60)
    print("TEST B — feature statistics on real samples")
    print("=" * 60)
    # x: [n, num_nodes, 2]  channel 0 = price, channel 1 = volume
    price = x[..., 0].numpy()
    vol = x[..., 1].numpy()
    half = x.shape[1] // 2  # ask nodes [:half], bid nodes [half:]

    def stats(name, a):
        print(f"  {name:18s} min={a.min():+.4e}  max={a.max():+.4e}  "
              f"mean={a.mean():+.4e}  std={a.std():.4e}")

    stats("price (all)", price)
    stats("price (ask)", price[:, :half])
    stats("price (bid)", price[:, half:])
    stats("volume (all)", vol)
    print(f"  price/volume std ratio: {price.std() / (vol.std() + 1e-12):.2e}")

    # per-sample variation: how much does the pooled mean differ across samples?
    price_per_sample = price.mean(axis=1)  # [n]
    vol_per_sample = vol.mean(axis=1)
    print(f"  across-sample std of mean(price): {price_per_sample.std():.4e}")
    print(f"  across-sample std of mean(volume): {vol_per_sample.std():.4e}")

    print(f"\n  label distribution in this batch: "
          f"{np.bincount(y.numpy(), minlength=3).tolist()}")


def test_c_overfit(model, x, y, edge_index, device, steps=300, lr=1e-3):
    print("\n" + "=" * 60)
    print(f"TEST C — single-batch overfit ({len(y)} samples, {steps} steps)")
    print("=" * 60)
    model.train()
    x = x.to(device)
    y = y.to(device)
    ei = edge_index.to(device)

    # Disable dropout to make overfitting easy (probe capacity, not regularization)
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        opt.zero_grad()
        if getattr(model, "conv_type", "") == "gat" or not hasattr(model, "forward_static"):
            out = model(Batch.from_data_list([Data(x=item, edge_index=ei) for item in x]))
        else:
            out = model.forward_static(x, ei)
        loss = F.cross_entropy(out, y)
        loss.backward()

        # gradient norm: is anything flowing?
        gnorm = torch.sqrt(sum(
            (p.grad ** 2).sum() for p in model.parameters() if p.grad is not None
        )).item()
        opt.step()

        if step == 1 or step % 50 == 0:
            acc = (out.argmax(1) == y).float().mean().item()
            pred_dist = torch.bincount(out.argmax(1), minlength=3).tolist()
            print(f"  step {step:4d} | loss {loss.item():.4f} | acc {acc:.3f} "
                  f"| grad_norm {gnorm:.3e} | pred_dist {pred_dist}")

    final_acc = (out.argmax(1) == y).float().mean().item()
    print(f"\n  final train acc on this batch: {final_acc:.3f}")
    if loss.item() < 0.1:
        print("  => Model CAN overfit. No gradient/architecture bug; "
              "issue is signal/generalization.")
    elif gnorm < 1e-6:
        print("  => Gradients ~0. Real bug: gradient flow is broken.")
    else:
        print("  => Loss stuck despite gradients. Likely feature collapse / "
              "oversmoothing or feature-scale problem.")


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    data_cfg = cfg["data"]
    paths = build_processed_paths(data_cfg["processed_dir"])
    num_nodes = 2 * int(data_cfg["n_levels"]) * (int(data_cfg["n_lags"]) + 1)

    model_type = cfg["model"]["type"].lower()
    if is_recurrent_sparse_model(model_type):
        edge_index = torch.empty((2, 0), dtype=torch.long)
        print(f"Graph: {num_nodes} recurrent sparse nodes | edge_index skipped")
    else:
        edge_index = load_tmfg_edge_index(
            data_cfg["adj_matrix_path"],
            n_lags=int(data_cfg["n_lags"]), n_levels=int(data_cfg["n_levels"]),
            cache_path=Path(data_cfg["processed_dir"]) / "edge_index.pt",
        )
        print(f"Graph: {num_nodes} nodes | {edge_index.shape[1]} edges")

    model = build_model(cfg).to(device)
    print(f"Model: {cfg['model']['type'].upper()} | "
          f"{model.count_parameters():,} params")

    bs = int(cfg["training"]["batch_size"])
    x, y = load_real_batch(paths, n=bs, num_nodes=num_nodes, balanced=True)

    test_b_features(x, y)
    if supports_static_batching(model_type):
        test_a_equivalence(model, x, edge_index, device)
    else:
        print("Static equivalence check is not applicable to this architecture.")
    test_c_overfit(model, x, y, edge_index, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn/default.yaml")
    args = parser.parse_args()
    main(args.config)
