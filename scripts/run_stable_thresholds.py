#!/usr/bin/env python3
"""Average and evaluate decision thresholds across walk-forward folds.

This is the Step 5 calibration script. It keeps every trained checkpoint fixed,
reads the per-fold validation-tuned thresholds produced by walk-forward/grid
runs, builds one stable threshold pair (mean or median), and applies that same
pair to each fold test split.

The script is intentionally non-destructive: it does not overwrite checkpoints,
processed tensors, or the per-fold ``thresholds.json`` files. All new artifacts
are written to a separate output directory under ``config/stable_thresholds`` by
default.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from remoma.config import load_config
from remoma.utils.checkpoints import load_checkpoint_state


CORE_METRICS = (
    "accuracy",
    "f1_macro",
    "f1_weighted",
    "signal_precision",
    "signal_recall",
    "signal_f1",
    "signal_rate",
    "flat_to_signal_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fold-results",
        required=True,
        help=(
            "Walk-forward summary JSON/CSV. Accepts either "
            "runs/generated_configs/walk_forward.../summary.json or "
            "runs/generated_configs/hparam_grid.../fold_results.json."
        ),
    )
    parser.add_argument(
        "--combo",
        default=None,
        help="Hyperparameter combo to select when --fold-results contains multiple combos.",
    )
    parser.add_argument("--method", choices=["mean", "median"], default="mean")
    parser.add_argument(
        "--round-to-step",
        type=float,
        default=0.01,
        help="Round stable thresholds to this grid step. Use 0 to disable rounding.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Separate directory for stable threshold outputs.",
    )
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Only save the stable threshold JSON; skip model evaluation.",
    )
    return parser.parse_args()


def _maybe_number(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if stripped == "":
        return value
    try:
        if stripped.isdigit():
            return int(stripped)
        return float(stripped)
    except ValueError:
        return value


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as f:
            rows = [
                {key: _maybe_number(value) for key, value in row.items()}
                for row in csv.DictReader(f)
            ]
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and "rows" in payload:
            payload = payload["rows"]
        if not isinstance(payload, list):
            raise ValueError("--fold-results must contain a list of fold rows.")
        rows = [dict(row) for row in payload]

    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def _select_rows(rows: list[dict[str, Any]], combo: str | None) -> list[dict[str, Any]]:
    combos = sorted({str(row["combo"]) for row in rows if row.get("combo") is not None})
    if combo is None and len(combos) > 1:
        raise ValueError(
            "Fold results contain multiple combos. Pass --combo. "
            f"Available combos: {', '.join(combos)}"
        )
    if combo is not None:
        rows = [row for row in rows if str(row.get("combo")) == combo]
        if not rows:
            raise ValueError(f"No rows found for combo={combo!r}")

    required = {"config", "t_down", "t_up"}
    missing = [row for row in rows if not required.issubset(row)]
    if missing:
        raise ValueError("Each selected row must contain config, t_down, and t_up.")

    return sorted(rows, key=lambda row: int(row.get("fold", 0)))


def _aggregate(values: list[float], method: str) -> float:
    if method == "mean":
        return float(statistics.fmean(values))
    if method == "median":
        return float(statistics.median(values))
    raise ValueError(f"Unsupported method: {method}")


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return float(round(round(value / step) * step, 6))


def _default_output_dir(source: Path, combo: str | None, method: str) -> Path:
    base = source.parent.name if source.name in {"summary.json", "fold_results.json"} else source.stem
    if combo:
        base = f"{base}_{combo}"
    return Path("runs/generated_configs/stable_thresholds") / f"{base}_{method}"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"folds": len(rows), "mean": {}, "std": {}}
    prefixes = ("val_stable", "test_stable", "test_fold_tuned")
    for prefix in prefixes:
        for metric in CORE_METRICS:
            key = f"{prefix}_{metric}"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                summary["mean"][key] = _mean(values)
                summary["std"][key] = _stdev(values)
    return summary


def _build_loaders(cfg: dict[str, Any], device: torch.device):
    import torch
    from torch.utils.data import DataLoader

    from remoma.dataset.lob_dataset import FastLOBDataset, StaticLOBTensorDataset
    from remoma.dataset.preprocessing import (
        build_processed_paths,
        has_compatible_processed_dataset,
        preprocess_to_disk,
    )
    from remoma.graph.adjacency import load_tmfg_edge_index
    from remoma.models import is_recurrent_sparse_model

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

    def make_loader(x_path: Path, y_path: Path):
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

    return (
        make_loader(paths["X_val"], paths["y_val"]),
        make_loader(paths["X_test"], paths["y_test"]),
        static_edge,
        static_mode,
    )


def _predict_probs(model, loader, edge_index, device: torch.device, static_mode: bool):
    import numpy as np
    import torch

    model.eval()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []

    with torch.no_grad():
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


def _evaluate_fold(
    row: dict[str, Any],
    stable_t_down: float,
    stable_t_up: float,
    checkpoint_name: str,
    device: torch.device,
) -> dict[str, Any]:
    import torch
    import yaml

    from remoma.models import build_model
    from remoma.training.threshold import apply_thresholds, decision_metrics

    config_path = Path(str(row["config"]))
    cfg = load_config(config_path)

    checkpoint = Path(cfg["paths"]["checkpoints"]) / checkpoint_name
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    val_loader, test_loader, edge_index, static_mode = _build_loaders(cfg, device)
    model = build_model(cfg).to(device)
    load_checkpoint_state(model, checkpoint, cfg, device)

    print(f"Evaluating fold {row.get('fold', '?')} | {checkpoint}")
    val_probs, val_labels = _predict_probs(model, val_loader, edge_index, device, static_mode)
    test_probs, test_labels = _predict_probs(model, test_loader, edge_index, device, static_mode)

    val_stable = apply_thresholds(val_probs, stable_t_down, stable_t_up)
    test_stable = apply_thresholds(test_probs, stable_t_down, stable_t_up)

    result: dict[str, Any] = {
        "fold": row.get("fold"),
        "combo": row.get("combo"),
        "config": str(config_path),
        "checkpoint": str(checkpoint),
        "source_t_down": float(row["t_down"]),
        "source_t_up": float(row["t_up"]),
        "stable_t_down": stable_t_down,
        "stable_t_up": stable_t_up,
    }
    result.update(_prefixed("val_argmax", decision_metrics(val_probs.argmax(1), val_labels)))
    result.update(_prefixed("val_stable", decision_metrics(val_stable, val_labels)))
    result.update(_prefixed("test_argmax", decision_metrics(test_probs.argmax(1), test_labels)))
    result.update(_prefixed("test_stable", decision_metrics(test_stable, test_labels)))

    legacy_metric_map = {
        "test_accuracy": "accuracy",
        "test_macro_f1": "f1_macro",
        "test_weighted_f1": "f1_weighted",
        "test_signal_precision": "signal_precision",
        "test_signal_recall": "signal_recall",
        "test_signal_f1": "signal_f1",
        "test_flat_to_signal": "flat_to_signal_rate",
    }
    for old_key, metric in legacy_metric_map.items():
        if old_key in row:
            result[f"test_fold_tuned_{metric}"] = float(row[old_key])

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    print("Retrospective cross-fold diagnostic: aggregated thresholds may use later calibration periods.")
    source = Path(args.fold_results)
    rows = _select_rows(_load_rows(source), args.combo)

    source_down = [float(row["t_down"]) for row in rows]
    source_up = [float(row["t_up"]) for row in rows]
    stable_t_down = _round_step(_aggregate(source_down, args.method), args.round_to_step)
    stable_t_up = _round_step(_aggregate(source_up, args.method), args.round_to_step)

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(source, args.combo, args.method)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_payload = {
        "evaluation_scope": "retrospective_cross_fold_diagnostic",
        "source": str(source),
        "combo": args.combo,
        "method": args.method,
        "round_to_step": args.round_to_step,
        "thresholds": {"down": stable_t_down, "up": stable_t_up},
        "fold_thresholds": [
            {
                "fold": row.get("fold"),
                "combo": row.get("combo"),
                "config": row.get("config"),
                "down": float(row["t_down"]),
                "up": float(row["t_up"]),
            }
            for row in rows
        ],
    }
    (output_dir / "stable_thresholds.json").write_text(
        json.dumps(threshold_payload, indent=2),
        encoding="utf-8",
    )

    print("\nStable threshold calibration")
    print(f"Source: {source}")
    if args.combo:
        print(f"Combo: {args.combo}")
    print(f"Method: {args.method}")
    print(f"Fold thresholds down={source_down} up={source_up}")
    print(f"Stable thresholds: down>={stable_t_down:.2f} up>={stable_t_up:.2f}")
    print(f"Saved: {output_dir / 'stable_thresholds.json'}")

    if args.no_evaluate:
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    eval_rows = [
        _evaluate_fold(row, stable_t_down, stable_t_up, args.checkpoint_name, device)
        for row in rows
    ]
    summary = _summarize(eval_rows)

    (output_dir / "fold_results.json").write_text(
        json.dumps(eval_rows, indent=2),
        encoding="utf-8",
    )
    _write_csv(output_dir / "fold_results.csv", eval_rows)

    full_summary = threshold_payload | {"evaluation": summary}
    (output_dir / "summary.json").write_text(
        json.dumps(full_summary, indent=2),
        encoding="utf-8",
    )

    mean_metrics = summary["mean"]
    print("\nStable threshold evaluation mean")
    print(
        "TEST stable  "
        f"acc={mean_metrics.get('test_stable_accuracy', 0.0):.4f}  "
        f"macro_f1={mean_metrics.get('test_stable_f1_macro', 0.0):.4f}  "
        f"weighted_f1={mean_metrics.get('test_stable_f1_weighted', 0.0):.4f}  "
        f"signal_prec={mean_metrics.get('test_stable_signal_precision', 0.0):.4f}  "
        f"signal_rec={mean_metrics.get('test_stable_signal_recall', 0.0):.4f}  "
        f"flat_to_signal={mean_metrics.get('test_stable_flat_to_signal_rate', 0.0):.4f}"
    )
    if "test_fold_tuned_f1_macro" in mean_metrics:
        print(
            "FOLD tuned   "
            f"acc={mean_metrics.get('test_fold_tuned_accuracy', 0.0):.4f}  "
            f"macro_f1={mean_metrics.get('test_fold_tuned_f1_macro', 0.0):.4f}  "
            f"weighted_f1={mean_metrics.get('test_fold_tuned_f1_weighted', 0.0):.4f}  "
            f"signal_prec={mean_metrics.get('test_fold_tuned_signal_precision', 0.0):.4f}  "
            f"signal_rec={mean_metrics.get('test_fold_tuned_signal_recall', 0.0):.4f}  "
            f"flat_to_signal={mean_metrics.get('test_fold_tuned_flat_to_signal_rate', 0.0):.4f}"
        )
    print(f"\nSaved stable-threshold fold results: {output_dir / 'fold_results.csv'}")
    print(f"Saved stable-threshold summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
