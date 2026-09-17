#!/usr/bin/env python3
"""Run a stable seed ensemble over selected walk-forward folds.

This is the Step 6 script. It ensembles only models that belong to the same
walk-forward fold, avoiding temporal leakage between folds. For each selected
fold it can:

1. train extra seeds into new checkpoint directories;
2. average member probabilities on that fold validation/test split;
3. tune fold-specific ensemble thresholds on validation;
4. average/median those fold thresholds into one stable threshold pair;
5. apply the stable pair to every fold test split.

The script is non-destructive. Seed checkpoints are written to directories with
``_seed{seed}`` suffixes, and all ensemble reports are written under
``config/ensembles`` by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

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
        help="Walk-forward/grid fold results JSON or CSV.",
    )
    parser.add_argument(
        "--combo",
        default=None,
        help="Hyperparameter combo to select when fold results contain multiple combos.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[123, 777])
    parser.add_argument(
        "--include-base",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include the original fold checkpoint as the first ensemble member.",
    )
    parser.add_argument(
        "--run-train",
        action="store_true",
        help="Train missing seed checkpoints before ensemble evaluation.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip seed training when that seed checkpoint already exists.",
    )
    parser.add_argument("--checkpoint-name", default="best.pt")
    parser.add_argument("--name", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--stable-method",
        choices=["mean", "median"],
        default="mean",
        help="How to aggregate fold-tuned ensemble thresholds.",
    )
    parser.add_argument("--round-to-step", type=float, default=0.01)
    parser.add_argument(
        "--objective",
        choices=["macro_f1", "macro_f1_penalized", "signal_f1", "signal_precision"],
        default="macro_f1_penalized",
    )
    parser.add_argument("--flat-fp-penalty", type=float, default=0.25)
    parser.add_argument("--lo", type=float, default=0.33)
    parser.add_argument("--hi", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--min-signal-precision", type=float, default=None)
    parser.add_argument("--min-signal-recall", type=float, default=None)
    parser.add_argument("--max-signal-rate", type=float, default=None)
    parser.add_argument("--max-flat-to-signal-rate", type=float, default=None)
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

    if any("config" not in row for row in rows):
        raise ValueError("Each selected fold row must contain a config path.")
    return sorted(rows, key=lambda row: int(row.get("fold", 0)))


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    return load_config(path)


def _seed_checkpoint_dir(base_checkpoint_dir: Path, seed: int) -> Path:
    return base_checkpoint_dir.with_name(f"{base_checkpoint_dir.name.rstrip('/')}_seed{seed}")


def _checkpoint_paths(
    fold_cfg: dict[str, Any],
    seeds: list[int],
    include_base: bool,
    checkpoint_name: str,
) -> tuple[list[Path], list[Path]]:
    base_dir = Path(fold_cfg["paths"]["checkpoints"])
    member_paths: list[Path] = []
    train_dirs: list[Path] = []

    if include_base:
        member_paths.append(base_dir / checkpoint_name)

    for seed in seeds:
        seed_dir = _seed_checkpoint_dir(base_dir, seed)
        train_dirs.append(seed_dir)
        member_paths.append(seed_dir / checkpoint_name)

    return member_paths, train_dirs


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _train_seed_members(
    rows: list[dict[str, Any]],
    seeds: list[int],
    checkpoint_name: str,
    skip_existing: bool,
) -> None:
    for row in rows:
        config_path = Path(str(row["config"]))
        fold_cfg = _read_yaml(config_path)
        base_dir = Path(fold_cfg["paths"]["checkpoints"])

        for seed in seeds:
            seed_dir = _seed_checkpoint_dir(base_dir, seed)
            checkpoint = seed_dir / checkpoint_name
            if skip_existing and checkpoint.exists():
                print(f"Skipping existing seed checkpoint: {checkpoint}")
                continue

            _run(
                [
                    sys.executable,
                    "scripts/train.py",
                    "--config",
                    str(config_path),
                    "--seed",
                    str(seed),
                    "--checkpoint-dir",
                    str(seed_dir),
                ]
            )


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


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _stdev(values: list[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def _prefixed(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"folds": len(rows), "mean": {}, "std": {}}
    prefixes = (
        "test_single_fold_tuned",
        "test_ensemble_argmax",
        "test_ensemble_fold_tuned",
        "test_ensemble_stable",
    )
    for prefix in prefixes:
        for metric in CORE_METRICS:
            key = f"{prefix}_{metric}"
            values = [float(row[key]) for row in rows if key in row]
            if values:
                summary["mean"][key] = _mean(values)
                summary["std"][key] = _stdev(values)
    return summary


def _build_loaders(cfg: dict[str, Any], device):
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


def _predict_probs(model, loader, edge_index, device, static_mode: bool):
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


def _load_member_probs(cfg, checkpoint: Path, val_loader, test_loader, edge_index, device, static_mode: bool):
    import torch

    from remoma.models import build_model

    model = build_model(cfg).to(device)
    load_checkpoint_state(model, checkpoint, cfg, device)

    val_probs, val_labels = _predict_probs(model, val_loader, edge_index, device, static_mode)
    test_probs, test_labels = _predict_probs(model, test_loader, edge_index, device, static_mode)

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return val_probs, val_labels, test_probs, test_labels


def _evaluate_fold(row: dict[str, Any], args: argparse.Namespace, device) -> dict[str, Any]:
    import numpy as np

    from remoma.training.threshold import apply_thresholds, decision_metrics, search_thresholds

    config_path = Path(str(row["config"]))
    cfg = _read_yaml(config_path)
    checkpoint_paths, _ = _checkpoint_paths(
        cfg,
        seeds=args.seeds,
        include_base=args.include_base,
        checkpoint_name=args.checkpoint_name,
    )
    missing = [path for path in checkpoint_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing ensemble checkpoints. Train them with --run-train first: "
            + ", ".join(map(str, missing))
        )

    print(f"\nEvaluating fold {row.get('fold', '?')} ensemble")
    print(f"Config: {config_path}")
    print(f"Members: {len(checkpoint_paths)}")

    val_loader, test_loader, edge_index, static_mode = _build_loaders(cfg, device)
    val_sum = test_sum = None
    val_labels_ref = test_labels_ref = None
    member_rows: list[dict[str, Any]] = []

    for idx, checkpoint in enumerate(checkpoint_paths, start=1):
        print(f"  [{idx}/{len(checkpoint_paths)}] {checkpoint}")
        val_probs, val_labels, test_probs, test_labels = _load_member_probs(
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
        member_rows.append(
            {
                "checkpoint": str(checkpoint),
                "test_argmax_metrics": decision_metrics(test_probs.argmax(1), test_labels),
            }
        )

    assert val_sum is not None and test_sum is not None
    assert val_labels_ref is not None and test_labels_ref is not None

    val_probs = val_sum / len(checkpoint_paths)
    test_probs = test_sum / len(checkpoint_paths)
    val_labels = val_labels_ref
    test_labels = test_labels_ref

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
    fold_t_down = float(search["t_down"])
    fold_t_up = float(search["t_up"])
    fold_tuned_test = apply_thresholds(test_probs, fold_t_down, fold_t_up)

    result: dict[str, Any] = {
        "fold": row.get("fold"),
        "combo": row.get("combo"),
        "config": str(config_path),
        "members": [str(path) for path in checkpoint_paths],
        "member_metrics": member_rows,
        "ensemble_fold_t_down": fold_t_down,
        "ensemble_fold_t_up": fold_t_up,
        "ensemble_fold_validation_score": float(search["score"]),
        "_val_probs": val_probs,
        "_val_labels": val_labels,
        "_test_probs": test_probs,
        "_test_labels": test_labels,
    }
    result.update(_prefixed("val_ensemble_argmax", decision_metrics(val_probs.argmax(1), val_labels)))
    result.update(_prefixed("test_ensemble_argmax", decision_metrics(test_probs.argmax(1), test_labels)))
    result.update(_prefixed("val_ensemble_fold_tuned", search["metrics"]))
    result.update(_prefixed("test_ensemble_fold_tuned", decision_metrics(fold_tuned_test, test_labels)))

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
            result[f"test_single_fold_tuned_{metric}"] = float(row[old_key])

    return result


def _apply_stable_thresholds(rows: list[dict[str, Any]], t_down: float, t_up: float) -> None:
    from remoma.training.threshold import apply_thresholds, decision_metrics

    for row in rows:
        val_preds = apply_thresholds(row["_val_probs"], t_down, t_up)
        test_preds = apply_thresholds(row["_test_probs"], t_down, t_up)
        row["ensemble_stable_t_down"] = t_down
        row["ensemble_stable_t_up"] = t_up
        row.update(
            _prefixed(
                "val_ensemble_stable",
                decision_metrics(val_preds, row["_val_labels"]),
            )
        )
        row.update(
            _prefixed(
                "test_ensemble_stable",
                decision_metrics(test_preds, row["_test_labels"]),
            )
        )


def _json_ready_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned = []
    for row in rows:
        cleaned.append({key: value for key, value in row.items() if not key.startswith("_")})
    return cleaned


def _default_name(source: Path, combo: str | None, seeds: list[int]) -> str:
    source_name = source.parent.name if source.name == "fold_results.json" else source.stem
    seed_tag = "base" if not seeds else "base_seed" + "_".join(map(str, seeds))
    if combo:
        return f"{source_name}_{combo}_{seed_tag}"
    return f"{source_name}_{seed_tag}"


def main() -> None:
    args = parse_args()
    print("Retrospective cross-fold diagnostic: aggregated thresholds may use later calibration periods.")
    source = Path(args.fold_results)
    rows = _select_rows(_load_rows(source), args.combo)

    if not args.include_base and not args.seeds:
        raise ValueError("At least one ensemble member is required.")

    name = args.name or _default_name(source, args.combo, args.seeds)
    output_dir = Path(args.output_dir) if args.output_dir else Path("runs/generated_configs/ensembles") / name
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": str(source),
        "combo": args.combo,
        "name": name,
        "output_dir": str(output_dir),
        "seeds": args.seeds,
        "include_base": args.include_base,
        "objective": args.objective,
        "flat_fp_penalty": args.flat_fp_penalty,
        "stable_method": args.stable_method,
        "round_to_step": args.round_to_step,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Selected folds: {len(rows)}")
    print(f"Combo: {args.combo}")
    print(f"Output: {output_dir}")
    print(f"Include base checkpoint: {args.include_base}")
    print(f"Extra seeds: {args.seeds}")

    if args.run_train:
        _train_seed_members(rows, args.seeds, args.checkpoint_name, args.skip_existing)

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    eval_rows = [_evaluate_fold(row, args, device) for row in rows]
    fold_down = [float(row["ensemble_fold_t_down"]) for row in eval_rows]
    fold_up = [float(row["ensemble_fold_t_up"]) for row in eval_rows]
    stable_t_down = _round_step(_aggregate(fold_down, args.stable_method), args.round_to_step)
    stable_t_up = _round_step(_aggregate(fold_up, args.stable_method), args.round_to_step)

    _apply_stable_thresholds(eval_rows, stable_t_down, stable_t_up)
    public_rows = _json_ready_rows(eval_rows)
    summary = _summarize(public_rows)
    payload = metadata | {
        "fold_ensemble_thresholds": [
            {
                "fold": row["fold"],
                "down": row["ensemble_fold_t_down"],
                "up": row["ensemble_fold_t_up"],
            }
            for row in public_rows
        ],
        "stable_thresholds": {"down": stable_t_down, "up": stable_t_up},
        "evaluation": summary,
    }

    (output_dir / "fold_results.json").write_text(json.dumps(public_rows, indent=2), encoding="utf-8")
    _write_csv(output_dir / "fold_results.csv", public_rows)
    (output_dir / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    mean_metrics = summary["mean"]
    print("\nStable ensemble summary")
    print(f"Fold ensemble thresholds down={fold_down} up={fold_up}")
    print(f"Stable ensemble thresholds: down>={stable_t_down:.2f} up>={stable_t_up:.2f}")
    if "test_single_fold_tuned_f1_macro" in mean_metrics:
        print(
            "Single model "
            f"acc={mean_metrics.get('test_single_fold_tuned_accuracy', 0.0):.4f}  "
            f"macro_f1={mean_metrics.get('test_single_fold_tuned_f1_macro', 0.0):.4f}  "
            f"signal_prec={mean_metrics.get('test_single_fold_tuned_signal_precision', 0.0):.4f}  "
            f"signal_rec={mean_metrics.get('test_single_fold_tuned_signal_recall', 0.0):.4f}  "
            f"flat_to_signal={mean_metrics.get('test_single_fold_tuned_flat_to_signal_rate', 0.0):.4f}"
        )
    print(
        "Ens stable  "
        f"acc={mean_metrics.get('test_ensemble_stable_accuracy', 0.0):.4f}  "
        f"macro_f1={mean_metrics.get('test_ensemble_stable_f1_macro', 0.0):.4f}  "
        f"signal_prec={mean_metrics.get('test_ensemble_stable_signal_precision', 0.0):.4f}  "
        f"signal_rec={mean_metrics.get('test_ensemble_stable_signal_recall', 0.0):.4f}  "
        f"flat_to_signal={mean_metrics.get('test_ensemble_stable_flat_to_signal_rate', 0.0):.4f}"
    )
    print(f"\nSaved ensemble fold results: {output_dir / 'fold_results.csv'}")
    print(f"Saved ensemble summary: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
