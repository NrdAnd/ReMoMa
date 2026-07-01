#!/usr/bin/env python3
"""Generate and optionally run expanding walk-forward experiments.

The script keeps a base config intact and writes one by-file config per fold.
With five Cisco days and ``--min-train-files 2`` the default folds are:

  fold_01: train=[0,1],   val=[2], test=[3]
  fold_02: train=[0,1,2], val=[3], test=[4]

Thresholds are tuned independently on each fold validation split and applied to
that fold test split. The saved JSON files are then aggregated into a CSV/JSON
summary.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.io import discover_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--name", default=None, help="Experiment name for output folders.")
    parser.add_argument("--config-dir", default="config/walk_forward")
    parser.add_argument("--min-train-files", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--run", action="store_true", help="Run preprocess/train/tune for each fold.")
    parser.add_argument("--evaluate", action="store_true", help="Also run scripts/evaluate.py per fold.")
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--objective", default="macro_f1_penalized")
    parser.add_argument("--flat-fp-penalty", type=float, default=0.25)
    parser.add_argument("--lo", type=float, default=0.33)
    parser.add_argument("--hi", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def _experiment_name(base_config: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = base_config.stem
    return name.removeprefix("recurrent_sparse_sthnn_")


def _folds(n_files: int, min_train_files: int, max_folds: int | None) -> list[dict]:
    if min_train_files < 1:
        raise ValueError("--min-train-files must be >= 1")
    folds = []
    fold_id = 1
    for test_file in range(min_train_files + 1, n_files):
        folds.append(
            {
                "fold": fold_id,
                "train_files": list(range(0, test_file - 1)),
                "val_files": [test_file - 1],
                "test_files": [test_file],
            }
        )
        fold_id += 1
        if max_folds is not None and len(folds) >= max_folds:
            break
    if not folds:
        raise ValueError(
            f"Need at least {min_train_files + 2} raw files for walk-forward; "
            f"found {n_files}."
        )
    return folds


def _with_fold_paths(cfg: dict, name: str, fold: dict) -> dict:
    out = copy.deepcopy(cfg)
    fold_name = f"fold_{fold['fold']:02d}"

    out["data"]["split_strategy"] = "by_file"
    out["data"]["train_files"] = fold["train_files"]
    out["data"]["val_files"] = fold["val_files"]
    out["data"]["test_files"] = fold["test_files"]

    base_processed = Path(out["data"]["processed_dir"])
    out["data"]["processed_dir"] = str(base_processed.with_name(f"{base_processed.name}_{fold_name}"))

    checkpoints = Path(out["paths"]["checkpoints"])
    out["paths"]["checkpoints"] = str(checkpoints.with_name(f"{checkpoints.name.rstrip('/')}_{fold_name}"))
    return out


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _threshold_payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_row(fold: dict, config_path: Path, threshold_path: Path) -> dict:
    payload = _threshold_payload(threshold_path)
    test = payload["test_tuned_metrics"]
    val = payload["validation_tuned_metrics"]
    thresholds = payload["thresholds"]
    return {
        "fold": fold["fold"],
        "train_files": " ".join(map(str, fold["train_files"])),
        "val_files": " ".join(map(str, fold["val_files"])),
        "test_files": " ".join(map(str, fold["test_files"])),
        "config": str(config_path),
        "t_down": thresholds["down"],
        "t_up": thresholds["up"],
        "val_macro_f1": val["f1_macro"],
        "val_signal_precision": val["signal_precision"],
        "val_signal_recall": val["signal_recall"],
        "val_flat_to_signal": val["flat_to_signal_rate"],
        "test_accuracy": test["accuracy"],
        "test_macro_f1": test["f1_macro"],
        "test_weighted_f1": test["f1_weighted"],
        "test_signal_precision": test["signal_precision"],
        "test_signal_recall": test["signal_recall"],
        "test_signal_f1": test["signal_f1"],
        "test_flat_to_signal": test["flat_to_signal_rate"],
    }


def _write_summary(rows: list[dict], root: Path) -> None:
    if not rows:
        return
    csv_path = root / "summary.csv"
    json_path = root / "summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved walk-forward summary: {csv_path}")
    print(f"Saved walk-forward summary: {json_path}")


def main() -> None:
    args = parse_args()
    base_config = Path(args.base_config)
    with base_config.open() as f:
        cfg = yaml.safe_load(f)

    name = _experiment_name(base_config, args.name)
    config_root = Path(args.config_dir) / name
    config_root.mkdir(parents=True, exist_ok=True)

    raw_files = discover_files(cfg["data"]["raw_dir"])
    folds = _folds(len(raw_files), args.min_train_files, args.max_folds)

    print(f"Experiment: {name}")
    print(f"Raw files: {len(raw_files)}")
    print(f"Folds: {len(folds)}")

    rows = []
    for fold in folds:
        fold_name = f"fold_{fold['fold']:02d}"
        fold_cfg = _with_fold_paths(cfg, name, fold)
        config_path = config_root / f"{fold_name}.yaml"
        config_path.write_text(yaml.safe_dump(fold_cfg, sort_keys=False), encoding="utf-8")

        checkpoint_dir = Path(fold_cfg["paths"]["checkpoints"])
        checkpoint = checkpoint_dir / "best.pt"
        thresholds = checkpoint_dir / "thresholds.json"

        print(
            f"\n{fold_name}: train={fold['train_files']} "
            f"val={fold['val_files']} test={fold['test_files']}"
        )
        print(f"Config: {config_path}")

        preprocess_cmd = [
            sys.executable,
            "scripts/preprocess_dataset.py",
            "--config",
            str(config_path),
        ]
        if args.force_preprocess:
            preprocess_cmd.append("--force")

        train_cmd = [sys.executable, "scripts/train.py", "--config", str(config_path)]
        tune_cmd = [
            sys.executable,
            "scripts/tune_threshold.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--objective",
            args.objective,
            "--flat-fp-penalty",
            str(args.flat_fp_penalty),
            "--lo",
            str(args.lo),
            "--hi",
            str(args.hi),
            "--step",
            str(args.step),
            "--save-thresholds",
            str(thresholds),
        ]
        eval_cmd = [
            sys.executable,
            "scripts/evaluate.py",
            "--checkpoint",
            str(checkpoint),
            "--config",
            str(config_path),
            "--thresholds",
            str(thresholds),
        ]

        if args.run:
            _run(preprocess_cmd)
            _run(train_cmd)
            _run(tune_cmd)
            if args.evaluate:
                _run(eval_cmd)
            rows.append(_summary_row(fold, config_path, thresholds))
        else:
            print("Commands:")
            print("  " + " ".join(preprocess_cmd))
            print("  " + " ".join(train_cmd))
            print("  " + " ".join(tune_cmd))
            if args.evaluate:
                print("  " + " ".join(eval_cmd))

    if rows:
        _write_summary(rows, config_root)


if __name__ == "__main__":
    main()
