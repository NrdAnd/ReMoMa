#!/usr/bin/env python3
"""Run a small walk-forward hyperparameter grid for RecurrentSparseSTHNN.

Each hyperparameter combination gets its own base config, processed directory,
checkpoint directory, walk-forward fold configs, and result summaries. This
keeps grid runs separate from the current best checkpoints.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--name", default="hparam_grid")
    parser.add_argument("--config-dir", default="config/hparam_grid")
    parser.add_argument("--walk-forward-config-dir", default="config/walk_forward_grid")
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[8, 16])
    parser.add_argument("--dropouts", nargs="+", type=float, default=[0.15, 0.20, 0.25])
    parser.add_argument("--message-iterations", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--max-combos", type=int, default=None)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--min-train-files", type=int, default=2)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument("--objective", default="macro_f1_penalized")
    parser.add_argument("--flat-fp-penalty", type=float, default=0.25)
    parser.add_argument("--lo", type=float, default=0.33)
    parser.add_argument("--hi", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.01)
    return parser.parse_args()


def _dropout_tag(value: float) -> str:
    return f"{int(round(value * 100)):02d}"


def _combo_tag(hidden_dim: int, dropout: float, message_iterations: int) -> str:
    return f"hd{hidden_dim}_do{_dropout_tag(dropout)}_mi{message_iterations}"


def _with_combo(cfg: dict, tag: str, hidden_dim: int, dropout: float, message_iterations: int) -> dict:
    out = copy.deepcopy(cfg)
    out["model"]["hidden_dim"] = int(hidden_dim)
    out["model"]["readout_dropout"] = float(dropout)
    out["model"]["message_iterations"] = int(message_iterations)

    processed_dir = Path(out["data"]["processed_dir"])
    out["data"]["processed_dir"] = str(processed_dir.with_name(f"{processed_dir.name}_{tag}"))

    checkpoint_dir = Path(out["paths"]["checkpoints"])
    out["paths"]["checkpoints"] = str(checkpoint_dir.with_name(f"{checkpoint_dir.name.rstrip('/')}_{tag}"))
    return out


def _run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _read_rows(path: Path, combo: dict) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        row.update(combo)
    return rows


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _combo_summary(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["combo"], []).append(row)

    summaries = []
    for combo, combo_rows in sorted(grouped.items()):
        first = combo_rows[0]
        summaries.append(
            {
                "combo": combo,
                "hidden_dim": first["hidden_dim"],
                "readout_dropout": first["readout_dropout"],
                "message_iterations": first["message_iterations"],
                "folds": len(combo_rows),
                "mean_test_accuracy": _mean([r["test_accuracy"] for r in combo_rows]),
                "mean_test_macro_f1": _mean([r["test_macro_f1"] for r in combo_rows]),
                "mean_test_weighted_f1": _mean([r["test_weighted_f1"] for r in combo_rows]),
                "mean_test_signal_precision": _mean(
                    [r["test_signal_precision"] for r in combo_rows]
                ),
                "mean_test_signal_recall": _mean(
                    [r["test_signal_recall"] for r in combo_rows]
                ),
                "mean_test_signal_f1": _mean([r["test_signal_f1"] for r in combo_rows]),
                "mean_test_flat_to_signal": _mean(
                    [r["test_flat_to_signal"] for r in combo_rows]
                ),
            }
        )
    summaries.sort(
        key=lambda row: (
            row["mean_test_macro_f1"],
            row["mean_test_signal_precision"],
            -row["mean_test_flat_to_signal"],
        ),
        reverse=True,
    )
    return summaries


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_summaries(root: Path, rows: list[dict]) -> None:
    if not rows:
        return
    root.mkdir(parents=True, exist_ok=True)
    combo_rows = _combo_summary(rows)

    (root / "fold_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (root / "combo_summary.json").write_text(
        json.dumps(combo_rows, indent=2),
        encoding="utf-8",
    )
    _write_csv(root / "fold_results.csv", rows)
    _write_csv(root / "combo_summary.csv", combo_rows)
    print(f"\nSaved grid fold results: {root / 'fold_results.csv'}")
    print(f"Saved grid combo summary: {root / 'combo_summary.csv'}")


def main() -> None:
    args = parse_args()
    base_config = Path(args.base_config)
    with base_config.open() as f:
        base_cfg = yaml.safe_load(f)

    grid_root = Path(args.config_dir) / args.name
    grid_root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    combo_count = 0
    for hidden_dim in args.hidden_dims:
        for dropout in args.dropouts:
            for message_iterations in args.message_iterations:
                combo_count += 1
                if args.max_combos is not None and combo_count > args.max_combos:
                    break

                tag = _combo_tag(hidden_dim, dropout, message_iterations)
                combo_name = f"{args.name}_{tag}"
                combo = {
                    "combo": tag,
                    "hidden_dim": int(hidden_dim),
                    "readout_dropout": float(dropout),
                    "message_iterations": int(message_iterations),
                }
                combo_cfg = _with_combo(
                    base_cfg,
                    tag,
                    hidden_dim,
                    dropout,
                    message_iterations,
                )
                combo_config_path = grid_root / f"{tag}.yaml"
                combo_config_path.write_text(
                    yaml.safe_dump(combo_cfg, sort_keys=False),
                    encoding="utf-8",
                )

                summary_path = (
                    Path(args.walk_forward_config_dir)
                    / combo_name
                    / "summary.json"
                )
                cmd = [
                    sys.executable,
                    "scripts/run_walk_forward.py",
                    "--base-config",
                    str(combo_config_path),
                    "--name",
                    combo_name,
                    "--config-dir",
                    args.walk_forward_config_dir,
                    "--min-train-files",
                    str(args.min_train_files),
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
                ]
                if args.max_folds is not None:
                    cmd.extend(["--max-folds", str(args.max_folds)])
                if args.run:
                    cmd.append("--run")

                print(
                    f"\nCombo {tag}: hidden_dim={hidden_dim}, "
                    f"dropout={dropout}, message_iterations={message_iterations}"
                )
                print(f"Config: {combo_config_path}")

                if args.skip_existing and summary_path.exists():
                    print(f"Skipping existing summary: {summary_path}")
                elif args.run:
                    _run(cmd)
                else:
                    print("Command:")
                    print("  " + " ".join(cmd))

                if summary_path.exists():
                    rows.extend(_read_rows(summary_path, combo))

            if args.max_combos is not None and combo_count >= args.max_combos:
                break
        if args.max_combos is not None and combo_count >= args.max_combos:
            break

    _write_summaries(grid_root, rows)


if __name__ == "__main__":
    main()
