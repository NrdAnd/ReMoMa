"""Shared configuration loading and repository-relative path resolution."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
# Editable/source installs know the checkout. Wheel users can select its root
# explicitly; otherwise paths resolve from the caller's working directory.
PROJECT_ROOT = Path(os.environ.get(
    "REMOMA_ROOT", str(_SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd())
)).expanduser().resolve()


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a complete YAML configuration; configured paths are relative to root."""
    with project_path(path).open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    for section in ("data", "model", "training", "paths"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Configuration requires a '{section}' mapping.")
    for key in ("raw_dir", "message_dir", "adj_matrix_path", "processed_dir"):
        if cfg["data"].get(key) is not None:
            cfg["data"][key] = str(project_path(cfg["data"][key]))
    for key, value in cfg["paths"].items():
        if value is not None:
            cfg["paths"][key] = str(project_path(value))
    model_sections = [cfg["model"], *(cfg["model"].get("overrides") or {}).values()]
    for section in model_sections:
        if section.get("adjacency_path"):
            section["adjacency_path"] = str(project_path(section["adjacency_path"]))
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict[str, Any]) -> None:
    """Reject unsupported tensor dimensions and invalid sampling parameters early."""
    data, training = cfg["data"], cfg["training"]
    if int(data["n_levels"]) != 10:
        raise ValueError("The LOBSTER loader requires data.n_levels=10 (40 columns).")
    for key, minimum in (("n_lags", 0), ("prediction_horizon", 1), ("n_volume_bins", 2)):
        if int(data[key]) < minimum:
            raise ValueError(f"data.{key} must be >= {minimum}.")
    if float(data["threshold"]) < 0:
        raise ValueError("data.threshold must be nonnegative.")
    if data.get("feature_dtype", "float16") not in {"float16", "float32"}:
        raise ValueError("data.feature_dtype must be float16 or float32.")
    for key in ("max_samples_per_class", "max_val_samples", "max_test_samples"):
        if data.get(key) is not None and int(data[key]) < 1:
            raise ValueError(f"data.{key} must be positive or null.")
    if data["split_strategy"] == "by_lag":
        train, val = float(data["train_ratio"]), float(data["val_ratio"])
        if not (0 < train < 1 and 0 < val < 1 and train + val < 1):
            raise ValueError("Split ratios must be positive and sum to less than one.")
    elif data["split_strategy"] != "by_file":
        raise ValueError("data.split_strategy must be by_file or by_lag.")
    for key in ("batch_size", "epochs", "early_stopping_patience"):
        if int(training[key]) < 1:
            raise ValueError(f"training.{key} must be positive.")
    if int(training.get("num_workers", 0)) < 0:
        raise ValueError("training.num_workers must be nonnegative.")
