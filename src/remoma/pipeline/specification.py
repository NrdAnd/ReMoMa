"""Strict pipeline settings and chronological, date-based experimental plans."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import re

import yaml

from remoma.config import load_config, project_path, validate_config
from remoma.graph.nmi import NMISettings
from remoma.models import model_family


def _keys(mapping, allowed, name):
    if not isinstance(mapping, dict) or set(mapping) - set(allowed):
        raise ValueError(f"Invalid {name} settings; accepted keys: {', '.join(sorted(allowed))}.")


def validate_spec(spec):
    _keys(spec, {"version", "raw_dir", "message_dir", "symbol", "output_dir", "cache_dir", "mode",
                 "by_file", "walk_forward", "models", "model_configs", "seeds", "nmi", "graph",
                 "data_overrides", "training_overrides"}, "pipeline")
    if spec.get("version") != 1 or spec.get("mode") not in {"by_file", "walk_forward"}:
        raise ValueError("Require pipeline version: 1 and mode: by_file or walk_forward.")
    if not re.fullmatch(r"[A-Z0-9.-]+", spec.get("symbol", "")):
        raise ValueError("Set a single explicit ticker symbol, for example CSCO.")
    for name in ("raw_dir", "output_dir", "cache_dir", "model_configs", "models", "seeds"):
        if name not in spec:
            raise ValueError(f"Missing pipeline setting: {name}.")
    _keys(spec.get("nmi", {}), {"method", "n_bins", "backend"}, "nmi")
    NMISettings(**spec.get("nmi", {})).validate()
    _keys(spec.get("graph", {}), {"weighted"}, "graph")
    if not isinstance(spec.get("graph", {}).get("weighted", False), bool):
        raise ValueError("graph.weighted must be a boolean.")
    _keys(spec["model_configs"], {"gnn", "recurrent"}, "model_configs")
    for name in ("models", "seeds"):
        if not isinstance(spec[name], list) or not spec[name] or len(set(spec[name])) != len(spec[name]):
            raise ValueError(f"{name} must be a nonempty list without duplicates.")
    if any(type(seed) is not int or seed < 0 or seed >= 2**32 for seed in spec["seeds"]):
        raise ValueError("Seeds must be integers in [0, 2^32).")
    _keys(spec.get("by_file", {}), {"train", "validation", "test"}, "by_file")
    _keys(spec.get("walk_forward", {}),
          {"min_train_files", "validation_files", "test_files", "step", "train_window", "max_folds"}, "walk_forward")
    for key, value in spec.get("walk_forward", {}).items():
        if (value is None and key not in {"train_window", "max_folds"}) or (value is not None and (type(value) is not int or value < 1)):
            raise ValueError(f"walk_forward.{key} must be a positive integer or null.")
    # These settings are owned by the runner and cannot be overridden by a
    # model preset or an accidental historical adjacency path.
    _keys(spec.get("data_overrides", {}), {
        "n_lags", "n_levels", "n_volume_bins", "feature_dtype", "storage_mode", "preprocess_chunk_size",
        "price_type", "label_mode", "threshold", "prediction_horizon", "normalize_prices",
        "max_samples_per_class", "max_val_samples", "max_test_samples", "subsample_seed",
        "extra_node_features", "node_feature_dim"}, "data_overrides")
    _keys(spec.get("training_overrides", {}), {
        "epochs", "batch_size", "static_graph_batching", "lr", "weight_decay", "early_stopping_patience",
        "lr_scheduler_patience", "lr_scheduler_factor", "grad_clip", "use_class_weights", "tune_threshold",
        "num_workers", "deterministic", "threshold_search"}, "training_overrides")
    for model in spec["models"]:
        if model_family(model) not in spec["model_configs"]:
            raise ValueError(f"Missing base configuration for {model}.")


def load_spec(path):
    spec = yaml.safe_load(project_path(path).read_text(encoding="utf-8"))
    validate_spec(spec)
    return spec


def model_configuration(spec, model):
    cfg = load_config(spec["model_configs"][model_family(model)])
    cfg["data"].update(deepcopy(spec.get("data_overrides", {})))
    cfg["training"].update(deepcopy(spec.get("training_overrides", {})))
    cfg["model"].update(type=model, family=model_family(model))
    validate_config(cfg)
    return cfg


def discover_days(spec):
    raw = project_path(spec["raw_dir"])
    messages = project_path(spec.get("message_dir", spec["raw_dir"]))
    pattern = re.compile(rf"^{re.escape(spec['symbol'])}_(\d{{4}}-\d{{2}}-\d{{2}})_(\d+)_(\d+)_orderbook_10\.csv$")
    days = []
    for path in sorted(raw.glob(f"{spec['symbol']}_*_orderbook_10.csv")):
        match = pattern.fullmatch(path.name)
        if not match:
            raise ValueError(f"Unsupported LOBSTER filename: {path.name}.")
        day = date.fromisoformat(match[1]).isoformat()
        if int(match[2]) >= int(match[3]):
            raise ValueError(f"Invalid session times in {path.name}.")
        message = messages / path.name.replace("_orderbook_10.csv", "_message_10.csv")
        days.append({"date": day, "orderbook": str(path.resolve()),
                     "message": str(message.resolve()) if message.is_file() else None})
    if not days or len({day["date"] for day in days}) != len(days):
        raise ValueError("Need exactly one orderbook file per date for the selected ticker.")
    return days


def make_plan(spec):
    validate_spec(spec)
    days = discover_days(spec)
    folds = []
    if spec["mode"] == "by_file":
        by_date = {day["date"]: day for day in days}
        fold = {}
        for key in ("train", "validation", "test"):
            values = spec.get("by_file", {}).get(key, [])
            if not isinstance(values, list):
                raise ValueError(f"by_file.{key} must be a list of dates.")
            selected = [str(value) for value in values]
            if not selected or selected != sorted(set(selected)):
                raise ValueError(f"by_file.{key} must contain unique dates in chronological order.")
            missing = set(selected) - set(by_date)
            if missing:
                raise ValueError(f"Missing orderbook dates: {sorted(missing)}.")
            fold[key] = [by_date[day] for day in selected]
        folds.append(fold)
    else:
        settings = spec.get("walk_forward", {})
        minimum = settings.get("min_train_files", 2)
        n_val, n_test = settings.get("validation_files", 1), settings.get("test_files", 1)
        step = settings.get("step", n_test)
        window = settings.get("train_window")
        if step < n_test or (window is not None and window < minimum):
            raise ValueError("Require step >= test_files and train_window >= min_train_files.")
        for end in range(minimum, len(days) - n_val - n_test + 1, step):
            folds.append({"train": days[max(0, end - window) if window else 0:end],
                          "validation": days[end:end + n_val], "test": days[end + n_val:end + n_val + n_test]})
            if settings.get("max_folds") is not None and len(folds) >= settings["max_folds"]:
                break
    if not folds:
        raise ValueError("Not enough days for the requested train/validation/test protocol.")
    for index, fold in enumerate(folds, 1):
        if not (fold["train"][-1]["date"] < fold["validation"][0]["date"]
                and fold["validation"][-1]["date"] < fold["test"][0]["date"]):
            raise ValueError("Require training dates strictly before validation, then test.")
        fold["name"] = f"fold_{index:03d}"
    configs = {model: model_configuration(spec, model) for model in spec["models"]}
    # Force a common evaluation task, even when architectures have different
    # lag depths. Require common lags for the official comparison as well.
    keys = ("n_lags", "price_type", "label_mode", "threshold", "prediction_horizon",
            "max_val_samples", "max_test_samples", "subsample_seed")
    contracts = [{key: cfg["data"].get(key) for key in keys} for cfg in configs.values()]
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Compared models must share lags, label settings and evaluation sampling; use data_overrides.")
    from remoma.dataset.preprocessing import normalize_extra_node_features, uses_order_flow_features
    for model, cfg in configs.items():
        if uses_order_flow_features(normalize_extra_node_features(cfg["data"].get("extra_node_features"))):
            if any(not day["message"] for fold in folds for key in ("train", "validation", "test") for day in fold[key]):
                raise ValueError(f"{model} requires exactly paired message files for order-flow features.")
    return {"folds": folds, "models": spec["models"], "seeds": spec["seeds"], "nmi": spec.get("nmi", {}),
            "configurations": configs}
