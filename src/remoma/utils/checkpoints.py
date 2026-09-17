"""Checkpoint provenance, configuration snapshots, and compatibility checks."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import platform
import shutil
import subprocess
import warnings

import torch
import yaml

from remoma.config import load_config
from remoma.graph.adjacency import sha256_file


def model_contract(cfg: dict) -> dict:
    model = deepcopy(cfg["model"])
    model_type = model["type"].lower()
    overrides = model.pop("overrides", {}) or {}
    model.update(overrides.get(model_type, {}) or {})
    model.pop("adjacency_path", None)
    model.pop("family", None)
    model["type"] = model_type
    data = cfg["data"]
    return {"model": model, "features": {key: data.get(key) for key in (
        "n_lags", "n_levels", "extra_node_features", "node_feature_dim",
        "n_volume_bins", "normalize_prices", "price_type", "label_mode",
        "threshold", "prediction_horizon",
    )}}


def adjacency_path(cfg: dict) -> Path:
    model = cfg["model"]
    selected = (model.get("overrides") or {}).get(model["type"].lower(), {}) or {}
    return Path(selected.get("adjacency_path", model.get("adjacency_path", cfg["data"]["adj_matrix_path"])))


def save_run_configuration(cfg: dict, directory: Path) -> dict:
    """Save effective CLI settings and an immutable copy of the selected graph."""
    directory.mkdir(parents=True, exist_ok=True)
    source = adjacency_path(cfg)
    target = directory / ("adjacency" + source.suffix)
    shutil.copy2(source, target)
    saved = deepcopy(cfg)
    saved["paths"]["checkpoints"] = str(directory.resolve())
    saved["data"]["adj_matrix_path"] = str(target.resolve())
    for section in [saved["model"], *(saved["model"].get("overrides") or {}).values()]:
        if "adjacency_path" in section:
            section["adjacency_path"] = str(target.resolve())
    (directory / "resolved_config.yaml").write_text(yaml.safe_dump(saved, sort_keys=False), encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    metadata = {"schema_version": 1, "git_commit": revision, "tracked_worktree_dirty": dirty,
                "python": platform.python_version(), "torch": str(torch.__version__),
                "graph_sha256": sha256_file(target), "contract": model_contract(saved)}
    processed = Path(cfg["data"]["processed_dir"])
    if (processed / "binner.pkl").exists():
        shutil.copy2(processed / "binner.pkl", directory / "binner.pkl")
        metadata["binner_sha256"] = sha256_file(directory / "binner.pkl")
    if (processed / "meta.json").exists():
        shutil.copy2(processed / "meta.json", directory / "preprocessing_meta.json")
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return saved


def checkpoint_config(checkpoint: str | Path, config: str | Path | None) -> dict:
    if config is not None:
        return load_config(config)
    saved = Path(checkpoint).resolve().parent / "resolved_config.yaml"
    if not saved.exists():
        raise ValueError("Legacy checkpoint has no resolved_config.yaml; supply its original --config.")
    return load_config(saved)


def load_checkpoint_state(model, checkpoint: str | Path, cfg: dict, device) -> None:
    """Reject changed architecture/feature/graph semantics before loading weights."""
    checkpoint = Path(checkpoint)
    metadata_path = checkpoint.parent / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["contract"] != model_contract(cfg):
            raise ValueError("Checkpoint configuration differs from the requested model/feature contract; use resolved_config.yaml.")
        if metadata["graph_sha256"] != sha256_file(adjacency_path(cfg)):
            raise ValueError("Checkpoint graph differs from the requested adjacency.")
        if "binner_sha256" in metadata:
            binner = Path(cfg["data"]["processed_dir"]) / "binner.pkl"
            if not binner.exists() or metadata["binner_sha256"] != sha256_file(binner):
                raise ValueError("Checkpoint volume binner differs from the processed dataset.")
    else:
        warnings.warn("Legacy checkpoint: graph and configuration provenance cannot be verified.", stacklevel=2)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
