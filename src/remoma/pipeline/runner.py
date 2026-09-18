"""Execute a frozen plan and aggregate independent out-of-sample results."""
from __future__ import annotations

from copy import deepcopy
import csv
from importlib.metadata import version
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import yaml

from remoma.config import PROJECT_ROOT, project_path
from remoma.dataset.preprocessing import preprocess_signature, preprocess_to_disk, uses_order_flow_features
from remoma.graph.adjacency import sha256_file
from remoma.graph.construction import TrainingGraphBuilder
from remoma.graph.nmi import NMISettings, array_backend
from remoma.pipeline.specification import make_plan
from remoma.training.threshold import metrics_from_confusion
from remoma.utils.artifacts import exclusive_lock, identity, read_json, write_json


def code_manifest():
    paths = sorted((PROJECT_ROOT / "src/remoma").rglob("*.py")) + [PROJECT_ROOT / "scripts/train.py"]
    return {str(path.relative_to(PROJECT_ROOT)): sha256_file(path) for path in paths}


def input_manifest(plan):
    paths = sorted({day[key] for fold in plan["folds"] for split in ("train", "validation", "test")
                    for day in fold[split] for key in ("orderbook", "message") if day[key]})
    return {path: sha256_file(path) for path in paths}


def _run_config(base, spec, fold, graph_path, graph_manifest, source_hashes, implementation):
    cfg = deepcopy(base)
    days = fold["train"] + fold["validation"] + fold["test"]
    n_train, n_val = len(fold["train"]), len(fold["validation"])
    cfg["data"].update(raw_dir=str(project_path(spec["raw_dir"])),
                       raw_files=[day["orderbook"] for day in days],
                       message_dir=str(project_path(spec.get("message_dir", spec["raw_dir"]))),
                       split_strategy="by_file", train_files=list(range(n_train)),
                       val_files=list(range(n_train, n_train + n_val)),
                       test_files=list(range(n_train + n_val, len(days))),
                       adj_matrix_path=str(graph_path), allow_missing_raw=False,
                       raw_cache_dir=str(project_path(spec["cache_dir"])),
                       use_precomputed=True, force_preprocess=False)
    for section in [cfg["model"], *(cfg["model"].get("overrides") or {}).values()]:
        section.pop("adjacency_path", None)
    signature = preprocess_signature(cfg)
    orderbooks = {day["orderbook"]: source_hashes[day["orderbook"]] for day in days}
    messages = ({day["message"]: source_hashes[day["message"]] for day in days}
                if uses_order_flow_features(signature["extra_node_features"]) else {})
    processed_key = {"signature": signature, "orderbooks": orderbooks, "messages": messages,
                     "implementation": {key: value for key, value in implementation.items()
                                        if "/dataset/" in key or key.endswith("utils/io.py")}}
    cfg["data"]["processed_dir"] = str(project_path(spec["cache_dir"]) / "processed" / identity(processed_key))
    cfg["pipeline_provenance"] = {"fold": fold, "graph_manifest": str(graph_manifest),
                                  "nmi": spec.get("nmi", {})}
    return cfg


def _verify_completed(directory):
    completed = read_json(directory / "complete.json")
    for relative, digest in completed["artifacts"].items():
        if sha256_file(directory / relative) != digest:
            raise ValueError(f"Completed run artifact changed: {directory / relative}.")
    return read_json(directory / "results/metrics.json")


def aggregate_results(records, output):
    """Pool fold confusion counts within each seed, then summarize seeds.

    Seeds repeat the same test data and are never pooled as additional samples.
    Folds in this runner have nonoverlapping test dates.
    """
    rows, groups = [], {}
    for fold, result in records:
        for rule, payload in result["rules"].items():
            row = {"fold": fold["name"], "test_dates": " ".join(day["date"] for day in fold["test"]),
                   "model": result["model"], "seed": result["seed"], "rule": rule,
                   "samples": result["samples"], **{k: v for k, v in payload["metrics"].items() if k != "f1_per_class"}}
            rows.append(row)
            key = (result["model"], rule, result["seed"])
            groups.setdefault(key, []).append(np.asarray(payload["confusion_matrix"], dtype=np.int64))
    if not rows:
        return
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    per_seed, aggregates = [], []
    for (model, rule, seed), matrices in sorted(groups.items()):
        cm = np.sum(matrices, axis=0)
        per_seed.append({"model": model, "rule": rule, "seed": seed, "folds": len(matrices),
                         "samples": int(cm.sum()), "metrics": metrics_from_confusion(cm),
                         "confusion_matrix": cm.tolist()})
    for model, rule in sorted({(r["model"], r["rule"]) for r in per_seed}):
        selected = [r for r in per_seed if (r["model"], r["rule"]) == (model, rule)]
        metrics = {}
        for key in selected[0]["metrics"]:
            if key == "f1_per_class":
                continue
            values = [r["metrics"][key] for r in selected]
            metrics[key] = {"mean": float(np.mean(values)),
                            "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else None}
        aggregates.append({"model": model, "rule": rule, "seeds": [r["seed"] for r in selected], "metrics": metrics})
    write_json(output / "summary.json", {"per_run": rows, "pooled_folds_per_seed": per_seed,
                                          "across_seeds": aggregates})
    lines = ["# Pipeline results", "", "Metrics pool nonoverlapping test folds within each seed. "
             "Reported deviations are sample standard deviations across seeds, not confidence intervals.", "",
             "| Model | Rule | Seeds | Macro F1 | MCC | Accuracy |", "| --- | --- | --- | --- | --- | --- |"]
    for item in aggregates:
        formatted = []
        for key in ("f1_macro", "mcc", "accuracy"):
            metric = item["metrics"][key]
            formatted.append(f"{metric['mean']:.6f}" +
                             (f" ± {metric['sample_std']:.6f}" if metric["sample_std"] is not None else ""))
        lines.append(f"| {item['model']} | {item['rule']} | {len(item['seeds'])} | " + " | ".join(formatted) + " |")
    (output / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def execute(spec, *, resume=False, stage="all", restart_incomplete=False, progress=True):
    plan = make_plan(spec)
    output, cache = project_path(spec["output_dir"]), project_path(spec["cache_dir"])
    raw = project_path(spec["raw_dir"])
    if output == cache or output in cache.parents or cache in output.parents:
        raise ValueError("output_dir and cache_dir must be separate, non-nested directories.")
    if raw == output or raw == cache or raw in output.parents or raw in cache.parents:
        raise ValueError("Outputs and caches must not be placed inside raw data.")
    implementation = code_manifest()
    sources = input_manifest(plan)
    _, backend = array_backend(spec.get("nmi", {}).get("backend", "auto"))
    import torch
    versions = {"python": sys.version.split()[0], "nmi_backend": backend,
                "packages": {name: version(name) for name in
                             ("numpy", "pandas", "torch", "torch-geometric", "scikit-learn", "PyYAML")},
                "platform": {"system": platform.system(), "release": platform.release(),
                             "machine": platform.machine()}, "cuda": torch.version.cuda,
                "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    frozen = {"plan": plan, "spec_yaml": yaml.safe_dump(spec, sort_keys=True),
              "inputs": sources, "code": implementation, "versions": versions}
    fingerprint = identity(frozen)
    with exclusive_lock(output / ".pipeline.lock"):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            if not resume:
                raise FileExistsError("Experiment already exists. Use --resume to validate and reuse completed stages.")
            if read_json(manifest_path)["fingerprint"] != fingerprint:
                raise ValueError("Inputs, code, environment, or settings changed. Select a new output_dir.")
        else:
            if any(path.name != ".pipeline.lock" for path in output.iterdir()):
                raise FileExistsError(f"Refusing to reuse a nonempty experiment directory: {output}.")
            write_json(manifest_path, {"fingerprint": fingerprint, **frozen})
        builder = TrainingGraphBuilder(cache, progress=progress)
        records = []
        processed = set()
        for fold in plan["folds"]:
            root = output / fold["name"]
            root.mkdir(parents=True, exist_ok=True)
            write_json(root / "split.json", fold)
            for model in spec["models"]:
                base = plan["configurations"][model]
                settings = NMISettings(max_lag=int(base["data"]["n_lags"]),
                                       n_levels=int(base["data"]["n_levels"]), **spec.get("nmi", {}))
                print(f"\n{fold['name']} / {model}: training-only {settings.method} NMI + TMFG", flush=True)
                graph_path, graph_record = builder.build(
                    [Path(day["orderbook"]) for day in fold["train"]], settings,
                    weighted=spec.get("graph", {}).get("weighted", False))
                expected_hashes = [sources[day["orderbook"]] for day in fold["train"]]
                if [r["key"]["source_sha256"] for r in graph_record["training_sources"]] != expected_hashes:
                    raise ValueError("Graph sources differ from the frozen training split.")
                graph_ref = root / f"graph_{model}.json"
                write_json(graph_ref, {"path": str(graph_path), **graph_record})
                if stage == "graphs":
                    continue
                cfg = _run_config(base, spec, fold, graph_path, graph_ref, sources, implementation)
                for seed in spec["seeds"]:
                    run_dir = root / model / f"seed_{seed}"
                    if (run_dir / "complete.json").exists():
                        records.append((fold, _verify_completed(run_dir)))
                        print(f"Reusing completed model: {run_dir}", flush=True)
                        continue
                    if run_dir.exists() and any(run_dir.iterdir()):
                        if not restart_incomplete:
                            raise RuntimeError(f"Incomplete model run: {run_dir}. Use --resume --restart-incomplete "
                                               "to archive the attempt and retrain that model from its seed.")
                        archive = run_dir.with_name(run_dir.name + f"_interrupted_{time.time_ns()}")
                        run_dir.rename(archive)
                    processed_dir = cfg["data"]["processed_dir"]
                    if processed_dir not in processed:
                        with exclusive_lock(cache / "locks" / (Path(processed_dir).name + ".processed.lock")):
                            preprocess_to_disk(cfg, verbose=True)
                        processed.add(processed_dir)
                    run_cfg = deepcopy(cfg)
                    run_cfg["training"]["seed"] = seed
                    run_cfg["paths"]["checkpoints"] = str(run_dir)
                    config_path = root / "configs" / f"{model}_seed_{seed}.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    config_path.write_text(yaml.safe_dump(run_cfg, sort_keys=False), encoding="utf-8")
                    env = dict(os.environ, PYTHONUNBUFFERED="1")
                    if run_cfg["training"].get("deterministic", False):
                        env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
                    started = time.perf_counter()
                    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/train.py"), "--config", str(config_path),
                                    "--checkpoint-dir", str(run_dir)], check=True, cwd=PROJECT_ROOT, env=env)
                    artifact_paths = [p for p in run_dir.rglob("*") if p.is_file()]
                    write_json(run_dir / "complete.json", {
                        "seconds": time.perf_counter() - started,
                        "artifacts": {str(p.relative_to(run_dir)): sha256_file(p) for p in artifact_paths},
                    })
                    records.append((fold, read_json(run_dir / "results/metrics.json")))
        # Fail if a source was modified while graph/preprocessing/training ran.
        if input_manifest(plan) != sources:
            raise ValueError("Source files changed during execution; this experiment cannot be finalized.")
        if code_manifest() != implementation:
            raise ValueError("Source code changed during execution; this experiment cannot be finalized.")
        aggregate_results(records, output)
        write_json(output / ("graphs_complete.json" if stage == "graphs" else "complete.json"),
                   {"fingerprint": fingerprint, "model_runs": len(records), "stage": stage})
        print(f"\nPipeline stage '{stage}' complete: {output}", flush=True)
    return output
