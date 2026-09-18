"""Training-only graph construction with reusable daily NMI artifacts."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import time

import numpy as np

from remoma.graph.adjacency import save_sparse_adjacency, sha256_file
from remoma.graph.nmi import NMISettings, daily_artifact, expand_relative
from remoma.graph.tmfg import TMFG, OutputMode
from remoma.utils.artifacts import exclusive_lock, identity, read_json, write_json


class TrainingGraphBuilder:
    """An invocation-scoped cache; callers pass only the fold's training files."""

    def __init__(self, cache: Path, progress=True):
        self.cache = Path(cache)
        self.progress = progress
        self.daily = {}
        self.sums = {}

    def build(self, train_paths: list[Path], settings: NMISettings, *, weighted=False):
        if not train_paths or len(set(train_paths)) != len(train_paths):
            raise ValueError("Graph training sources must be nonempty and unique.")
        settings_key = identity(asdict(settings))
        records = []
        for path in train_paths:
            key = (str(path.resolve()), settings_key)
            if key not in self.daily:
                _, record = daily_artifact(path, settings, self.cache, progress=self.progress)
                # Keep arrays memory-mapped: full mode can be large.
                self.daily[key] = record
            record = dict(self.daily[key], source_name=path.name)
            if sha256_file(path) != record["key"]["source_sha256"]:
                raise ValueError(f"Training input changed during the run: {path}.")
            records.append(record)
        ids = [record["id"] for record in records]
        key = {"schema": 1, "daily_ids": ids, "daily_names": [p.name for p in train_paths],
               "aggregation": "equal_weight_daily_mean", "weighted": weighted,
               "tmfg_sha256": sha256_file(Path(__file__).with_name("tmfg.py")),
               "builder_sha256": sha256_file(Path(__file__))}
        digest = identity(key)
        root = self.cache / "graphs" / digest
        manifest_path, graph_path = root / "manifest.json", root / "adjacency.npz"

        def existing():
            manifest = read_json(manifest_path)
            if manifest["key"] != key or sha256_file(graph_path) != manifest["graph_sha256"]:
                raise ValueError(f"Graph cache integrity failure: {root}.")
            return graph_path, manifest

        if manifest_path.exists():
            return existing()
        with exclusive_lock(self.cache / "locks" / (digest + ".graph.lock")):
            if manifest_path.exists():
                return existing()
            started = time.perf_counter()
            # Expanding folds extend the same sum in chronological order. For a
            # rolling window or reordered list, recompute from cached daily NMI.
            old_ids, total = self.sums.get(settings_key, ([], None))
            if old_ids != ids[:len(old_ids)]:
                old_ids, total = [], None
            for record in records[len(old_ids):]:
                array = np.load(self.cache / "nmi" / record["id"] / "values.npy", mmap_mode="r")
                if total is None:
                    total = np.array(array, dtype=np.float64, copy=True)
                else:
                    total += array
            self.sums[settings_key] = (ids, total)
            mean = total / len(ids)
            similarity = expand_relative(mean) if settings.method == "relative" else mean
            graph = TMFG()
            _, _, adjacency = graph.fit_transform(similarity, OutputMode.UNWEIGHTED_SPARSE_W_MATRIX.value)
            expected_edges = 6 * len(adjacency) - 12
            if np.count_nonzero(adjacency) != expected_edges:
                raise ValueError("TMFG edge count differs from 3N-6 undirected edges.")
            if weighted:
                adjacency *= similarity
                if np.count_nonzero(adjacency) != expected_edges:
                    raise ValueError("Selected TMFG edges have zero NMI; use an unweighted graph.")
            labels = [f"{side}{level}_lag{lag}" for side in ("ASKs", "BIDs")
                      for level in range(1, settings.n_levels + 1) for lag in range(settings.max_lag + 1)]
            save_sparse_adjacency(graph_path, adjacency, labels)
            manifest = {"key": key, "id": digest, "training_only": True,
                        "training_sources": records, "nmi": asdict(settings),
                        "graph_sha256": sha256_file(graph_path), "nodes": len(labels),
                        "directed_edges": expected_edges, "seconds": time.perf_counter() - started}
            write_json(manifest_path, manifest)
            return graph_path, manifest
