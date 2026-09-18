"""Lossless window indexing: store each raw row once, assemble only the current batch."""
from __future__ import annotations

from pathlib import Path
import numpy as np

from remoma.graph.adjacency import sha256_file
from remoma.utils.artifacts import read_json


def write_indexed_splits(paths, splits, all_data, all_labels, all_flow, binner, raw_paths, chunk):
    sources, artifacts = [], []
    for index, data in enumerate(all_data):
        volumes_path = paths["dir"] / f"volumes_{index:03d}.npy"
        volume_map = np.lib.format.open_memmap(volumes_path, mode="w+", dtype=np.float32,
                                             shape=(len(data), 20))
        for start in range(0, len(data), chunk):
            end = min(start + chunk, len(data))
            ask, bid = binner.transform_window(data[start:end, 1::4], data[start:end, 3::4])
            volume_map[start:end] = np.concatenate((ask, bid), axis=1)
        volume_map.flush()
        del volume_map
        source = {"raw": str(raw_paths[index].resolve()), "volumes": str(volumes_path.resolve())}
        if all_flow is not None:
            flow_path = paths["dir"] / f"flow_{index:03d}.npy"
            np.save(flow_path, all_flow[index], allow_pickle=False)
            source["flow"] = str(flow_path.resolve())
        sources.append(source)
        artifacts.extend({"path": path, "sha256": sha256_file(path)} for path in source.values())
    for split, samples in splits.items():
        # Match the materialized writer's deterministic grouping exactly.
        _, first = np.unique(samples[:, 0], return_index=True)
        order = samples[np.sort(first), 0]
        samples = np.concatenate([samples[samples[:, 0] == index] for index in order])
        labels = np.concatenate([all_labels[int(index)][samples[samples[:, 0] == index, 1]] for index in order])
        np.save(paths[f"X_{split}"], samples.astype(np.int32), allow_pickle=False)
        np.save(paths[f"y_{split}"], labels.astype(np.int64), allow_pickle=False)
    return {"indexed_sources": sources, "indexed_artifacts": artifacts}


class FeatureReader:
    """Batch feature access for materialized tensors or indexed windows.

    Memmaps reopen lazily in each worker. Indexed batches reuse pre-binned
    volumes and the same vectorized feature function as materialized storage.
    """

    def __init__(self, path):
        self.path = str(path)
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        self.indexed = array.ndim == 2 and array.shape[1] == 2 and array.dtype == np.int32
        self.meta = read_json(Path(path).parent / "meta.json") if self.indexed else None
        if self.indexed and self.meta.get("storage_mode") != "indexed":
            raise ValueError("Indexed sample arrays require matching metadata.")
        self.array = None
        self.sources = None

    def __getstate__(self):
        return {**self.__dict__, "array": None, "sources": None}

    def read(self, indices):
        indices = np.asarray(indices, dtype=np.intp)
        if self.array is None:
            self.array = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if not self.indexed:
            return np.asarray(self.array[indices], dtype=np.float32)
        from remoma.dataset.preprocessing import _extract_chunk_features_with_extras
        if self.sources is None:
            self.sources = [{key: np.load(path, mmap_mode="r", allow_pickle=False) for key, path in src.items()}
                            for src in self.meta["indexed_sources"]]
        samples = self.array[indices]
        sig = self.meta["signature"]
        result = np.empty((len(indices), self.meta["n_nodes"], self.meta["n_features"]), dtype=np.float32)
        for file_index in np.unique(samples[:, 0]):
            positions = np.flatnonzero(samples[:, 0] == file_index)
            source = self.sources[int(file_index)]
            block = _extract_chunk_features_with_extras(
                source["raw"], samples[positions, 1], sig["n_lags"], None,
                sig["normalize_prices"], sig["extra_node_features"], source.get("flow"), source["volumes"],
            )
            # Match an explicitly requested float16 materialization; the final
            # pipeline defaults to float32 and incurs no extra quantization.
            result[positions] = block.astype(self.meta["feature_dtype"]).astype(np.float32, copy=False)
        if not np.isfinite(result).all():
            raise ValueError("Feature overflow; use normalized prices or float32.")
        return result
