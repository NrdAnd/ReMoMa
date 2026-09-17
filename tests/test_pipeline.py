"""Regression coverage for model integration and scientific data contracts."""
from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.data import Batch, Data

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from remoma.config import load_config
from remoma.dataset.labeling import compute_labels
from remoma.dataset.preprocessing import (
    has_compatible_processed_dataset, preprocess_signature, preprocess_to_disk,
    split_by_file, split_by_lag,
)
from remoma.graph.adjacency import load_labeled_adjacency, load_tmfg_edge_index
from remoma.models import build_model, model_family, model_names, supports_static_batching
from remoma.training.metrics import compute_metrics, format_report
from remoma.training.threshold import apply_thresholds
from remoma.utils.checkpoints import load_checkpoint_state, save_run_configuration
from remoma.utils.io import discover_message_files

torch.set_num_threads(1)


def make_fixture(root: Path, model_type: str = "gcn") -> dict:
    raw = root / "raw"
    raw.mkdir(exist_ok=True)
    rng = np.random.default_rng(17)
    for day in range(5):
        ticks = np.arange(140)
        mid = 100_000 + 300 * np.sin(ticks / 4).round()
        book = np.empty((len(ticks), 40), dtype=np.float32)
        for level in range(10):
            book[:, 4 * level] = mid + 50 + level * 100
            book[:, 4 * level + 2] = mid - 50 - level * 100
            book[:, 4 * level + 1] = rng.integers(1, 500, size=len(ticks))
            book[:, 4 * level + 3] = rng.integers(1, 500, size=len(ticks))
        np.savetxt(raw / f"day{day}_orderbook_10.csv", book, delimiter=",")
    labels = [f"{side}_{level}_lag_{lag}" for side in ("ask", "bid")
              for level in range(10) for lag in range(3)]
    adjacency = np.zeros((60, 60), dtype=np.float32)
    for index in range(60):
        adjacency[index, (index + 1) % 60] = adjacency[(index + 1) % 60, index] = 1
    graph = root / "graph.csv"
    pd.DataFrame(adjacency, index=labels, columns=labels).to_csv(graph)
    return {
        "data": {"raw_dir": str(raw), "adj_matrix_path": str(graph),
                 "processed_dir": str(root / "processed"), "n_lags": 2,
                 "n_levels": 10, "n_volume_bins": 8, "prediction_horizon": 2,
                 "threshold": 100, "price_type": "mid", "label_mode": "abs",
                 "split_strategy": "by_file", "train_files": [0, 1, 2],
                 "val_files": [3], "test_files": [4], "train_ratio": 0.6, "val_ratio": 0.2,
                 "normalize_prices": True, "feature_dtype": "float32",
                 "subsample_seed": 42, "max_val_samples": 30, "max_test_samples": 30},
        "model": {"type": model_type, "family": model_family(model_type),
                  "hidden_channels": 8, "hidden_dim": 4, "num_layers": 1,
                  "dropout": 0.0, "num_heads": 2, "cnn_channels": 4,
                  "cnn_kernel": 3, "add_lag_feature": True},
        "training": {"epochs": 1, "batch_size": 32, "num_workers": 0,
                     "static_graph_batching": True, "lr": 0.001, "weight_decay": 0.0,
                     "early_stopping_patience": 2, "use_class_weights": False,
                     "tune_threshold": True, "seed": 42},
        "paths": {"checkpoints": str(root / "runs")},
    }


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_model_forward_backward_and_static_equivalence(self):
        edges = load_tmfg_edge_index(self.cfg["data"]["adj_matrix_path"], n_lags=2)
        for name in model_names():
            with self.subTest(model=name):
                cfg = deepcopy(self.cfg)
                cfg["model"].update(type=name, family=model_family(name))
                # Engineered channels must work in both families.
                cfg["data"]["extra_node_features"] = ["spread", "side"]
                model = build_model(cfg)
                x = torch.randn(3, 60, 4)
                batch = Batch.from_data_list([Data(x=item, edge_index=edges) for item in x])
                model.eval()
                output = model(batch)
                self.assertEqual(tuple(output.shape), (3, 3))
                self.assertTrue(torch.isfinite(output).all())
                if supports_static_batching(name):
                    torch.testing.assert_close(output, model.forward_static(x, edges), atol=2e-6, rtol=2e-5)
                model.train()
                torch.nn.functional.cross_entropy(model(batch), torch.tensor([0, 1, 2])).backward()
                self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()))

    def test_graph_reordering_and_cache_invalidation(self):
        path = Path(self.cfg["data"]["adj_matrix_path"])
        cache = self.root / "edges.pt"
        original = load_tmfg_edge_index(path, cache, n_lags=2)
        frame = pd.read_csv(path, index_col=0)
        frame.iloc[::-1, ::-1].to_csv(path)
        torch.testing.assert_close(original, load_tmfg_edge_index(path, cache, n_lags=2))
        frame.iloc[0, 1] = frame.iloc[1, 0] = 0
        frame.to_csv(path)
        self.assertEqual(load_tmfg_edge_index(path, cache, n_lags=2).shape[1], original.shape[1] - 2)
        with self.assertRaisesRegex(ValueError, "labels do not match"):
            load_tmfg_edge_index(path, n_lags=3)
        frame.iloc[0, 2] = 5
        frame.to_csv(path)
        with self.assertRaisesRegex(ValueError, "symmetric"):
            load_labeled_adjacency(path)

    def test_by_file_rejects_overlap_and_out_of_range(self):
        arrays = [np.empty((140, 40)) for _ in range(5)]
        self.cfg["data"]["val_files"] = [2]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            split_by_file(arrays, self.cfg, 2, 2)
        self.cfg["data"]["val_files"] = [5]
        with self.assertRaisesRegex(ValueError, "outside"):
            split_by_file(arrays, self.cfg, 2, 2)

    def test_temporal_windows_and_targets_do_not_overlap(self):
        train, val, test = split_by_lag([np.empty((140, 40))], self.cfg, 2, 2)
        self.assertLess(train[:, 1].max() + 2, val[:, 1].min() - 2)
        self.assertLess(val[:, 1].max() + 2, test[:, 1].min() - 2)

    def test_preprocessing_cache_tracks_data_seed_content_and_shapes(self):
        preprocess_to_disk(self.cfg, verbose=False)
        self.assertTrue(has_compatible_processed_dataset(self.cfg))
        other = deepcopy(self.cfg)
        other["training"]["seed"] += 1
        self.assertEqual(preprocess_signature(self.cfg), preprocess_signature(other))
        other["data"]["subsample_seed"] += 1
        self.assertFalse(has_compatible_processed_dataset(other))
        raw = self.root / "raw/day0_orderbook_10.csv"
        raw.write_text(raw.read_text().replace("1.000500000000000000e+05", "1.001500000000000000e+05", 1))
        # Append a valid row to guarantee a content change regardless of formatting.
        with raw.open("a") as handle:
            handle.write(raw.read_text().splitlines()[0] + "\n")
        self.assertFalse(has_compatible_processed_dataset(self.cfg))

    def test_by_lag_binner_uses_only_training_rows(self):
        cfg = deepcopy(self.cfg)
        cfg["data"]["split_strategy"] = "by_lag"
        for path in (self.root / "raw").glob("*.csv"):
            book = np.loadtxt(path, delimiter=",")
            book[95:, 1::2] = 1e8
            np.savetxt(path, book, delimiter=",")
        paths = preprocess_to_disk(cfg, verbose=False)
        from remoma.dataset.binning import VolumeBinner
        self.assertLess(VolumeBinner.load(paths["binner"])._edges.max(), 1000)

    def test_order_flow_requires_exact_file_identity(self):
        (self.root / "raw/unrelated_message_10.csv").write_text("0,1,1,1,1,1\n")
        with self.assertRaises(FileNotFoundError):
            discover_message_files(self.root / "raw", [self.root / "raw/day0_orderbook_10.csv"])

    def test_zero_threshold_and_absent_classes(self):
        book = np.ones((5, 40)) * 100
        np.testing.assert_array_equal(compute_labels(book, 0, k=1, label_mode="abs"), [1, 1, 1, 1])
        metrics = compute_metrics(np.array([1, 1]), np.array([1, 1]))
        self.assertAlmostEqual(metrics["f1_macro"], 1 / 3)
        self.assertEqual(len(metrics["f1_per_class"]), 3)
        self.assertIn("down", format_report(np.array([1]), np.array([1])))
        np.testing.assert_array_equal(apply_thresholds(np.array([[.7, .2, .1], [.1, .8, .1], [.1, .2, .7]]), .6, .6), [0, 1, 2])

    def test_checkpoint_rejects_graph_and_model_changes(self):
        model = build_model(self.cfg)
        saved = save_run_configuration(self.cfg, self.root / "checkpoint")
        path = self.root / "checkpoint/best.pt"
        torch.save(model.state_dict(), path)
        load_checkpoint_state(model, path, saved, torch.device("cpu"))
        other = deepcopy(saved)
        other["model"]["hidden_channels"] = 9
        with self.assertRaisesRegex(ValueError, "contract"):
            load_checkpoint_state(model, path, other, torch.device("cpu"))
        graph = Path(saved["data"]["adj_matrix_path"])
        with graph.open("a") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(ValueError, "graph differs"):
            load_checkpoint_state(model, path, saved, torch.device("cpu"))

    def test_training_evaluation_and_threshold_reload_for_both_families(self):
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        for name in ("gcn", "gat", "recurrent_sparse_sthnn"):
            with self.subTest(model=name):
                cfg = deepcopy(self.cfg)
                cfg["model"].update(type=name, family=model_family(name))
                cfg["training"]["num_workers"] = 1 if name == "gat" else 0
                config = self.root / f"{name}.yaml"
                config.write_text(yaml.safe_dump(cfg))
                run = self.root / name
                commands = [
                    ["train.py", "--config", str(config), "--checkpoint-dir", str(run)],
                    ["evaluate.py", "--checkpoint", str(run / "best.pt")],
                    ["evaluate.py", "--checkpoint", str(run / "best.pt"), "--thresholds", str(run / "thresholds.json")],
                ]
                for script, *args in commands:
                    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                            cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertTrue((run / "resolved_config.yaml").exists())
                self.assertTrue((run / "results/summary.csv").exists())
                self.assertEqual(json.loads((run / "metadata.json").read_text())["contract"]["model"]["type"], name)

    def test_walk_forward_and_ensemble_workflows_on_tiny_fixture(self):
        cfg = deepcopy(self.cfg)
        cfg["model"].update(type="recurrent_sparse_sthnn", family="recurrent")
        cfg["training"]["tune_threshold"] = False
        config = self.root / "recurrent.yaml"
        config.write_text(yaml.safe_dump(cfg))
        generated = self.root / "folds"
        summary = generated / "tiny/summary.json"
        environment = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
        commands = [
            ["run_walk_forward.py", "--base-config", str(config), "--config-dir", str(generated),
             "--name", "tiny", "--max-folds", "1", "--run", "--evaluate", "--lo", "0.4", "--hi", "0.8", "--step", "0.2"],
            ["run_stable_thresholds.py", "--fold-results", str(summary), "--output-dir", str(self.root / "stable")],
            ["run_walk_forward_ensemble.py", "--fold-results", str(summary), "--seeds", "123", "--run-train",
             "--output-dir", str(self.root / "ensemble"), "--lo", "0.4", "--hi", "0.8", "--step", "0.2"],
            ["run_hparam_grid.py", "--base-config", str(config), "--config-dir", str(self.root / "grid"),
             "--hidden-dims", "4", "--dropouts", "0.15", "--message-iterations", "1", "--max-combos", "1"],
        ]
        for script, *args in commands:
            result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                    cwd=ROOT, env=environment, capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(summary.exists())
        self.assertTrue((self.root / "stable/stable_thresholds.json").exists())



if __name__ == "__main__":
    unittest.main()
