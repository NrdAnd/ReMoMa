"""Small CPU references for graph provenance, numerical equivalence, and complete runs."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import confusion_matrix, normalized_mutual_info_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from remoma.dataset.indexed import FeatureReader
from remoma.dataset.preprocessing import preprocess_to_disk, has_compatible_processed_dataset
from remoma.graph.adjacency import load_labeled_adjacency, load_tmfg_edge_index
from remoma.graph.construction import TrainingGraphBuilder
from remoma.graph.nmi import NMISettings, compute_daily_nmi, discretize, discrete_nmi, expand_relative
from remoma.graph.tmfg import TMFG, OutputMode
from remoma.pipeline.runner import execute, aggregate_results
from remoma.pipeline.specification import make_plan
from remoma.training.threshold import (
    _threshold_confusions, apply_thresholds, decision_metrics, metrics_from_confusion, search_thresholds,
)
from test_pipeline import make_fixture

torch.set_num_threads(1)


class NMITests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("REMOMA_TEST_CUDA") == "1", "Enable explicitly on the CUDA compute host")
    def test_cuda_parity(self):
        volumes = np.random.default_rng(8).integers(0, 50, (100, 4))
        for method in ("relative", "full"):
            cpu = NMISettings(method=method, max_lag=3, n_bins=8, n_levels=2, backend="cpu")
            expected = compute_daily_nmi(volumes, cpu, progress=False)
            actual = compute_daily_nmi(volumes, replace(cpu, backend="cuda"), progress=False)
            np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-11)

    def test_full_reference_and_relative_representatives(self):
        rng = np.random.default_rng(42)
        volumes = rng.integers(0, 20, size=(90, 4))
        settings = NMISettings(method="full", max_lag=3, n_bins=5, n_levels=2, backend="cpu")
        full = compute_daily_nmi(volumes, settings, progress=False)
        columns = [discretize(volumes[3-lag:len(volumes)-lag, base], 5)
                   for base in range(4) for lag in range(4)]
        for i in range(16):
            for j in range(16):
                expected = normalized_mutual_info_score(columns[i], columns[j], average_method="arithmetic")
                self.assertAlmostEqual(full[i, j], expected, places=12)
        relative = compute_daily_nmi(volumes, replace(settings, method="relative"), progress=False)
        for lag in range(4):
            for i in range(4):
                for j in range(4):
                    self.assertAlmostEqual(relative[lag, i, j], full[i * 4 + lag, j * 4], places=12)
        expanded = expand_relative(relative)
        np.testing.assert_array_equal(expanded, expanded.T)
        # Same base pair, same signed lag difference, different absolute lags.
        self.assertEqual(expanded[3, 4 + 2], expanded[1, 4])
        self.assertEqual(expanded[0, 4 + 2], relative[2, 1, 0])

    def test_binary_constant_and_sparse_histograms(self):
        binary = np.tile([0, 1], 50)
        self.assertEqual(len(np.unique(discretize(binary, 2000))), 2)
        constant = np.zeros(100, dtype=np.uint16)
        self.assertEqual(discrete_nmi(constant, constant, 2000), 1)
        self.assertEqual(discrete_nmi(constant, binary, 2000), 0)
        rng = np.random.default_rng(3)
        x, y = rng.integers(0, 40, (2, 100))
        self.assertAlmostEqual(discrete_nmi(x, y, 2000), normalized_mutual_info_score(x, y), places=12)
        with self.assertRaises(ValueError):
            compute_daily_nmi(np.ones((3, 4)), NMISettings(max_lag=3, n_levels=2), progress=False)

    def test_tmfg_zero_gain_and_sparse_graph(self):
        for size in (4, 5, 16):
            for matrix in (np.zeros((size, size)), np.ones((size, size))):
                _, _, adjacency = TMFG().fit_transform(matrix, OutputMode.UNWEIGHTED_SPARSE_W_MATRIX.value)
                self.assertEqual(np.count_nonzero(adjacency), 6 * size - 12)
                np.testing.assert_array_equal(adjacency, adjacency.T)


class ThresholdTests(unittest.TestCase):
    def test_exact_histogram_search_matches_brute_force(self):
        rng = np.random.default_rng(25)
        grid = np.array([0, .1, .33, .5, .7, .9, 1])
        for dtype in (np.float32, np.float64):
            probs = rng.dirichlet([1, 1, 1], 100).astype(dtype)
            probs = np.concatenate((probs, np.array([[.1, .8, .1], [.7, .2, .1], [.5, 0, .5],
                                                     [.33, .34, .33]], dtype=dtype)))
            labels = rng.integers(0, 3, len(probs))
            matrices = _threshold_confusions(probs, labels, grid)
            for i, td in enumerate(grid):
                for j, tu in enumerate(grid):
                    pred = apply_thresholds(probs, float(td), float(tu))
                    cm = confusion_matrix(labels, pred, labels=[0, 1, 2])
                    np.testing.assert_array_equal(matrices[i, j], cm)
                    expected = decision_metrics(pred, labels)
                    actual = metrics_from_confusion(cm)
                    for key, value in expected.items():
                        self.assertAlmostEqual(actual[key], value, places=13)
            found = search_thresholds(probs, labels, lo=.1, hi=.9, step=.2)
            brute = max(decision_metrics(apply_thresholds(probs, td, tu), labels)["f1_macro"]
                        for td in [.1, .3, .5, .7, .9] for tu in [.1, .3, .5, .7, .9])
            self.assertAlmostEqual(found["score"], brute, places=13)


class CompletePipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cfg = make_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def spec(self):
        dates = ["2019-01-22", "2019-01-23", "2019-01-24", "2019-01-25", "2019-01-28"]
        for i, day in enumerate(dates):
            path = self.root / "raw" / f"day{i}_orderbook_10.csv"
            if path.exists():
                path.rename(path.with_name(f"CSCO_{day}_34200000_57600000_orderbook_10.csv"))
        base = self.root / "base.yaml"
        base.write_text(yaml.safe_dump(self.cfg))
        return {
            "version": 1, "symbol": "CSCO", "raw_dir": str(self.root / "raw"),
            "output_dir": str(self.root / "experiment"), "cache_dir": str(self.root / "cache"),
            "mode": "walk_forward", "walk_forward": {"min_train_files": 2},
            "by_file": {"train": dates[:3], "validation": dates[3:4], "test": dates[4:]},
            "models": ["gcn", "recurrent_sparse_sthnn"], "model_configs": {"gnn": str(base), "recurrent": str(base)},
            "seeds": [42], "nmi": {"method": "relative", "n_bins": 5, "backend": "cpu"},
            "data_overrides": {"storage_mode": "indexed", "feature_dtype": "float32"},
            "training_overrides": {"epochs": 1, "threshold_search": {"lo": .3, "hi": .7, "step": .2}},
        }

    def test_indexed_features_equal_materialized_including_extras(self):
        cfg = deepcopy(self.cfg)
        cfg["data"].update(extra_node_features=["spread", "level_imbalance", "depth_imbalance", "microprice", "side",
                                                 "order_cancel_flow", "order_limit_flow", "order_trade_flow"],
                           max_samples_per_class=15, preprocess_chunk_size=19)
        for book in sorted((self.root / "raw").glob("*.csv")):
            messages = np.column_stack((np.arange(140), np.tile([1, 2, 4, 5, 3], 28), np.arange(140),
                                        np.ones(140) * 10, np.ones(140) * 100000, np.tile([-1, 1], 70)))
            np.savetxt(book.with_name(book.name.replace("orderbook", "message")), messages, delimiter=",")
        dense = preprocess_to_disk(cfg, verbose=False)
        cfg["data"].update(storage_mode="indexed", processed_dir=str(self.root / "indexed"),
                            raw_cache_dir=str(self.root / "cache"))
        indexed = preprocess_to_disk(cfg, verbose=False)
        self.assertTrue(has_compatible_processed_dataset(cfg))
        for split in ("train", "val", "test"):
            expected = np.load(dense[f"X_{split}"])
            reader = FeatureReader(indexed[f"X_{split}"])
            np.testing.assert_array_equal(expected, reader.read(np.arange(len(expected))))
            np.testing.assert_array_equal(np.load(dense[f"y_{split}"]), np.load(indexed[f"y_{split}"]))
        self.assertLess(indexed["X_train"].stat().st_size, dense["X_train"].stat().st_size)
        # Changing cached binary data cannot be mistaken for a valid artifact.
        metadata = json.loads(indexed["meta"].read_text())
        target = Path(metadata["indexed_sources"][0]["volumes"])
        with target.open("ab") as handle:
            handle.write(b"modified")
        self.assertFalse(has_compatible_processed_dataset(cfg))

    def test_training_only_graph_cache_and_mutation_detection(self):
        spec = self.spec()
        plan = make_plan(spec)
        train = [Path(day["orderbook"]) for day in plan["folds"][0]["train"]]
        settings = NMISettings(max_lag=2, n_bins=5, backend="cpu")
        cache = self.root / "cache"
        graph, manifest = TrainingGraphBuilder(cache, progress=False).build(train, settings)
        frame = load_labeled_adjacency(graph)
        self.assertEqual(frame.shape, (60, 60))
        self.assertEqual(load_tmfg_edge_index(graph, n_lags=2).shape[1], 348)
        self.assertEqual([r["source_name"] for r in manifest["training_sources"]], [p.name for p in train])
        heldout = Path(plan["folds"][0]["test"][0]["orderbook"])
        heldout.write_text("Held-out content must not be read by the graph builder.")
        with patch("remoma.graph.nmi.compute_daily_nmi", side_effect=AssertionError("must reuse")):
            graph_again, _ = TrainingGraphBuilder(cache, progress=False).build(train, settings)
        self.assertEqual(graph, graph_again)
        original = pd.read_csv(train[0], header=None)
        original.iloc[0, 1] += 100
        original.to_csv(train[0], header=False, index=False)
        graph_new, _ = TrainingGraphBuilder(cache, progress=False).build(train, settings)
        self.assertNotEqual(graph_new, graph)

    def test_date_plans_and_invalid_settings(self):
        spec = self.spec()
        plan = make_plan(spec)
        self.assertEqual([len(f["train"]) for f in plan["folds"]], [2, 3])
        spec["mode"] = "by_file"
        self.assertEqual(len(make_plan(spec)["folds"]), 1)
        spec["by_file"]["validation"] = ["2019-01-23"]
        with self.assertRaisesRegex(ValueError, "strictly before"):
            make_plan(spec)

    def test_complete_two_fold_two_family_run_and_resume(self):
        spec = self.spec()
        spec["models"] = ["gcn", "gat", "recurrent_sparse_sthnn"]
        spec["training_overrides"]["num_workers"] = 1
        config = self.root / "pipeline.yaml"
        config.write_text(yaml.safe_dump(spec))
        command = [sys.executable, str(ROOT / "scripts/run_pipeline.py"), "--config", str(config)]
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        dry = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
        self.assertEqual(dry.returncode, 0, dry.stderr)
        self.assertFalse((self.root / "experiment").exists())
        self.assertFalse((self.root / "cache").exists())
        run = subprocess.run(command + ["--run", "--quiet-progress"], cwd=ROOT, env=env,
                             capture_output=True, text=True, timeout=180)
        self.assertEqual(run.returncode, 0, run.stdout[-5000:] + run.stderr[-5000:])
        output = self.root / "experiment"
        result = json.loads((output / "summary.json").read_text())
        self.assertEqual(len(result["per_run"]), 12)  # two rules, three models, two folds
        self.assertEqual(len(list((self.root / "cache/nmi").iterdir())), 3)
        self.assertEqual(len(list((self.root / "cache/graphs").iterdir())), 2)
        self.assertEqual(len(list((self.root / "cache/processed").iterdir())), 2)
        self.assertTrue((output / "complete.json").is_file())
        checkpoint = output / "fold_002/gat/seed_42/best.pt"
        evaluation = subprocess.run([sys.executable, str(ROOT / "scripts/evaluate.py"),
                                     "--checkpoint", str(checkpoint)], cwd=ROOT, env=env,
                                    capture_output=True, text=True, timeout=60)
        self.assertEqual(evaluation.returncode, 0, evaluation.stdout[-3000:] + evaluation.stderr[-3000:])
        with patch("remoma.pipeline.runner.subprocess.run", side_effect=AssertionError("No retraining on resume")):
            execute(spec, resume=True, progress=False)
        changed = deepcopy(spec)
        changed["seeds"] = [43]
        with self.assertRaisesRegex(ValueError, "changed"):
            execute(changed, resume=True, progress=False)
        # Also exercise the full estimator and fixed split, without training again.
        full = deepcopy(spec)
        full.update(mode="by_file", output_dir=str(self.root / "full_graphs"))
        full["nmi"]["method"] = "full"
        execute(full, stage="graphs", progress=False)
        self.assertTrue((self.root / "full_graphs/graphs_complete.json").is_file())

    def test_rolling_windows_and_sampling_contract(self):
        spec = self.spec()
        spec["walk_forward"]["train_window"] = 2
        folds = make_plan(spec)["folds"]
        self.assertEqual([d["date"] for d in folds[1]["train"]], ["2019-01-23", "2019-01-24"])
        spec["walk_forward"].update(test_files=2, step=1)
        with self.assertRaisesRegex(ValueError, "step"):
            make_plan(spec)
        spec["walk_forward"] = {"min_train_files": 2}
        altered = deepcopy(self.cfg)
        altered["data"]["n_lags"] = 3
        path = self.root / "different.yaml"
        path.write_text(yaml.safe_dump(altered))
        spec["model_configs"]["recurrent"] = str(path)
        with self.assertRaisesRegex(ValueError, "share lags"):
            make_plan(spec)

    def test_seed_aggregation_does_not_duplicate_test_samples(self):
        fold = {"name": "fold_001", "test": [{"date": "2019-01-28"}]}
        cm = np.array([[3, 1, 0], [1, 5, 1], [0, 2, 4]])
        records = [(fold, {"model": "gcn", "seed": seed, "samples": int(cm.sum()),
                           "rules": {"argmax": {"metrics": metrics_from_confusion(cm), "confusion_matrix": cm.tolist()}}})
                   for seed in (42, 43)]
        aggregate_results(records, self.root)
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual([r["samples"] for r in summary["pooled_folds_per_seed"]], [17, 17])
        self.assertEqual(summary["across_seeds"][0]["metrics"]["f1_macro"]["sample_std"], 0)


if __name__ == "__main__":
    unittest.main()
