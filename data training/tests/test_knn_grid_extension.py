import copy
import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import knn_grid_extension as extension
from compare_knn_grid_extension import PAIRS, paired_differences, validate_alignment


class GridExtensionTests(unittest.TestCase):
    def test_bounded_symmetric_config_preserves_old_settings(self):
        original = extension.knn_configuration()
        config, plan = extension.extension_configuration()
        self.assertEqual(config["n_neighbors_grid"], [5, 15, 31, 63, 127, 255, 511, 1023])
        self.assertEqual({k: v for k, v in config.items() if k != "n_neighbors_grid"},
                         {k: v for k, v in original.items() if k != "n_neighbors_grid"})
        self.assertEqual(original["n_neighbors_grid"], [5, 15, 31, 63, 127])
        self.assertIn("One bounded expansion", plan["stop_rule"])

    def test_real_fold_scaler_checks_and_CV_selection(self):
        rng = np.random.default_rng(20)
        train = {"X_raw": rng.normal(size=(100, 9)), "y": np.tile([0, 1], 50)}
        config, _ = extension.extension_configuration()
        config["n_neighbors_grid"] = [3, 5]
        model, selected, _, summary, folds, checked = extension.fit_with_checks(train, config)
        self.assertEqual(len(checked), 21)
        self.assertEqual([row["n_fit"] for row in checked], [80] * 20 + [100])
        self.assertEqual(set(folds), set(range(5)))
        best = max(summary, key=lambda r: (r[2], -r[0], r[1] == "uniform"))
        self.assertEqual(selected, {"n_neighbors": best[0], "weights": best[1]})
        np.testing.assert_allclose(model[0].mean_, train["X_raw"].mean(axis=0))

    def test_scaling_on_all_rows_in_a_fold_is_rejected(self):
        train = {"X_raw": np.arange(900).reshape(100, 9), "y": np.tile([0, 1], 50)}
        config, _ = extension.extension_configuration()
        config["n_neighbors_grid"] = [3]
        def leaked_fit(*args):
            StandardScaler().fit(train["X_raw"].astype(float), train["y"])
        with patch.object(extension, "tune_and_fit", side_effect=leaked_fit):
            with self.assertRaises(AssertionError):
                extension.fit_with_checks(train, config)

    def test_original_grid_reproduction_check_rejects_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.csv"
            path.write_text("k,weights,mean_AP,fold_AP_std\n5,distance,0.8,0.01\n", encoding="utf-8")
            self.assertEqual(extension.legacy_cv_check([[5, "distance", .8, .01]], path)[0][2:], [.8, .8])
            with self.assertRaises(AssertionError):
                extension.legacy_cv_check([[5, "distance", .81, .01]], path)

    def test_synthetic_test_access_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed").mkdir()
            for relative in ("train.npz", "test_inputs.npz", "sealed/test_labels.npz"):
                (root / relative).write_bytes(b"synthetic sentinel")
            with patch.object(extension, "DEFAULT_OUTPUT", root):
                opened = extension.install_test_guard()
                self.assertTrue((root / "train.npz").read_bytes())
                for relative in ("test_inputs.npz", "sealed/test_labels.npz"):
                    with self.assertRaises(PermissionError):
                        (root / relative).read_bytes()
            self.assertEqual(opened, {"train.npz"})


class ExpandedComparisonTests(unittest.TestCase):
    def setUp(self):
        shared = {"bootstrap_replicates": 20, "bootstrap_seed": 21, "confidence_level": .8,
                  "cv_folds": 5, "cv_seed": 42, "minimum_validation_recall": .5}
        old = {**shared, "n_neighbors_grid": [5, 15, 31, 63, 127], "weights_grid": ["uniform", "distance"],
               "metric": "euclidean", "algorithm": "brute", "n_jobs": 1, "thread_limit": 4}
        new = {**old, "n_neighbors_grid": old["n_neighbors_grid"] + [255, 511, 1023]}
        rng = np.random.default_rng(10)
        self.runs = {}
        for i, name in enumerate(("lr", "old_history", "new_history", "old_genre", "new_genre")):
            result = {"config": copy.deepcopy(shared if name == "lr" else new if name.startswith("new") else old),
                      "input_sha256": {"train.npz": "a" * 64},
                      "metrics": {"precision": .7 + i/100, "recall": .5, "AP": .8}}
            if name.endswith("genre"):
                result.update(graph_sha256="b" * 64, genre_config_sha256="c" * 64,
                              genre_feature_definition={"history_only": True}, feature_matrix_sha256={"train": "d" * 64},
                              base_features_unchanged=True, development_alignment_verified=True)
            if name.startswith("new"):
                result.update(n_input_features=66 if name.endswith("genre") else 9,
                              representation="genre" if name.endswith("genre") else "history",
                              model_restore_verified=True, legacy_CV_grid_reproduced=True, actual_scaler_fits_verified=81)
            self.runs[name] = {"root": Path(name), "keys": [(0, 1, 1, 1)], "result": result,
                               "bootstrap": {"metric_names": np.asarray(["precision", "recall", "AP"]),
                                             "user_ids": np.arange(5), "samples": rng.uniform(.5, .9, (20, 3))}}
        self.folds = [{"row_id": str(i), "fold": str(i % 5)} for i in range(10)]

    def test_pairing_and_paired_intervals_use_draw_differences(self):
        with patch("compare_knn_grid_extension._read_csv", return_value=self.folds):
            self.assertEqual(validate_alignment(self.runs), 10)
        evidence = paired_differences(self.runs)
        for label, (left, right) in PAIRS.items():
            for i, metric in enumerate(("precision", "recall", "AP")):
                delta = self.runs[left]["bootstrap"]["samples"][:, i] - self.runs[right]["bootstrap"]["samples"][:, i]
                self.assertAlmostEqual(evidence[label][metric]["lower"], np.quantile(delta, .1))
                self.assertAlmostEqual(evidence[label][metric]["upper"], np.quantile(delta, .9))

    def test_changed_grid_or_genre_provenance_rejected(self):
        for modify in (lambda r: r["new_genre"]["result"]["config"].update(n_neighbors_grid=[255]),
                       lambda r: r["new_genre"]["result"].update(graph_sha256="e" * 64)):
            runs = copy.deepcopy(self.runs)
            modify(runs)
            with patch("compare_knn_grid_extension._read_csv", return_value=self.folds):
                with self.assertRaises(ValueError):
                    validate_alignment(runs)

    def test_changed_CV_assignments_or_validation_rows_rejected(self):
        def changed(path, required):
            return self.folds[::-1] if path.parent.name == "new_genre" else self.folds
        with patch("compare_knn_grid_extension._read_csv", side_effect=changed):
            with self.assertRaisesRegex(ValueError, "fold assignments differ"):
                validate_alignment(self.runs)
        self.runs["new_history"]["keys"] = [(0, 2, 1, 1)]
        with patch("compare_knn_grid_extension._read_csv", return_value=self.folds):
            with self.assertRaisesRegex(ValueError, "Unpaired validation"):
                validate_alignment(self.runs)


if __name__ == "__main__":
    unittest.main()
