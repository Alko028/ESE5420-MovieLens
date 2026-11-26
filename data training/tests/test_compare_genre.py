import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation import METRICS, RankedMetrics, bootstrap_users, choose_threshold
from compare_genre import MODELS, PAIRS, compare_runs


class CompareGenreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directories = {name: self.root / name for name in MODELS}
        self.output = self.root / "comparison"
        shared = {"bootstrap_replicates": 20, "bootstrap_seed": 21, "confidence_level": .8,
                  "cv_folds": 2, "cv_seed": 42, "minimum_validation_recall": .5}
        lr = {**shared, "C_grid": [.01, .1, 1], "solver": "lbfgs", "max_iter": 3000, "tol": 1e-6}
        knn = {**shared, "n_neighbors_grid": [3, 5], "weights_grid": ["uniform", "distance"],
               "metric": "euclidean", "algorithm": "brute", "n_jobs": 1, "thread_limit": 4}
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        users = np.repeat(np.arange(1, 5), 2)
        score_sets = [
            [.9, .8, .7, .6, .5, .4, .3, .2],
            [.9, .6, .8, .4, .7, .5, .3, .2],
            [.9, .6, .7, .5, .8, .4, .3, .2],
            [.9, .5, .8, .4, .7, .6, .3, .2],
        ]
        self.samples, self.results = {}, {}
        for name, values in zip(MODELS, score_sets):
            directory = self.directories[name]
            directory.mkdir()
            scores = np.asarray(values)
            threshold = choose_threshold(y, scores, .5)
            point = RankedMetrics(y, scores).evaluate(threshold)
            samples, intervals, universe = bootstrap_users(y, scores, users, threshold, 20, 21, .8)
            self.samples[name] = samples
            result = {
                "model": name, "evaluation_partition": "validation", "test_accessed": False,
                "input_sha256": {key: "a" * 64 for key in ("train.npz", "validation.npz", "manifest.json")},
                "config": dict(lr if name.startswith("logistic") else knn),
                "n_samples": len(y), "n_users": len(universe), "threshold": threshold,
                "metrics": dict(zip(METRICS, point.tolist())), "intervals": intervals,
            }
            if name.endswith("_genre"):
                result.update(n_input_features=66, model_restore_verified=True,
                              base_features_unchanged=True, development_alignment_verified=True,
                              graph_sha256="b" * 64, genre_config_sha256="c" * 64,
                              genre_feature_definition={"base": 9, "genre": 57, "history_only": True})
            self.results[name] = result
            self.write_result(name)
            with (directory / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("row_id", "user_id", "movie_id", "y", "score", "prediction"))
                writer.writerows(zip(range(len(y)), users, np.arange(1, 9), y, scores,
                                     (scores >= threshold).astype(int)))
            np.savez(directory / "bootstrap.npz", samples=samples, metric_names=np.asarray(METRICS), user_ids=universe)
            (directory / "cv_assignments.csv").write_text("row_id,fold\n10,0\n11,1\n12,0\n13,1\n")
            # Invalid sentinel: loading any test file would fail.
            (directory / "test.npz").write_text("SEALED: must not read")

    def write_result(self, name):
        (self.directories[name] / "result.json").write_text(
            json.dumps(self.results[name], allow_nan=False), encoding="utf-8")

    def compare(self):
        return compare_runs(self.directories, self.output)

    def test_aligned_paired_intervals_points_and_no_overwrite(self):
        report = self.compare()
        self.assertFalse(report["test_accessed"])
        self.assertEqual(report["primary_comparisons"], ["lr_genre_minus_lr", "knn_genre_minus_knn"])
        self.assertTrue(report["within_method_configuration_unchanged"])
        self.assertEqual(report["bootstrap_configuration"]["confidence_level"], .8)
        self.assertEqual(report["training_target_count"], 4)
        for label, (left, right) in PAIRS.items():
            for i, metric in enumerate(METRICS):
                result = report["paired_differences"][label][metric]
                expected_point = self.results[left]["metrics"][metric] - self.results[right]["metrics"][metric]
                self.assertAlmostEqual(result["estimate"], expected_point)
                delta = self.samples[left][:, i] - self.samples[right][:, i]
                valid = delta[np.isfinite(delta)]
                self.assertAlmostEqual(result["lower"], np.quantile(valid, .1))
                self.assertAlmostEqual(result["upper"], np.quantile(valid, .9))
                self.assertEqual(result["undefined_replicates"], len(delta) - len(valid))
        with (self.output / "comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(MODELS) * len(METRICS))
        self.assertEqual({row["partition"] for row in rows}, {"validation"})
        for row in rows:
            self.assertEqual(float(row["threshold"]), self.results[row["model"]]["threshold"])
        before = (self.output / "paired_comparison.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.compare()
        self.assertEqual(before, (self.output / "paired_comparison.json").read_bytes())

    def test_changed_hash_or_row_order_rejected(self):
        result = self.results["knn_genre"]
        result["input_sha256"]["train.npz"] = "d" * 64
        self.write_result("knn_genre")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            self.compare()
        result["input_sha256"]["train.npz"] = "a" * 64
        self.write_result("knn_genre")
        path = self.directories["knn_genre"] / "validation_predictions.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
        with self.assertRaisesRegex(ValueError, "Unpaired validation rows"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_cv_changes_rejected_for_every_model(self):
        for name in MODELS:
            with self.subTest(name=name):
                path = self.directories[name] / "cv_assignments.csv"
                old = path.read_text()
                path.write_text("row_id,fold\n10,1\n11,0\n12,0\n13,1\n")
                with self.assertRaisesRegex(ValueError, "CV assignments differ"):
                    self.compare()
                path.write_text(old)
        self.assertFalse(self.output.exists())

    def test_original_grid_change_rejected(self):
        for name, key, value in (("logistic_regression", "C_grid", [.01, .1, 1, 10]),
                                 ("knn", "n_neighbors_grid", [3, 5, 7]),
                                 ("knn", "weights_grid", ["uniform"]),
                                 ("logistic_regression_genre", "solver", "liblinear")):
            with self.subTest(name=name, key=key):
                config = self.results[name]["config"]
                old = config[key]
                config[key] = value
                self.write_result(name)
                with self.assertRaisesRegex(ValueError, "HPO configuration differs"):
                    self.compare()
                config[key] = old
                self.write_result(name)
        self.assertFalse(self.output.exists())

    def test_genre_feature_metadata_and_checks_required(self):
        result = self.results["knn_genre"]
        for field, value in (("n_input_features", 65), ("model_restore_verified", False),
                             ("base_features_unchanged", False), ("development_alignment_verified", False),
                             ("graph_sha256", "d" * 64), ("genre_config_sha256", "d" * 64),
                             ("genre_feature_definition", ["different"]), ("graph_sha256", "bad")):
            with self.subTest(field=field):
                old = result[field]
                result[field] = value
                self.write_result("knn_genre")
                with self.assertRaises(ValueError):
                    self.compare()
                result[field] = old
                self.write_result("knn_genre")
        self.assertFalse(self.output.exists())

    def test_bootstrap_and_validation_contracts_required(self):
        result = self.results["knn_genre"]
        for key, value in (("bootstrap_seed", 22), ("minimum_validation_recall", .6), ("cv_seed", 43)):
            old = result["config"][key]
            result["config"][key] = value
            self.write_result("knn_genre")
            with self.assertRaisesRegex(ValueError, "Unpaired evaluation"):
                self.compare()
            result["config"][key] = old
        for field, value in (("evaluation_partition", "test"), ("test_accessed", True)):
            old = result[field]
            result[field] = value
            self.write_result("knn_genre")
            with self.assertRaises(ValueError):
                self.compare()
            result[field] = old
        self.write_result("knn_genre")
        for metric_names, users in ((np.asarray(METRICS[::-1]), np.arange(1, 5)),
                                    (np.asarray(METRICS), np.arange(2, 6))):
            np.savez(self.directories["knn_genre"] / "bootstrap.npz", samples=self.samples["knn_genre"],
                     metric_names=metric_names, user_ids=users)
            with self.assertRaises(ValueError):
                self.compare()
        self.assertFalse(self.output.exists())

    def test_undefined_difference_is_json_null(self):
        samples = self.samples["knn_genre"].copy()
        samples[:, 0] = np.nan
        np.savez(self.directories["knn_genre"] / "bootstrap.npz", samples=samples,
                 metric_names=np.asarray(METRICS), user_ids=np.arange(1, 5))
        self.results["knn_genre"]["intervals"]["precision"] = {
            "lower": None, "upper": None, "valid_replicates": 0, "undefined_replicates": 20}
        self.write_result("knn_genre")
        report = self.compare()
        for label in ("knn_genre_minus_knn", "knn_genre_minus_lr_genre", "knn_genre_minus_lr"):
            interval = report["paired_differences"][label]["precision"]
            self.assertIsNone(interval["lower"])
            self.assertIsNone(interval["upper"])
            self.assertEqual(interval["undefined_replicates"], 20)
        encoded = (self.output / "paired_comparison.json").read_text()
        self.assertNotIn("NaN", encoded)
        self.assertIn('"lower": null', encoded)


if __name__ == "__main__":
    unittest.main()
