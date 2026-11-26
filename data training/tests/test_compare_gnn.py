import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation import METRICS, RankedMetrics, bootstrap_users, choose_threshold
from compare_gnn import MODELS, compare_runs


class CompareGnnTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.directories = {name: self.root / name for name in MODELS}
        self.output = self.root / "comparison"
        self.config = {"bootstrap_replicates": 20, "bootstrap_seed": 21, "confidence_level": .8,
                       "cv_folds": 2, "cv_seed": 42, "minimum_validation_recall": .5}
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        users = np.repeat(np.arange(1, 5), 2)
        # Deliberately unequal rankings so a paired difference is nontrivial.
        score_sets = [
            [.9, .8, .7, .6, .5, .4, .3, .2],
            [.9, .6, .8, .4, .7, .5, .3, .2],
            [.9, .6, .7, .5, .8, .4, .3, .2],
            [.9, .5, .8, .4, .7, .6, .3, .2],
        ]
        self.samples = {}
        self.results = {}
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
                "config": dict(self.config), "n_samples": len(y), "n_users": len(universe),
                "threshold": threshold, "metrics": dict(zip(METRICS, point.tolist())), "intervals": intervals,
            }
            if name == "gnn":
                result.update(all_training_losses_finite=True, model_restore_verified=True)
            self.results[name] = result
            self.write_result(name)
            with (directory / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("row_id", "user_id", "movie_id", "y", "score", "prediction"))
                writer.writerows(zip(range(len(y)), users, np.arange(1, 9), y, scores, scores >= threshold))
            # CSV integer predictions, as saved by baseline_run.save_result.
            path = directory / "validation_predictions.csv"
            path.write_text(path.read_text().replace("True", "1").replace("False", "0"))
            np.savez(directory / "bootstrap.npz", samples=samples, metric_names=np.asarray(METRICS), user_ids=universe)
            if name != "movie_high_rating_rate":
                (directory / "cv_assignments.csv").write_text("row_id,fold\n10,0\n11,1\n12,0\n13,1\n")
            # This is invalid on purpose; the comparison must never load it.
            (directory / "test.npz").write_text("SEALED: must not read")

    def write_result(self, name):
        (self.directories[name] / "result.json").write_text(
            json.dumps(self.results[name], allow_nan=False), encoding="utf-8")

    def compare(self):
        return compare_runs(self.directories, self.output)

    def test_aligned_runs_dynamic_confidence_and_no_overwrite(self):
        report = self.compare()
        self.assertFalse(report["test_accessed"])
        self.assertEqual(report["primary_comparison"], "gnn_minus_logistic_regression")
        self.assertEqual(report["bootstrap_configuration"]["confidence_level"], .8)
        for other in MODELS[:-1]:
            differences = report["paired_differences"]["gnn_minus_" + other]
            for i, metric in enumerate(METRICS):
                delta = self.samples["gnn"][:, i] - self.samples[other][:, i]
                valid = delta[np.isfinite(delta)]
                self.assertAlmostEqual(differences[metric]["lower"], np.quantile(valid, .1))
                self.assertAlmostEqual(differences[metric]["upper"], np.quantile(valid, .9))
                self.assertEqual(differences[metric]["undefined_replicates"], 20 - len(valid))
        with (self.output / "comparison.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(MODELS) * len(METRICS))
        self.assertEqual({row["partition"] for row in rows}, {"validation"})
        before = (self.output / "paired_comparison.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.compare()
        self.assertEqual(before, (self.output / "paired_comparison.json").read_bytes())

    def test_misaligned_hash_rejected(self):
        self.results["gnn"]["input_sha256"]["train.npz"] = "b" * 64
        self.write_result("gnn")
        with self.assertRaisesRegex(ValueError, "hashes differ"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_changed_row_order_rejected(self):
        path = self.directories["gnn"] / "validation_predictions.csv"
        lines = path.read_text().splitlines()
        path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n")
        with self.assertRaisesRegex(ValueError, "Unpaired validation rows"):
            self.compare()

    def test_changed_cv_assignments_rejected(self):
        path = self.directories["gnn"] / "cv_assignments.csv"
        path.write_text("row_id,fold\n10,1\n11,0\n12,0\n13,1\n")
        with self.assertRaisesRegex(ValueError, "CV assignments differ"):
            self.compare()

    def test_unpaired_bootstrap_seed_rejected(self):
        self.results["gnn"]["config"]["bootstrap_seed"] += 1
        self.write_result("gnn")
        with self.assertRaisesRegex(ValueError, "Unpaired evaluation configuration"):
            self.compare()

    def test_bad_bootstrap_shape_rejected(self):
        np.savez(self.directories["gnn"] / "bootstrap.npz", samples=self.samples["gnn"][:-1],
                 metric_names=np.asarray(METRICS), user_ids=np.arange(1, 5))
        with self.assertRaisesRegex(ValueError, "sample shape"):
            self.compare()

    def test_nonvalidation_test_access_and_failed_training_rejected(self):
        for field, value in (("evaluation_partition", "test"), ("test_accessed", True),
                             ("all_training_losses_finite", False), ("model_restore_verified", False)):
            with self.subTest(field=field):
                original = self.results["gnn"][field]
                self.results["gnn"][field] = value
                self.write_result("gnn")
                with self.assertRaises(ValueError):
                    self.compare()
                self.results["gnn"][field] = original
        self.write_result("gnn")
        self.assertFalse(self.output.exists())

    def test_all_undefined_differences_written_as_json_null(self):
        samples = self.samples["gnn"].copy()
        samples[:, 0] = np.nan
        np.savez(self.directories["gnn"] / "bootstrap.npz", samples=samples,
                 metric_names=np.asarray(METRICS), user_ids=np.arange(1, 5))
        self.results["gnn"]["intervals"]["precision"] = {
            "lower": None, "upper": None, "valid_replicates": 0, "undefined_replicates": 20}
        self.write_result("gnn")
        report = self.compare()
        delta = report["paired_differences"]["gnn_minus_logistic_regression"]["precision"]
        self.assertIsNone(delta["lower"])
        self.assertIsNone(delta["upper"])
        self.assertEqual(delta["undefined_replicates"], 20)
        encoded = (self.output / "paired_comparison.json").read_text()
        self.assertNotIn("NaN", encoded)
        self.assertIn('"lower": null', encoded)


if __name__ == "__main__":
    unittest.main()
