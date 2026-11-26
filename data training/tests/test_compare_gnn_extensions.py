import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation import METRICS, RankedMetrics, bootstrap_users, choose_threshold
from compare_gnn_extensions import COUNTS, TABLE_NAMES, VARIANTS, compare_runs


class CompareGnnExtensionsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.extensions = self.root / "extensions"
        self.extensions.mkdir()
        self.paths = {"gnn": self.root / "original", "logistic_regression": self.root / "lr"}
        self.paths.update({name: self.extensions / name for name in VARIANTS})
        self.output = self.extensions / "comparison"
        self.config = {"bootstrap_replicates": 20, "bootstrap_seed": 21, "confidence_level": .8,
                       "cv_folds": 2, "cv_seed": 42, "minimum_validation_recall": .5,
                       "hidden_channels": 32, "learning_rate": .005, "weight_decay": .0001,
                       "batch_size": 256, "gradient_clip_norm": 10., "training_seed": 20261125,
                       "thread_limit": 4, "epoch_grid": [20, 40]}
        y = np.array([1, 0, 1, 0, 1, 1, 0, 0])
        users = np.repeat(np.arange(1, 5), 2)
        self.samples = {}
        self.results = {}
        scores = [
            [.9, .6, .8, .4, .7, .5, .3, .2],
            [.9, .8, .7, .6, .5, .4, .3, .2],
            [.9, .5, .8, .4, .7, .6, .3, .2],
            [.9, .6, .7, .5, .8, .4, .3, .2],
            [.9, .7, .8, .6, .5, .4, .3, .2],
        ]
        for name, values in zip(TABLE_NAMES, scores):
            directory = self.paths[name]
            directory.mkdir()
            score = np.asarray(values)
            threshold = choose_threshold(y, score, .5)
            point = RankedMetrics(y, score).evaluate(threshold)
            samples, intervals, universe = bootstrap_users(y, score, users, threshold, 20, 21, .8)
            self.samples[name] = samples
            result = {"model": "gnn_extensions" if name in VARIANTS else name,
                      "evaluation_partition": "validation", "test_accessed": False,
                      "input_sha256": {key: "a" * 64 for key in ("train.npz", "validation.npz", "manifest.json")},
                      "config": dict(self.config), "n_samples": len(y), "n_users": len(universe),
                      "threshold": threshold, "metrics": dict(zip(METRICS, point.tolist())),
                      "intervals": intervals, "cv_mean_AP": .75,
                      "fit_and_cv_seconds": 2.5, "evaluation_seconds": .2}
            if name != "logistic_regression":
                result.update(all_training_losses_finite=True, model_restore_verified=True,
                              graph_sha256="b" * 64, selected_parameters={"epochs": 40},
                              input_channels=["history_rating/5", "history_mask"], genre_used=False)
            if name == "gnn":
                result["selected_parameters"]["taps"] = 5
            if name in VARIANTS:
                result.update(variant_name=name, parameter_count=COUNTS[name], max_hops=4)
            self.results[name] = result
            self.write_result(name)
            with (directory / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("row_id", "user_id", "movie_id", "y", "score", "prediction"))
                writer.writerows(zip(range(len(y)), users, np.arange(1, 9), y, score, (score >= threshold).astype(int)))
            np.savez(directory / "bootstrap.npz", samples=samples, metric_names=np.asarray(METRICS), user_ids=universe)
            (directory / "cv_assignments.csv").write_text("row_id,fold\n10,0\n11,1\n12,0\n13,1\n", encoding="utf-8")
            # No dataset loader should inspect this deliberately invalid sealed artifact.
            (directory / "test.npz").write_text("SEALED: must not read", encoding="utf-8")
        self.write_preservation({"status": "PASS", "original_files_preserved": True})

    def write_result(self, name):
        (self.paths[name] / "result.json").write_text(json.dumps(self.results[name], allow_nan=False), encoding="utf-8")

    def write_preservation(self, report):
        (self.extensions / "preservation.json").write_text(json.dumps(report), encoding="utf-8")

    def compare(self, plot=False):
        if plot:
            return compare_runs(self.extensions, self.output, self.paths["gnn"], self.paths["logistic_regression"])
        with patch("compare_gnn_extensions._plot"):
            return compare_runs(self.extensions, self.output, self.paths["gnn"], self.paths["logistic_regression"])

    def test_aligned_runs_paired_intervals_no_mutation_no_overwrite(self):
        before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                  for directory in self.paths.values() for path in directory.iterdir()}
        report = self.compare()
        self.assertFalse(report["test_accessed"])
        self.assertTrue(report["exploratory"])
        self.assertFalse(report["multiple_comparisons_adjusted"])
        self.assertTrue(report["original_artifacts_preserved"])
        self.assertEqual(report["parameter_counts"]["gnn"], 385)
        self.assertEqual(report["training_target_count"], 4)
        for name in VARIANTS:
            for other in ("gnn", "logistic_regression"):
                values = report["paired_differences"][name + "_minus_" + other]
                for index, metric in enumerate(METRICS):
                    delta = self.samples[name][:, index] - self.samples[other][:, index]
                    valid = delta[np.isfinite(delta)]
                    self.assertAlmostEqual(values[metric]["lower"], np.quantile(valid, .1))
                    self.assertAlmostEqual(values[metric]["upper"], np.quantile(valid, .9))
        after = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                 for directory in self.paths.values() for path in directory.iterdir()}
        self.assertEqual(before, after)
        with (self.output / "comparison.csv").open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), len(TABLE_NAMES) * len(METRICS))
        self.assertEqual({row["partition"] for row in rows}, {"validation"})
        self.assertEqual({row["model"] for row in rows}, set(TABLE_NAMES))
        self.assertIn("80%", (self.output / "comparison.md").read_text(encoding="utf-8"))
        self.assertIn("multiple comparisons", (self.output / "paired_differences.md").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            self.compare()

    def test_hash_mismatch_rejected_before_writes(self):
        self.results["wide64"]["input_sha256"]["train.npz"] = "c" * 64
        self.write_result("wide64")
        with self.assertRaisesRegex(ValueError, "input hashes differ"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_validation_row_reordering_rejected(self):
        path = self.paths["wide128"] / "validation_predictions.csv"
        rows = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join([rows[0], *reversed(rows[1:])]) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Unpaired validation rows"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_cv_assignments_and_cv_seed_rejected(self):
        path = self.paths["deep2x32"] / "cv_assignments.csv"
        original = path.read_text(encoding="utf-8")
        path.write_text("row_id,fold\n10,1\n11,0\n12,0\n13,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "CV assignments differ"):
            self.compare()
        path.write_text(original, encoding="utf-8")
        self.results["deep2x32"]["config"]["cv_seed"] = 43
        self.write_result("deep2x32")
        with self.assertRaisesRegex(ValueError, "CV configuration"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_lr_pairing_is_also_required(self):
        self.results["logistic_regression"]["config"]["bootstrap_seed"] += 1
        self.write_result("logistic_regression")
        with self.assertRaisesRegex(ValueError, "evaluation or CV configuration"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_gnn_optimizer_configuration_change_rejected(self):
        self.results["wide64"]["config"]["learning_rate"] = .01
        self.write_result("wide64")
        with self.assertRaisesRegex(ValueError, "GNN training configuration differs"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_failed_graph_training_parameter_depth_and_variant_checks_rejected(self):
        changes = (("all_training_losses_finite", False), ("model_restore_verified", False),
                   ("graph_sha256", "c" * 64), ("parameter_count", 770),
                   ("max_hops", 5), ("variant_name", "unregistered"), ("genre_used", True))
        for field, value in changes:
            with self.subTest(field=field):
                original = self.results["wide64"][field]
                self.results["wide64"][field] = value
                self.write_result("wide64")
                with self.assertRaises(ValueError):
                    self.compare()
                self.results["wide64"][field] = original
                self.write_result("wide64")
        self.assertFalse(self.output.exists())

    def test_test_access_or_partition_rejected(self):
        for field, value in (("test_accessed", True), ("evaluation_partition", "test")):
            with self.subTest(field=field):
                original = self.results["wide128"][field]
                self.results["wide128"][field] = value
                self.write_result("wide128")
                with self.assertRaises(ValueError):
                    self.compare()
                self.results["wide128"][field] = original
                self.write_result("wide128")
        self.assertFalse(self.output.exists())

    def test_preservation_failure_rejected_and_missing_evidence_not_assumed(self):
        for report in ({"status": "FAIL", "original_files_preserved": False},
                       {"status": "PASS", "original_files_preserved": False}):
            self.write_preservation(report)
            with self.assertRaisesRegex(ValueError, "preservation check"):
                self.compare()
        (self.extensions / "preservation.json").unlink()
        self.assertIsNone(self.compare()["original_artifacts_preserved"])

    def test_contradictory_preservation_report_rejected(self):
        reports = [
            {"status": "PASS", "original_files_preserved": True, "test_accessed": True},
            {"status": "PASS", "original_files_preserved": True, "protected_file_count": 1,
             "before": {"model.pt": "a" * 64}, "after": {"model.pt": "b" * 64}},
        ]
        for report in reports:
            self.write_preservation(report)
            with self.assertRaises(ValueError):
                self.compare()
        self.assertFalse(self.output.exists())

    def test_all_undefined_metric_differences_are_json_null(self):
        name = "wide64"
        samples = self.samples[name].copy()
        samples[:, 0] = np.nan
        np.savez(self.paths[name] / "bootstrap.npz", samples=samples,
                 metric_names=np.asarray(METRICS), user_ids=np.arange(1, 5))
        self.results[name]["intervals"]["precision"] = {
            "lower": None, "upper": None, "valid_replicates": 0, "undefined_replicates": 20}
        self.write_result(name)
        report = self.compare()
        difference = report["paired_differences"]["wide64_minus_gnn"]["precision"]
        self.assertIsNone(difference["lower"])
        self.assertEqual(difference["undefined_replicates"], 20)
        self.assertNotIn("NaN", (self.output / "paired_differences.json").read_text(encoding="utf-8"))

    def test_refuses_output_inside_model_directory(self):
        self.output = self.paths["gnn"] / "comparison"
        with self.assertRaisesRegex(ValueError, "inside a model run"):
            self.compare()
        self.assertFalse(self.output.exists())

    def test_real_plots_have_png_and_pdf_artifacts(self):
        self.compare(plot=True)
        for stem in ("validation_pr_curves", "paired_precision_differences"):
            self.assertGreater((self.output / (stem + ".png")).stat().st_size, 1000)
            self.assertGreater((self.output / (stem + ".pdf")).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
