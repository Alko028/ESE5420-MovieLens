import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_evaluation as final


def synthetic_evaluation():
    y = np.tile([0, 1], 6)
    batch = {"row_id": np.arange(12), "user_id": np.repeat(np.arange(1, 7), 2),
             "movie_id": np.arange(1, 13), "y": y,
             "user_index": np.repeat(np.arange(6), 2), "movie_index": np.arange(12)}
    models = [{"id": name, "label": name, "role": "primary" if name == "a" else "baseline",
               "threshold": .5, "selected_parameters": {}} for name in ("a", "b")]
    protocol = {"models": models, "comparisons": [{"id": "a_minus_b", "left": "a", "right": "b", "role": "primary"}],
                "evaluation": {"bootstrap_replicates": 20, "bootstrap_seed": 21, "confidence_level": .8},
                "interpretation": "synthetic fixed protocol", "gnn_device": "cpu"}
    scores = {"a": np.array([.2, .8, .3, .7, .6, .9, .2, .6, .9, .7, .4, .9]),
              "b": np.array([.7, .8, .2, .6, .3, .9, .1, .7, .8, .9, .6, .8])}
    return batch, protocol, scores


class FrozenInferenceTests(unittest.TestCase):
    def test_logistic_restores_weights_and_scaler_without_fitting_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = np.asarray(["a", "b"])
            np.savez(root / "model.npz", mean=[1., 2.], scale=[2., 1.], coef=[[1., -.5]],
                     intercept=[.2], classes=[0, 1], feature_names=names, C=np.asarray(.1))
            batch = {"row_id": np.arange(3), "X_raw": np.array([[1., 2.], [3., 2.], [1., 4.]]),
                     "feature_names": names, "y": np.array([0, 1, 1])}
            spec = {"id": "lr", "kind": "logistic", "run_directory": ".", "model_file": "model.npz",
                    "genre": False, "selected_parameters": {"C": .1}}
            with patch.object(final, "BASE", root), patch.object(StandardScaler, "fit", side_effect=AssertionError("No scaler fit")), \
                    patch.object(LogisticRegression, "fit", side_effect=AssertionError("No LR fit")):
                values = final.predict_frozen(spec, batch, {}, "cpu")
                batch["y"] = 1 - batch["y"]
                np.testing.assert_array_equal(values, final.predict_frozen(spec, batch, {}, "cpu"))
            from scipy.special import expit
            np.testing.assert_allclose(values, expit([.2, 1.2, -.8]), rtol=0, atol=1e-15)

    def test_knn_rebuilds_only_serialized_training_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = np.asarray(["a"])
            fitting, labels = np.array([[0.], [2.], [4.]]), np.array([0, 1, 1])
            np.savez(root / "model.npz", mean=[0.], scale=[1.], X_fit=fitting, y_fit=labels,
                     feature_names=names, n_neighbors=1, weights="distance", metric="euclidean")
            batch = {"row_id": np.arange(2), "X_raw": np.array([[0.], [3.9]]), "feature_names": names, "y": [1, 0]}
            spec = {"id": "knn", "kind": "knn", "run_directory": ".", "model_file": "model.npz",
                    "genre": False, "selected_parameters": {"n_neighbors": 1, "weights": "distance"}}
            original, calls = KNeighborsClassifier.fit, []
            def track(estimator, X, y):
                calls.append((np.array(X), np.array(y)))
                return original(estimator, X, y)
            with patch.object(final, "BASE", root), patch.object(StandardScaler, "fit", side_effect=AssertionError("No scaler fit")), \
                    patch.object(KNeighborsClassifier, "fit", track):
                np.testing.assert_array_equal(final.predict_frozen(spec, batch, {}, "cpu"), [0., 1.])
            np.testing.assert_array_equal(calls[0][0], fitting)
            np.testing.assert_array_equal(calls[0][1], labels)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch not installed")
    def test_actual_gnn_checkpoint_inference_is_label_independent(self):
        import gnn
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "graph.npz").write_bytes(b"synthetic graph hash sentinel")
            rating = np.asarray([[5, 0, 0], [0, 4, 3]], dtype=np.float32)
            graph = {"history_rating": rating, "history_mask": rating > 0,
                     "edge_index": np.array([[0, 1, 1, 2], [1, 0, 2, 1]]), "edge_weight": np.ones(4)}
            model = gnn.ShallowMovieGNN(2, 3)
            gnn.torch.save({"state_dict": model.state_dict(), "graph_sha256": final.sha256(root / "graph.npz")}, root / "model.pt")
            batch = {"row_id": np.arange(2), "user_index": np.array([0, 1]), "movie_index": np.array([1, 0]), "y": [0, 1]}
            spec = {"id": "gnn", "kind": "shallow_gnn", "run_directory": ".", "model_file": "model.pt",
                    "selected_parameters": {"taps": 2, "epochs": 1}, "hidden_channels": 3}
            with patch.object(final, "BASE", root), patch.object(final, "DATA_ROOT", root):
                first = final.predict_frozen(spec, batch, graph, "cpu")
                batch["y"] = [1, 0]
                np.testing.assert_array_equal(first, final.predict_frozen(spec, batch, graph, "cpu"))


class FrozenMetricTests(unittest.TestCase):
    def test_fixed_threshold_is_not_adjusted_to_test_recall(self):
        batch, protocol, scores = synthetic_evaluation()
        protocol["models"][0]["threshold"] = .99
        summary, _ = final.evaluate_predictions(batch, scores, protocol)
        result = summary["results"]["a"]
        self.assertEqual(result["threshold"], .99)
        self.assertEqual(result["metrics"]["recall"], 0.)
        self.assertTrue(np.isnan(result["metrics"]["precision"]))
        self.assertEqual(result["intervals"]["precision"]["undefined_replicates"], 20)
        self.assertIsNone(final.clean(result)["metrics"]["precision"])

    def test_metrics_and_user_bootstrap_match_independent_sklearn(self):
        batch, protocol, scores = synthetic_evaluation()
        summary, bootstrap = final.evaluate_predictions(batch, scores, protocol)
        users, inverse = np.unique(batch["user_id"], return_inverse=True)
        for name, values in scores.items():
            predicted = values >= .5
            expected = [precision_score(batch["y"], predicted), recall_score(batch["y"], predicted), average_precision_score(batch["y"], values)]
            np.testing.assert_allclose(list(summary["results"][name]["metrics"].values()), expected)
            rng = np.random.default_rng(21)
            for i in range(20):
                weights = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))[inverse]
                expected = [precision_score(batch["y"], predicted, sample_weight=weights),
                            recall_score(batch["y"], predicted, sample_weight=weights),
                            average_precision_score(batch["y"], values, sample_weight=weights)]
                np.testing.assert_allclose(bootstrap[name][i], expected, rtol=0, atol=1e-14)
        self.assertEqual(summary["prevalence"]["positive_share"], .5)
        self.assertEqual(summary["prevalence"]["all_positive_recall"], 1.)

    def test_paired_intervals_are_draw_differences_not_subtracted_margins(self):
        batch, protocol, scores = synthetic_evaluation()
        summary, bootstrap = final.evaluate_predictions(batch, scores, protocol)
        for i, metric in enumerate(final.METRICS):
            delta = bootstrap["a"][:, i] - bootstrap["b"][:, i]
            interval = summary["paired_differences"]["a_minus_b"]["metrics"][metric]
            self.assertAlmostEqual(interval["lower"], np.quantile(delta, .1))
            self.assertAlmostEqual(interval["upper"], np.quantile(delta, .9))

    def test_missing_models_duplicate_rows_and_bad_shapes_are_rejected(self):
        batch, protocol, scores = synthetic_evaluation()
        with self.assertRaises(ValueError):
            final.evaluate_predictions(batch, {"a": scores["a"]}, protocol)
        batch["row_id"][0] = batch["row_id"][1]
        with self.assertRaises(ValueError):
            final.evaluate_predictions(batch, scores, protocol)


class FrozenAccessTests(unittest.TestCase):
    def test_guard_blocks_only_synthetic_test_until_enabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sealed").mkdir()
            for name in ("train.npz", "test_inputs.npz", "sealed/test_labels.npz"):
                (root / name).write_bytes(b"synthetic sentinel")
            state = final.install_access_guard(root)
            (root / "train.npz").read_bytes()
            for name in ("test_inputs.npz", "sealed/test_labels.npz"):
                with self.assertRaises(PermissionError):
                    (root / name).read_bytes()
            state["allow_test"] = True
            (root / "sealed/test_labels.npz").read_bytes()
            self.assertEqual(state["blocked_test_events"], 2)
            self.assertEqual(state["test_open_events"], 1)

    def test_changed_protocol_hash_rejected_before_any_data_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text('{"status":"frozen"}', encoding="utf-8")
            (root / "protocol.sha256").write_text("0" * 64, encoding="ascii")
            with self.assertRaisesRegex(ValueError, "protocol hash changed"):
                final.verify_frozen(root)

    def test_provenance_failure_cannot_consume_or_load_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(final, "verify_frozen", side_effect=ValueError("source changed")), \
                    patch.object(final, "load_final_test") as loader:
                with self.assertRaises(ValueError):
                    final.run_test(root, root / "results")
                loader.assert_not_called()
            self.assertFalse((root / "test_access_started.json").exists())

    def test_exclusive_marker_blocks_second_run_even_with_new_output(self):
        batch, protocol, scores = synthetic_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("synthetic protocol", encoding="utf-8")
            graph = {"history_mask": np.zeros((6, 12), dtype=bool)}
            with patch.object(final, "verify_frozen", return_value=protocol), patch.object(final, "load_graph", return_value=graph), \
                    patch.object(final, "load_final_test", return_value=batch) as loader, \
                    patch.object(final, "predict_frozen", side_effect=lambda spec, *args: scores[spec["id"]]):
                summary = final.run_test(root, root / "results")
                self.assertEqual(summary["n_samples"], 12)
                loader.assert_called_once()
                with self.assertRaises(FileExistsError):
                    final.run_test(root, root / "alternate_results")
                loader.assert_called_once()
            self.assertTrue((root / "test_access_started.json").exists())
            saved = json.loads((root / "results/results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["partition"], "test")

    def test_post_access_failure_keeps_consumed_marker(self):
        batch, protocol, _ = synthetic_evaluation()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "protocol.json").write_text("synthetic protocol", encoding="utf-8")
            graph = {"history_mask": np.zeros((6, 12), dtype=bool)}
            with patch.object(final, "verify_frozen", return_value=protocol), patch.object(final, "load_graph", return_value=graph), \
                    patch.object(final, "load_final_test", return_value=batch), patch.object(final, "predict_frozen", side_effect=RuntimeError("synthetic failure")):
                with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                    final.run_test(root, root / "results")
            self.assertTrue((root / "test_failure.json").exists())
            with self.assertRaises(FileExistsError):
                final.run_test(root, root / "alternate_results")

    def test_freeze_refuses_overwrite_before_loading_anything(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(final, "collect_models") as collect:
            with self.assertRaises(FileExistsError):
                final.freeze(Path(directory))
            collect.assert_not_called()


if __name__ == "__main__":
    unittest.main()
