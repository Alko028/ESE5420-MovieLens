"""Synthetic capacity/propagation/CV checks; no real test artifacts are opened."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("PyTorch is optional for the baseline-only environment") from exc
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import gnn
import gnn_extensions as extension
from common.deep_movie_gnn import TwoLayerMovieGNN, build_full_history_taps


def fixture():
    ratings = np.diag(np.array([5., 4., 3., 2.], dtype=np.float32))
    graph = {
        "history_rating": ratings,
        "history_mask": ratings > 0,
        "edge_index": np.array([[0, 2, 1, 2], [1, 1, 2, 3]], dtype=np.int64),
        "edge_weight": np.array([2., 1., 3., 1.], dtype=np.float32),
        "user_ids": np.arange(1, 5),
        "movie_ids": np.arange(1, 5),
    }
    pairs = np.array([(u, m) for u in range(4) for m in range(4) if u != m])
    batch = {
        "row_id": np.arange(len(pairs)),
        "user_index": pairs[:, 0],
        "movie_index": pairs[:, 1],
        "y": np.tile([0, 1], len(pairs) // 2).astype(np.int8),
    }
    return graph, batch


class ExtensionArchitectureTests(unittest.TestCase):
    def test_three_recorded_variants_have_expected_parameter_counts(self):
        graph, _ = fixture()
        adjacent = gnn.normalized_adjacency(graph)
        variants = extension.extension_configuration()["variants"]
        self.assertEqual([v["name"] for v in variants], ["wide64", "wide128", "deep2x32"])
        self.assertEqual([v["parameter_count"] for v in variants], [769, 1537, 3361])
        for variant in variants:
            model = extension.new_model(variant, adjacent, torch.device("cpu"))
            self.assertEqual(sum(p.numel() for p in model.parameters()), variant["parameter_count"])
            self.assertEqual(variant["max_hops"], 4)

    def test_full_history_cache_matches_shallow_target_cache(self):
        graph, batch = fixture()
        for taps in (3, 5):
            full = build_full_history_taps(graph, taps)
            gathered = full[batch["user_index"], batch["movie_index"]]
            np.testing.assert_array_equal(gathered, gnn.polynomial_features(graph, batch, taps))

    def test_second_layer_matches_dense_hidden_signal_propagation(self):
        graph, batch = fixture()
        adjacent = gnn.normalized_adjacency(graph)
        torch.manual_seed(18)
        model = TwoLayerMovieGNN(adjacent, taps=3, hidden_channels=4)
        full = torch.from_numpy(build_full_history_taps(graph, taps=3))
        users = torch.as_tensor(batch["user_index"], dtype=torch.long)
        movies = torch.as_tensor(batch["movie_index"], dtype=torch.long)
        hidden = torch.relu(model.first(full))
        dense = torch.from_numpy(adjacent.toarray())
        powers = [hidden]
        for _ in range(1, 3):
            hidden = torch.einsum("mn,unh->umh", dense, hidden)
            powers.append(hidden)
        expected = model.readout(torch.relu(model.second(torch.cat(powers, dim=-1)))).squeeze(-1)
        actual = model(full, users, movies)
        torch.testing.assert_close(actual, expected[users, movies], rtol=1e-6, atol=1e-7)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(actual, torch.as_tensor(batch["y"], dtype=torch.float32))
        loss.backward()
        for layer in (model.first, model.second, model.readout):
            self.assertTrue(torch.isfinite(layer.weight.grad).all())
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0.)


class ExtensionTrainingTests(unittest.TestCase):
    def test_heldout_label_changes_cannot_change_fit_weights_or_loss(self):
        graph, batch = fixture()
        adjacent = gnn.normalized_adjacency(graph)
        fit_rows, heldout_rows = np.arange(8), np.arange(8, 12)
        fit_batch = extension.subset(batch, fit_rows)
        first_heldout = extension.subset(batch, heldout_rows)
        second_heldout = {**first_heldout, "y": 1 - first_heldout["y"]}
        config = extension.extension_configuration()
        config.update(batch_size=4, thread_limit=1)
        variants = [
            {"name": "synthetic_wide", "depth": 1, "taps": 3, "hidden_channels": 4, "parameter_count": 33},
            {"name": "synthetic_deep", "depth": 2, "taps": 3, "hidden_channels": 4, "parameter_count": 85},
        ]
        for variant in variants:
            if variant["depth"] == 1:
                cache = gnn.polynomial_features(graph, batch, variant["taps"])
                fit_data, heldout_data = cache[fit_rows], cache[heldout_rows]
            else:
                fit_data = heldout_data = torch.from_numpy(build_full_history_taps(graph, variant["taps"]))
            results = [extension.fit_one(
                fit_batch, fit_data, variant, config, adjacent, torch.device("cpu"),
                seed=13, epochs=[1, 2], heldout=(heldout, heldout_data))
                for heldout in (first_heldout, second_heldout)]
            for name, value in results[0][0].state_dict().items():
                # Sparse adjacency is a fixed buffer, not a learned parameter.
                second = results[1][0].state_dict()[name]
                if value.is_sparse:
                    torch.testing.assert_close(value.to_dense(), second.to_dense(), rtol=0, atol=0)
                else:
                    torch.testing.assert_close(value, second, rtol=0, atol=0)
            for first, second in zip(results[0][2], results[1][2]):
                self.assertEqual(first["train_loss"], second["train_loss"])
                self.assertEqual(first["train_AP"], second["train_AP"])
            for result, heldout in zip(results, (first_heldout, second_heldout)):
                predictions = extension.predict(result[0], heldout_data, heldout, variant, torch.device("cpu"))
                self.assertEqual(result[1][2], average_precision_score(heldout["y"], predictions))

    def test_runner_reuses_seed42_folds_and_selects_epochs_inside_training(self):
        graph, _ = fixture()
        rng = np.random.default_rng(66)
        all_features = rng.uniform(size=(110, 10)).astype(np.float32)
        all_features[:, 0] = np.arange(110)
        train = {"row_id": np.arange(100), "user_index": np.arange(100) % 4,
                 "movie_index": (np.arange(100) + 1) % 4,
                 "y": np.tile([0, 1], 50), "X_raw": rng.normal(size=(100, 9))}
        validation = {"row_id": np.arange(100, 110), "user_index": np.arange(10) % 4,
                      "movie_index": (np.arange(10) + 1) % 4, "y": np.tile([0, 1], 5)}
        config = extension.extension_configuration()
        variant = config["variants"][0]
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(train["X_raw"], train["y"]))
        calls = []

        def fake_fit(batch, data, variant, config, adjacency, device, seed, epochs, heldout=None):
            calls.append((batch, np.array(data), tuple(epochs), heldout))
            gnn.seed_training(seed, 1)
            model = extension.new_model(variant, adjacency, device)
            scores = {epoch: .5 + epoch / 1000 for epoch in epochs} if heldout is not None else {}
            trace = [{"epoch": epoch, "train_loss": .7, "train_AP": .6,
                      "heldout_AP": scores.get(epoch)} for epoch in epochs]
            return model, scores, trace

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "outputs/gnn"
            expected.mkdir(parents=True)
            fold_ids = np.empty(100, dtype=np.int8)
            for fold, (_, heldout) in enumerate(folds):
                fold_ids[heldout] = fold
            extension.write_csv(expected / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_ids))
            with patch.object(extension, "BASE", root), \
                 patch.object(extension, "sha256", return_value="synthetic"), \
                 patch.object(extension.gnn, "polynomial_features", side_effect=lambda graph, batch, taps: all_features[batch["row_id"]]), \
                 patch.object(extension, "fit_one", side_effect=fake_fit), \
                 patch.object(extension, "save_result", side_effect=lambda *args: args[-1]):
                result = extension.run_variant(variant, config, train, validation, graph, root / "new", torch.device("cpu"))
            self.assertEqual(result["selected_parameters"], {"epochs": 40})
        self.assertEqual(len(calls), 6)
        for fold, ((batch, data, epochs, heldout), (fit_rows, heldout_rows)) in enumerate(zip(calls[:-1], folds)):
            np.testing.assert_array_equal(batch["row_id"], train["row_id"][fit_rows])
            np.testing.assert_array_equal(data, all_features[fit_rows])
            np.testing.assert_array_equal(heldout[0]["row_id"], train["row_id"][heldout_rows])
            np.testing.assert_array_equal(heldout[1], all_features[heldout_rows])
            self.assertEqual(len(set(batch["row_id"]) & set(heldout[0]["row_id"])), 0)
            self.assertEqual(epochs, (20, 40))
        self.assertIsNone(calls[-1][3])
        self.assertEqual(calls[-1][2], (40,))
        np.testing.assert_array_equal(calls[-1][0]["row_id"], train["row_id"])


class ExtensionSafetyTests(unittest.TestCase):
    def test_preservation_snapshot_never_hashes_test_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "outputs/gnn/result.json"
            original.parent.mkdir(parents=True)
            original.write_text("original", encoding="utf-8")
            preprocessed = root / "outputs/preprocessed"
            accessed = []

            def track_hash(path):
                path = Path(path)
                self.assertNotIn("sealed", path.parts)
                self.assertNotEqual(path.name, "test_inputs.npz")
                accessed.append(path)
                return "synthetic"

            with patch.object(extension, "BASE", root), \
                 patch.object(extension, "DEFAULT_OUTPUT", preprocessed), \
                 patch.object(extension, "sha256", side_effect=track_hash):
                snapshot = extension.protected_files()
            self.assertIn("outputs/gnn/result.json", snapshot)
            self.assertIn(preprocessed / "graph.npz", accessed)

    def test_audit_guard_blocks_only_synthetic_test_paths_in_subprocess(self):
        # Audit hooks cannot be removed; isolate the hook from this test process.
        code = r'''
import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import gnn_extensions as extension
root = Path(sys.argv[2])
extension.DEFAULT_OUTPUT = root
opened = extension.install_test_guard()
assert (root / "train.npz").read_bytes() == b"allowed"
blocked = []
for relative in ("test_inputs.npz", "sealed/test_labels.npz"):
    try:
        (root / relative).read_bytes()
    except PermissionError:
        blocked.append(relative)
assert len(blocked) == 2
assert sorted(opened) == ["train.npz"]
print(json.dumps({"blocked": blocked, "opened": sorted(opened)}))
'''
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "sealed").mkdir()
            (root / "train.npz").write_bytes(b"allowed")
            (root / "test_inputs.npz").write_bytes(b"synthetic")
            (root / "sealed/test_labels.npz").write_bytes(b"synthetic")
            result = subprocess.run([sys.executable, "-c", code, str(BASE), str(root)], capture_output=True,
                                    text=True, timeout=60, check=True)
        evidence = json.loads(result.stdout.strip())
        self.assertEqual(evidence["blocked"], ["test_inputs.npz", "sealed/test_labels.npz"])


if __name__ == "__main__":
    unittest.main()
