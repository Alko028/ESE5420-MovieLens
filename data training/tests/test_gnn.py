"""Synthetic checks for the history-only shallow movie-graph classifier.

These fixtures contain no MovieLens outcomes and never load sealed test files.
"""

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Optional GNN tests require requirements-gnn.txt / PyTorch") from exc

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

import gnn
from gnn import ShallowMovieGNN, normalized_adjacency, polynomial_features


def directed_fixture():
    """A deliberately asymmetric graph with one completely isolated movie."""
    ratings = np.array([[5., 0., 1., 0.], [0., 4., 0., 2.]], dtype=np.float32)
    graph = {
        # A[target, source]: movie 1 receives from 0 and 2; movie 2 from 1.
        "edge_index": np.array([[0, 2, 1], [1, 1, 2]], dtype=np.int64),
        "edge_weight": np.array([2., 1., 3.], dtype=np.float32),
        "history_rating": ratings,
        "history_mask": ratings > 0,
        "user_ids": np.arange(1, 3),
        "movie_ids": np.arange(1, 5),
    }
    user_index = np.array([0, 0, 1, 1], dtype=np.int64)
    movie_index = np.array([1, 3, 0, 2], dtype=np.int64)
    batch = {
        "row_id": np.arange(len(user_index)),
        "user_index": user_index,
        "movie_index": movie_index,
        "user_id": user_index + 1,
        "movie_id": movie_index + 1,
    }
    return graph, batch


class GraphSignalTests(unittest.TestCase):
    def test_directed_normalization_and_isolated_rows(self):
        graph, _ = directed_fixture()
        adjacency = normalized_adjacency(graph)
        expected = np.zeros((4, 4), dtype=np.float32)
        expected[1, 0], expected[1, 2], expected[2, 1] = 2 / 3, 1 / 3, 1
        np.testing.assert_allclose(adjacency.toarray(), expected, atol=1e-7)
        np.testing.assert_allclose(np.asarray(adjacency.sum(axis=1)).ravel(), [0, 1, 1, 0])
        self.assertTrue(np.isfinite(adjacency.data).all())

    def test_hand_calculated_signal_order_and_propagation(self):
        graph, batch = directed_fixture()
        actual = polynomial_features(graph, batch, taps=3)
        # Interleaved [rating/5, mask] for k=0, then k=1, then k=2.
        # User 0 -> target 1 receives (2*1 + .2)/3 = 11/15, mask = 1.
        # User 1 -> target 2 receives its observed movie-1 rating .8, mask = 1.
        expected = np.array([
            [0, 0, 11 / 15, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, .8, 1, 0, 0],
        ], dtype=np.float32)
        self.assertEqual(actual.shape, (4, 6))
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_target_already_in_history_is_rejected(self):
        graph, batch = directed_fixture()
        contaminated = {key: value[:1].copy() for key, value in batch.items()}
        contaminated["movie_index"][:] = 0
        contaminated["movie_id"][:] = 1
        with self.assertRaises(ValueError):
            polynomial_features(graph, contaminated, taps=3)

    def test_target_labels_cannot_change_graph_inputs(self):
        graph, batch = directed_fixture()
        first = polynomial_features(graph, {**batch, "y": np.zeros(4)}, taps=3)
        second = polynomial_features(graph, {**batch, "y": np.ones(4)}, taps=3)
        np.testing.assert_array_equal(first, second)


class ShallowModelTests(unittest.TestCase):
    def test_target_local_logits_match_full_graph_convolution(self):
        graph, batch = directed_fixture()
        taps, hidden = 3, 4
        adjacency = normalized_adjacency(graph).toarray()
        signal = np.stack([graph["history_rating"] / 5,
                           graph["history_mask"].astype(np.float32)], axis=1)
        # Reference full-graph implementation, independently using dense matmul.
        powers = []
        for _ in range(taps):
            powers.append(signal.transpose(0, 2, 1))  # user, movie, channel
            signal = signal @ adjacency.T
        full_features = np.concatenate(powers, axis=2).astype(np.float32)
        torch.manual_seed(37)
        model = ShallowMovieGNN(taps, hidden_channels=hidden).eval()
        with torch.no_grad():
            full_logits = model.readout(torch.relu(model.hidden(torch.from_numpy(full_features)))).squeeze(-1)
            expected = full_logits[batch["user_index"], batch["movie_index"]]
            actual = model(torch.from_numpy(polynomial_features(graph, batch, taps)))
        torch.testing.assert_close(actual.reshape(-1), expected.reshape(-1), rtol=1e-6, atol=1e-7)

    def test_finite_gradients_and_portable_state_restore(self):
        graph, batch = directed_fixture()
        features = torch.from_numpy(polynomial_features(graph, batch, taps=3))
        labels = torch.tensor([1., 0., 0., 1.])
        torch.manual_seed(12)
        model = ShallowMovieGNN(3, hidden_channels=4)
        before = [parameter.detach().clone() for parameter in model.parameters()]
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(features).reshape(-1), labels)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, model.parameters())))
        stored = io.BytesIO()
        torch.save(model.state_dict(), stored)
        stored.seek(0)
        restored = ShallowMovieGNN(3, hidden_channels=4)
        restored.load_state_dict(torch.load(stored, map_location="cpu", weights_only=True))
        with torch.no_grad():
            torch.testing.assert_close(model(features), restored(features), rtol=0, atol=0)


class TrainingProtocolTests(unittest.TestCase):
    def test_training_only_cv_uses_lr_folds_and_refits_selected_epochs(self):
        rng = np.random.default_rng(29)
        train = {"y": np.tile([0, 1], 50), "row_id": np.arange(100) + 500,
                 "X_raw": rng.normal(size=(100, 9)).astype(np.float32)}
        config = gnn.gnn_configuration()
        config.update(taps_grid=[2, 3], epoch_grid=[1, 2], hidden_channels=4,
                      batch_size=16, thread_limit=1)
        features = {}
        for taps in config["taps_grid"]:
            features[taps] = rng.normal(size=(100, taps * 2)).astype(np.float32)
            features[taps][:, 0] = np.arange(100)  # row identity for the fit spy
        calls = []

        def fit_spy(X, y, taps, config, device, seed, checkpoints, heldout=None):
            calls.append((taps, np.array(X), np.array(y), tuple(checkpoints), heldout))
            torch.manual_seed(seed)
            model = ShallowMovieGNN(taps, hidden_channels=config["hidden_channels"])
            # Synthetic CV scores deliberately choose taps=3 and epoch=2.
            scores = {epoch: .5 + taps / 100 + epoch / 1000 for epoch in checkpoints}
            history = [{"epoch": epoch, "train_loss": .7,
                        "heldout_AP": scores.get(epoch) if heldout is not None else None}
                       for epoch in range(1, max(checkpoints) + 1)]
            return model, scores if heldout is not None else {}, history

        # Tuning must consume supplied training arrays, without loading any dataset.
        with patch.object(gnn, "train_one", side_effect=fit_spy), \
             patch.object(gnn, "load_development", side_effect=AssertionError("Unexpected data access")), \
             patch.object(gnn, "load_graph", side_effect=AssertionError("Unexpected graph access")):
            _, selected, records, summaries, fold_id, _ = gnn.tune_and_fit(
                train, features, config, torch.device("cpu"))
        expected_folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(
            train["X_raw"], train["y"]))
        self.assertEqual(len(calls), 11)  # one curve per tap/fold, then final fit
        self.assertEqual(len(records), 20)  # two checkpoint epochs per CV curve
        self.assertEqual(len(summaries), 4)
        self.assertEqual(selected, {"taps": 3, "epochs": 2})
        for i, (_, heldout_rows) in enumerate(expected_folds):
            np.testing.assert_array_equal(fold_id[heldout_rows], i)
        for call_index, (taps, X, y, checkpoints, heldout) in enumerate(calls[:-1]):
            fit_rows, heldout_rows = expected_folds[call_index % 5]
            np.testing.assert_array_equal(X, features[taps][fit_rows])
            np.testing.assert_array_equal(y, train["y"][fit_rows])
            self.assertEqual(checkpoints, (1, 2))
            np.testing.assert_array_equal(heldout[0], features[taps][heldout_rows])
            np.testing.assert_array_equal(heldout[1], train["y"][heldout_rows])
            self.assertEqual(len(set(X[:, 0]) & set(heldout[0][:, 0])), 0)
        final_taps, final_X, final_y, final_checkpoints, final_heldout = calls[-1]
        self.assertEqual(final_taps, selected["taps"])
        np.testing.assert_array_equal(final_X, features[final_taps])
        np.testing.assert_array_equal(final_y, train["y"])
        self.assertEqual(final_checkpoints, (selected["epochs"],))
        self.assertIsNone(final_heldout)

    def test_seeded_training_repeats_exactly_on_cpu(self):
        rng = np.random.default_rng(97)
        X = rng.normal(size=(24, 4)).astype(np.float32)
        y = np.tile([0, 1], 12).astype(np.float32)
        config = gnn.gnn_configuration()
        config.update(hidden_channels=4, batch_size=8, thread_limit=1)
        results = [gnn.train_one(X[:20], y[:20], 2, config, torch.device("cpu"),
                                 seed=20261125, checkpoints=[1, 2], heldout=(X[20:], y[20:]))
                   for _ in range(2)]
        first, second = results
        self.assertEqual(first[1], second[1])
        self.assertEqual(first[2], second[2])
        for name, value in first[0].state_dict().items():
            torch.testing.assert_close(value, second[0].state_dict()[name], rtol=0, atol=0)
            self.assertTrue(torch.isfinite(value).all())
        for row in first[2]:
            self.assertTrue(np.isfinite(row["train_loss"]))
            self.assertTrue(np.isfinite(row["heldout_AP"]))


if __name__ == "__main__":
    unittest.main()
