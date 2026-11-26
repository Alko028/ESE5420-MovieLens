"""Synthetic second-graph-layer checks; no MovieLens outcome files are read."""

import io
import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as exc:
    raise unittest.SkipTest("Optional deep-GNN tests require PyTorch") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.deep_movie_gnn import TwoLayerMovieGNN, build_full_history_taps
from gnn import normalized_adjacency, seed_training
from test_gnn import directed_fixture


class DeepMovieGNNTests(unittest.TestCase):
    def setUp(self):
        seed_training(72, 2)
        self.graph, self.batch = directed_fixture()
        self.adjacency = normalized_adjacency(self.graph)
        self.features = torch.from_numpy(build_full_history_taps(self.graph, 3))
        self.users = torch.from_numpy(self.batch["user_index"])
        self.movies = torch.from_numpy(self.batch["movie_index"])

    def test_hand_calculated_directed_taps_and_label_independence(self):
        features = build_full_history_taps(self.graph, 3)
        np.testing.assert_allclose(features[0, 1], [0, 0, 11 / 15, 1, 0, 0], atol=1e-7)
        np.testing.assert_allclose(features[1, 2], [0, 0, .8, 1, 0, 0], atol=1e-7)
        np.testing.assert_array_equal(features, build_full_history_taps({**self.graph, "y": np.ones(99)}, 3))

    def test_full_dense_reference_and_duplicate_users(self):
        model = TwoLayerMovieGNN(self.adjacency, 3, 5)
        dense = torch.from_numpy(self.adjacency.toarray())
        first = torch.relu(model.first(self.features))
        taps = [first]
        for _ in range(2):
            taps.append(torch.einsum("mn,unh->umh", dense, taps[-1]))
        all_logits = model.readout(torch.relu(model.second(torch.cat(taps, dim=-1)))).squeeze(-1)
        actual = model(self.features, self.users, self.movies)
        torch.testing.assert_close(actual, all_logits[self.users, self.movies])
        repeated = torch.tensor([2, 0, 2, 1, 3, 0], dtype=torch.long)
        torch.testing.assert_close(model(self.features, self.users[repeated], self.movies[repeated]), actual[repeated])

    def test_second_filter_really_propagates_hidden_neighbors(self):
        model = TwoLayerMovieGNN(self.adjacency, 3, 1)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            model.first.weight[0, 0] = 1  # only the original observed rating
            model.second.weight[0, 1] = 1  # only one-hop hidden propagation
            model.readout.weight.fill_(1)
        actual = model(self.features, torch.tensor([0]), torch.tensor([1]))
        torch.testing.assert_close(actual, torch.tensor([11 / 15]))

    def test_finite_gradients_and_all_second_taps_participate(self):
        model = TwoLayerMovieGNN(self.adjacency, 3, 4)
        # Positive biases avoid accidental zero-gradient ReLU fixtures.
        with torch.no_grad():
            model.first.bias.fill_(1.)
            model.second.bias.fill_(1.)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(self.features, self.users, self.movies), torch.tensor([1., 0., 0., 1.]))
        loss.backward()
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        for order in range(3):
            self.assertGreater(model.second.weight.grad[:, order * 4:(order + 1) * 4].abs().sum(), 0)
        self.assertGreater(model.first.weight.grad.abs().sum(), 0)

    def test_parameter_count_and_state_buffer_restore(self):
        model = TwoLayerMovieGNN(self.adjacency)
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 3361)
        expected = model(self.features, self.users, self.movies)
        stored = io.BytesIO()
        torch.save(model.state_dict(), stored)
        stored.seek(0)
        restored = TwoLayerMovieGNN(self.adjacency)
        restored.load_state_dict(torch.load(stored, weights_only=True))
        torch.testing.assert_close(restored(self.features, self.users, self.movies), expected, rtol=0, atol=0)
        torch.testing.assert_close(restored.adjacency.to_dense(), model.adjacency.to_dense())
        self.assertNotIn("adjacency", dict(restored.named_parameters()))
        self.assertIn("adjacency", dict(restored.named_buffers()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_cuda_deterministic_forward_backward_and_buffer_to(self):
        outputs, gradients = [], []
        for _ in range(2):
            seed_training(72, 2)
            model = TwoLayerMovieGNN(self.adjacency, 3, 4).to("cuda")
            self.assertEqual(model.adjacency.device.type, "cuda")
            output = model(self.features.cuda(), self.users.cuda(), self.movies.cuda())
            output.sum().backward()
            outputs.append(output.detach().cpu())
            gradients.append(torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters()]))
        torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
        torch.testing.assert_close(gradients[0], gradients[1], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
