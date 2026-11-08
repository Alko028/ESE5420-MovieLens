"""Leakage, alignment, split stability, and hand-calculated feature checks."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from common.dataset import load_development, load_final_test, load_graph
from common.features import build_history, fit_scaler, make_features, transform
from common.preprocessing import prepare
from common.raw_data import load_movielens
from common.splitting import make_splits


def make_fixture(root):
    root.mkdir()
    (root / "u.info").write_text("4 users\n20 items\n80 ratings\n", encoding="ascii")
    rows = [(u, m, (u + m) % 5 + 1, 880000000 + m) for u in range(1, 5) for m in range(1, 21)]
    (root / "u.data").write_text("".join("\t".join(map(str, row)) + "\n" for row in rows), encoding="ascii")
    # Shuffled u.item order tests explicit movie_id mapping, not row position.
    items = []
    for m in range(20, 0, -1):
        items.append("|".join([str(m), "Amélie", "", "", "url"] +
                              [str(int(i == m % 19)) for i in range(19)]))
    (root / "u.item").write_text("\n".join(items) + "\n", encoding="latin-1")
    (root / "u.genre").write_text("".join(f"genre{i}|{i}\n" for i in range(19)), encoding="ascii")


class FeatureTests(unittest.TestCase):
    def test_manual_weighted_features(self):
        pairs = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
        state = build_history(pairs, np.array([5, 1, 4, 2]), 2, 3, neighbor_k=2)
        x = make_features(state, np.array([[1, 3]]))
        # Movie 3 shares a history user with movie 2, but not movie 1.
        np.testing.assert_allclose(x[0], [3, .5, 2, 2, 0, 1, 1, 0, 1])
        self.assertAlmostEqual(state.adjacency[2, 1], 4 / np.sqrt(17))
        np.testing.assert_array_equal(state.adjacency.diagonal(), 0)

    def test_no_neighbor_fallback_and_leakage_rejection(self):
        state = build_history(np.array([[1, 1], [2, 2]]), np.array([5, 2]), 2, 2)
        x = make_features(state, np.array([[1, 2]]))
        np.testing.assert_allclose(x[0], [5, 1, 1, 2, 0, 1, 5, 1, 0])
        with self.assertRaisesRegex(ValueError, "leakage"):
            make_features(state, np.array([[1, 1]]))

    def test_scaler_train_only_and_constant_column(self):
        train = np.array([[1, 5], [3, 5]], dtype=np.float32)
        mean, scale = fit_scaler(train)
        np.testing.assert_allclose(mean, [2, 5])
        np.testing.assert_allclose(scale, [1, 1])
        np.testing.assert_allclose(transform(np.array([[100., 5.]]), mean, scale), [[98, 0]])

    def test_neighbor_tie_order(self):
        state = build_history(np.array([[1, 1], [1, 2], [1, 3]]), np.array([5, 5, 5]), 1, 3, 1)
        self.assertEqual(state.adjacency[2].indices.tolist(), [0])


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw"
        make_fixture(self.raw)
        self.config = BASE / "configs" / "preprocessing.json"

    def tearDown(self):
        self.temp.cleanup()

    def run_pipeline(self, suffix):
        output = self.root / f"outputs_{suffix}"
        manifest = prepare(self.raw, output, self.root / f"splits_{suffix}", self.config)
        return output, manifest

    def test_split_membership_and_fixed_test(self):
        data = load_movielens(self.raw)
        a = make_splits(data.pairs, 123, 42)
        b = make_splits(data.pairs, 123, 43)
        np.testing.assert_array_equal(a["test"], b["test"])
        self.assertFalse(np.array_equal(a["history"], b["history"]))
        np.testing.assert_array_equal(np.sort(np.concatenate(list(a.values()))), np.arange(80))
        self.assertEqual({k: len(v) for k, v in a.items()}, dict(history=48, train=16, validation=8, test=8))
        reverse = make_splits(data.pairs[::-1], 123, 42)
        for name in a:
            self.assertEqual(set(map(tuple, data.pairs[a[name]])), set(map(tuple, data.pairs[::-1][reverse[name]])))
        self.assertEqual(data.movie_genres[0, 1], 1)
        self.assertEqual(data.movie_genres[19, 1], 1)

    def test_target_rating_and_timestamp_changes_cannot_change_inputs(self):
        first, manifest = self.run_pipeline("before")
        data = load_movielens(self.raw)
        config = json.loads(self.config.read_text())
        split = make_splits(data.pairs, config["test_seed"], config["development_seed"])
        raw = np.loadtxt(self.raw / "u.data", dtype=np.int64)
        targets = np.concatenate([split[k] for k in ("train", "validation", "test")])
        raw[targets, 2] = np.where(raw[targets, 2] >= 4, 1, 5)
        raw[:, 3] += 99999
        np.savetxt(self.raw / "u.data", raw, fmt="%d", delimiter="\t")
        second, _ = self.run_pipeline("after")
        for filename in ("train.npz", "validation.npz", "test_inputs.npz", "graph.npz", "scaler.npz"):
            with np.load(first / filename) as left, np.load(second / filename) as right:
                for key in left.files:
                    if key != "y":
                        np.testing.assert_array_equal(left[key], right[key], err_msg=f"Leakage: {filename}/{key}")
        self.assertNotIn("test", manifest["development_positive_counts"])

    def test_model_interfaces_and_test_gate(self):
        output, _ = self.run_pipeline("interfaces")
        real_load = np.load
        def no_test_load(path, *args, **kwargs):
            self.assertNotIn("test", Path(path).name)
            return real_load(path, *args, **kwargs)
        with patch("common.dataset.np.load", side_effect=no_test_load):
            train, val = load_development(output)
            graph = load_graph(output)
        for batch in (train, val):
            self.assertEqual(batch["X"].shape[1], 9)
            self.assertEqual(batch["X"].dtype, np.float32)
            np.testing.assert_array_equal(batch["frequency_score"], batch["X_raw"][:, 4])
            self.assertFalse(graph["history_mask"][batch["user_index"], batch["movie_index"]].any())
            self.assertEqual(len(batch["y"]), len(batch["row_id"]))
        with self.assertRaises(PermissionError):
            load_final_test(output)
        # Explicitly exercise final gate only on synthetic fixture data.
        final = load_final_test(output, protocol_frozen=True)
        self.assertEqual(len(final["y"]), len(final["X"]))
        with np.load(output / "test_inputs.npz") as inputs:
            self.assertNotIn("y", inputs.files)
        with self.assertRaises(FileExistsError):
            prepare(self.raw, output, self.root / "other_split", self.config)

    def test_repeated_run_identical_arrays_and_corruption_detection(self):
        first, _ = self.run_pipeline("a")
        second, _ = self.run_pipeline("b")
        for filename in ("train.npz", "validation.npz", "test_inputs.npz", "graph.npz", "scaler.npz"):
            with np.load(first / filename) as a, np.load(second / filename) as b:
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
        with (first / "train.npz").open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_development(first)

    def test_invalid_rating_rejected(self):
        raw = np.loadtxt(self.raw / "u.data", dtype=np.int64)
        raw[0, 2] = 260  # Must validate before int8 wraparound can hide it.
        np.savetxt(self.raw / "u.data", raw, fmt="%d", delimiter="\t")
        with self.assertRaises(ValueError):
            load_movielens(self.raw)


if __name__ == "__main__":
    unittest.main()
