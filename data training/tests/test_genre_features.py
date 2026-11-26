"""Hand-calculated history-only genre features; no sealed test data is read."""

import copy
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))

from common.features import FEATURE_NAMES
from common.genre_features import build_genre_features, load_genre_development


def genre_fixture():
    """Three genres, including a multi-genre movie and an unseen user genre."""
    ratings = np.array([
        [5., 2., 0., 4., 0., 0.],
        [1., 0., 4., 0., 5., 0.],
        [0., 4., 0., 0., 0., 3.],
    ], dtype=np.float32)
    graph = {
        "history_rating": ratings,
        "history_mask": ratings > 0,
        "movie_genres": np.array([
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 0, 0],
        ], dtype=np.int8),
        "genre_names": np.array(["action", "comedy", "drama"]),
        "user_ids": np.arange(1, 4),
        "movie_ids": np.arange(1, 7),
    }
    user_index = np.array([0, 0, 1, 2, 2], dtype=np.int64)
    movie_index = np.array([2, 4, 3, 0, 4], dtype=np.int64)
    n = len(user_index)
    batch = {
        "X_raw": np.arange(n * 9, dtype=np.float32).reshape(n, 9) / 10,
        "feature_names": np.array(FEATURE_NAMES),
        "row_id": np.arange(50, 50 + n),
        "user_index": user_index,
        "movie_index": movie_index,
        "user_id": user_index + 1,
        "movie_id": movie_index + 1,
        "y": np.array([0, 1, 1, 0, 1], dtype=np.int8),
    }
    return graph, batch


class GenreFeatureTests(unittest.TestCase):
    def test_manual_multigenre_rates_and_interactions(self):
        graph, batch = genre_fixture()
        actual, names = build_genre_features(graph, batch)
        # User 0 has action ratings [5, 2], comedy [2], drama [4].
        # The multi-genre movie contributes independently to action and comedy.
        # User 2 has action/comedy [4] and no drama history, so drama falls
        # back to that user's overall high-rating rate: 1 / 2.
        expected_user_rates = np.array([
            [.5, 0, 1],
            [.5, 0, 1],
            [.5, 1, 1],
            [1, 1, .5],
            [1, 1, .5],
        ])
        expected_movie_bits = np.array([
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 1],
            [1, 0, 0],
            [1, 0, 1],
        ])
        self.assertEqual(actual.shape, (5, 18))
        self.assertEqual(len(names), 18)
        np.testing.assert_array_equal(actual[:, :9], batch["X_raw"])
        np.testing.assert_array_equal(actual[:, 9:12], expected_movie_bits)
        np.testing.assert_allclose(actual[:, 12:15], expected_user_rates)
        np.testing.assert_allclose(
            actual[:, 15:18], expected_movie_bits * expected_user_rates)
        self.assertTrue(np.isfinite(actual).all())

    def test_existing_features_and_genre_name_order_are_preserved(self):
        graph, batch = genre_fixture()
        original = copy.deepcopy(batch)
        actual, names = build_genre_features(graph, batch)
        self.assertEqual(tuple(names[:9]), FEATURE_NAMES)
        self.assertEqual(len(set(names)), len(names))
        expected_names = FEATURE_NAMES + tuple(
            f"{prefix}:{genre}"
            for prefix in ("movie_genre", "user_genre_high_rating_rate",
                           "target_genre_x_user_preference")
            for genre in graph["genre_names"])
        self.assertEqual(tuple(names), expected_names)
        np.testing.assert_array_equal(actual[:, :9], original["X_raw"])
        for key in original:
            np.testing.assert_array_equal(batch[key], original[key])

    def test_target_labels_never_change_model_inputs(self):
        graph, batch = genre_fixture()
        original, names = build_genre_features(graph, batch)
        altered = {**batch, "y": 1 - batch["y"]}
        changed, changed_names = build_genre_features(graph, altered)
        np.testing.assert_array_equal(original, changed)
        self.assertEqual(tuple(names), tuple(changed_names))
        without_labels = {key: value for key, value in batch.items() if key != "y"}
        unlabeled, _ = build_genre_features(graph, without_labels)
        np.testing.assert_array_equal(original, unlabeled)

    def test_target_pair_in_observed_history_is_rejected(self):
        graph, batch = genre_fixture()
        batch["movie_index"][0] = 0
        batch["movie_id"][0] = 1
        with self.assertRaises(ValueError):
            build_genre_features(graph, batch)

    def test_movie_with_no_observed_history_is_rejected(self):
        graph, batch = genre_fixture()
        graph["history_rating"][:, 2] = 0
        graph["history_mask"][:, 2] = False
        with self.assertRaises(ValueError):
            build_genre_features(graph, batch)

    def test_nonbinary_or_nonfinite_genres_are_rejected(self):
        for value in (2, -1, .5, np.nan, np.inf):
            with self.subTest(value=value):
                graph, batch = genre_fixture()
                graph["movie_genres"] = graph["movie_genres"].astype(float)
                graph["movie_genres"][0, 0] = value
                with self.assertRaises(ValueError):
                    build_genre_features(graph, batch)

    def test_genre_catalog_alignment_is_checked(self):
        for mutation in ("movie_rows", "genre_columns", "genre_names"):
            with self.subTest(mutation=mutation):
                graph, batch = genre_fixture()
                if mutation == "movie_rows":
                    graph["movie_genres"] = graph["movie_genres"][:-1]
                elif mutation == "genre_columns":
                    graph["movie_genres"] = graph["movie_genres"][:, :-1]
                else:
                    graph["genre_names"] = graph["genre_names"][:-1]
                with self.assertRaises(ValueError):
                    build_genre_features(graph, batch)

    def test_invalid_observed_ratings_are_rejected(self):
        for value in (0, 1.5, 6, np.nan):
            with self.subTest(value=value):
                graph, batch = genre_fixture()
                graph["history_rating"][0, 0] = value
                with self.assertRaises(ValueError):
                    build_genre_features(graph, batch)

    def test_official_loader_rejects_non_movielens_genre_width(self):
        graph, batch = genre_fixture()
        # The generic builder supports the 3-genre hand-calculation fixture;
        # the production MovieLens loader must enforce the official 19 fields.
        with patch("common.genre_features.genre_configuration", return_value={"genre_fields": 19}), \
                patch("common.genre_features.load_development", return_value=(batch, batch)), \
                patch("common.genre_features.load_graph", return_value=graph):
            with self.assertRaises(ValueError):
                load_genre_development()


if __name__ == "__main__":
    unittest.main()
