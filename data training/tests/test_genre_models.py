"""The genre wrappers retain original tuning and CV with all 66 raw features."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import knn
import knn_genre
import logistic_regression
import logistic_regression_genre
from common.baseline_run import configuration


class GenreModelTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(512)
        X = rng.normal(size=(100, 66))
        # Include zero and constant columns, as movie genre metadata can have them.
        X[:, 9] = 0
        X[:, 10] = 1
        X[-1, -1] = 1000  # Makes leaking full-data scaling easily detectable.
        self.train = {"X_raw": X, "y": np.tile([0, 1], 50)}

    def test_wrappers_reuse_original_tuners_and_configurations(self):
        self.assertIs(logistic_regression_genre.tune_and_fit, logistic_regression.tune_and_fit)
        self.assertIs(knn_genre.tune_and_fit, knn.tune_and_fit)
        self.assertEqual(logistic_regression_genre.configuration(), configuration())
        self.assertEqual(knn_genre.knn_configuration(), knn.knn_configuration())

    def check_scaling(self, tuner, config, n_cv_fits):
        calls = []
        original = StandardScaler.fit

        def tracked(scaler, X, *args, **kwargs):
            calls.append(np.array(X))
            return original(scaler, X, *args, **kwargs)

        with patch.object(StandardScaler, "fit", tracked), threadpool_limits(limits=2):
            fitted, selected, records, means, ids = tuner(self.train, config)
        self.assertEqual(len(calls), n_cv_fits + 1)
        self.assertTrue(all(X.shape == (80, 66) for X in calls[:-1]))
        for i, fit_data in enumerate(calls[:-1]):
            np.testing.assert_array_equal(fit_data, self.train["X_raw"][ids != i % 5])
        np.testing.assert_array_equal(calls[-1], self.train["X_raw"])
        np.testing.assert_allclose(fitted[0].mean_, self.train["X_raw"].mean(axis=0))
        self.assertEqual(fitted[0].n_features_in_, 66)
        expected = np.empty(len(ids), dtype=np.int8)
        # The original nine-column representation gives identical splits.
        for fold, (_, heldout) in enumerate(StratifiedKFold(
                n_splits=5, shuffle=True, random_state=42).split(self.train["X_raw"][:, :9], self.train["y"])):
            expected[heldout] = fold
        np.testing.assert_array_equal(ids, expected)

    def test_lr_66_column_scaling_stays_inside_original_folds(self):
        config = configuration()
        config["C_grid"] = [0.001, 0.01]
        self.check_scaling(logistic_regression_genre.tune_and_fit, config, 10)

    def test_knn_66_column_scaling_stays_inside_original_folds(self):
        config = knn.knn_configuration()
        config.update(n_neighbors_grid=[3, 5], weights_grid=["uniform", "distance"])
        self.check_scaling(knn_genre.tune_and_fit, config, 20)


if __name__ == "__main__":
    unittest.main()
