import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.evaluation import RankedMetrics, bootstrap_users, choose_threshold
from common.baseline_run import configuration
from logistic_regression import tune_and_fit


class EvaluationTests(unittest.TestCase):
    def test_metrics_against_sklearn_with_ties_and_weights(self):
        y = np.array([1, 0, 1, 1, 0, 0])
        scores = np.array([.9, .8, .8, .4, .4, .1])
        for w in (np.ones(6), np.array([2, 0, 2, 1, 1, 3])):
            actual = RankedMetrics(y, scores).evaluate(.4, w)
            expected = [precision_score(y, scores >= .4, sample_weight=w),
                        recall_score(y, scores >= .4, sample_weight=w),
                        average_precision_score(y, scores, sample_weight=w)]
            np.testing.assert_allclose(actual, expected)

    def test_threshold_matches_exhaustive_search(self):
        rng = np.random.default_rng(123)
        for _ in range(20):
            y = rng.integers(0, 2, 50)
            scores = rng.integers(0, 8, 50) / 8
            candidates = []
            for t in np.unique(scores):
                p, r, _ = RankedMetrics(y, scores).evaluate(t)
                if r >= .5:
                    candidates.append((p, r, t))
            self.assertEqual(choose_threshold(y, scores), max(candidates)[2])
        self.assertEqual(choose_threshold([1, 1], [.8, .3]), .3)
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [.1, .2])

    def test_user_bootstrap_equals_explicit_cluster_duplication(self):
        y = np.array([1, 0, 1, 0, 1, 0])
        scores = np.array([.9, .4, .8, .3, .7, .2])
        ids = np.array([1, 1, 2, 2, 3, 3])
        samples, ci, users = bootstrap_users(y, scores, ids, .5, 30, 123)
        rng = np.random.default_rng(123)
        for b in range(30):
            selected = rng.integers(0, 3, 3)
            rows = np.concatenate([np.flatnonzero(ids == users[u]) for u in selected])
            np.testing.assert_allclose(samples[b], RankedMetrics(y[rows], scores[rows]).evaluate(.5))
        again, _, _ = bootstrap_users(y, scores, ids, .5, 30, 123)
        np.testing.assert_array_equal(samples, again)
        self.assertEqual(ci['precision']['valid_replicates'], 30)

    def test_undefined_replicates_are_reported_not_silently_zeroed(self):
        samples, ci, _ = bootstrap_users([1, 0], [.9, .1], [1, 2], .5, 100, 1)
        self.assertGreater(ci['precision']['undefined_replicates'], 0)
        self.assertEqual(ci['precision']['lower'], 1)

    def test_cv_scalers_fitted_inside_each_fold(self):
        from sklearn.preprocessing import StandardScaler
        rng = np.random.default_rng(0)
        X = rng.normal(size=(60, 9))
        y = np.tile([0, 1], 30)
        config = configuration()
        config.update(C_grid=[.01, 1.], cv_folds=3)
        calls = []
        original = StandardScaler.fit
        def tracked(self, data, *args, **kwargs):
            calls.append(np.array(data))
            return original(self, data, *args, **kwargs)
        with patch.object(StandardScaler, 'fit', tracked):
            model, _, _, _, fold_id = tune_and_fit({'X_raw':X, 'y':y}, config)
        self.assertEqual([len(x) for x in calls], [40] * 6 + [60])
        for index in range(6):
            np.testing.assert_array_equal(calls[index], X[fold_id != index % 3])
        np.testing.assert_allclose(model[0].mean_, X.mean(axis=0))


if __name__ == '__main__':
    unittest.main()
