import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from knn import knn_configuration, model_for, tune_and_fit


class KnnTests(unittest.TestCase):
    def test_uniform_and_distance_vote_including_zero_distance(self):
        config = knn_configuration()
        X, y = np.array([[0.], [2.], [4.]]), np.array([0, 1, 1])
        uniform = model_for(3, 'uniform', config).fit(X, y)
        weighted = model_for(3, 'distance', config).fit(X, y)
        np.testing.assert_allclose(uniform.predict_proba([[0.]])[0, 1], 2/3)
        np.testing.assert_allclose(weighted.predict_proba([[0.]])[0, 1], 0)
        # At x=1 the distances are 1,1,3; standardized distance scales cancel.
        np.testing.assert_allclose(weighted.predict_proba([[1.]])[0, 1], 4/7)

    def test_cv_fold_local_scaling_and_same_assignments_as_lr(self):
        rng = np.random.default_rng(21)
        train = {'X_raw': rng.normal(size=(100, 9)), 'y': np.tile([0, 1], 50)}
        config = knn_configuration()
        config.update(n_neighbors_grid=[3, 5], weights_grid=['uniform', 'distance'])
        calls = []
        original = StandardScaler.fit
        def track(self, X, *args, **kwargs):
            calls.append(np.array(X))
            return original(self, X, *args, **kwargs)
        with patch.object(StandardScaler, 'fit', track):
            model, selected, records, summary, ids = tune_and_fit(train, config)
        self.assertEqual([len(x) for x in calls], [80]*20 + [100])
        for i in range(20):
            np.testing.assert_array_equal(calls[i], train['X_raw'][ids != i % 5])
        for i, (_, heldout) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(train['X_raw'], train['y'])):
            self.assertTrue((ids[heldout] == i).all())
        best = max(summary, key=lambda r: (r[2], -r[0], r[1] == 'uniform'))
        self.assertEqual(selected, {'n_neighbors': best[0], 'weights': best[1]})
        np.testing.assert_allclose(model[0].mean_, train['X_raw'].mean(axis=0))

    def test_impossible_k_rejected(self):
        config = knn_configuration()
        with self.assertRaises(ValueError):
            tune_and_fit({'X_raw': np.zeros((20,9)), 'y': np.tile([0,1],10)}, config)


if __name__ == '__main__':
    unittest.main()
