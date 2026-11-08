"""One random-within-user development split; independently seeded fixed test."""

import numpy as np


def make_splits(pairs, test_seed=20261106, development_seed=42):
    buckets = {name: [] for name in ("history", "train", "validation", "test")}
    for user_id in np.unique(pairs[:, 0]):
        rows = np.flatnonzero(pairs[:, 0] == user_id)
        # Canonical movie order means membership is independent of input row order.
        rows = rows[np.argsort(pairs[rows, 1], kind="stable")]
        n = len(rows)
        n_test, n_val, n_train = n // 10, n // 10, n // 5
        if min(n_test, n_val, n_train, n - n_test - n_val - n_train) < 1:
            raise ValueError("Every user needs enough ratings for all four partitions")
        test_rng = np.random.default_rng(np.random.SeedSequence([test_seed, int(user_id)]))
        order = test_rng.permutation(rows)
        buckets["test"].extend(order[:n_test])
        remaining = np.sort(order[n_test:])
        remaining = remaining[np.argsort(pairs[remaining, 1], kind="stable")]
        dev_rng = np.random.default_rng(np.random.SeedSequence([development_seed, int(user_id), 1]))
        dev = dev_rng.permutation(remaining)
        buckets["validation"].extend(dev[:n_val])
        buckets["train"].extend(dev[n_val:n_val + n_train])
        buckets["history"].extend(dev[n_val + n_train:])
    result = {name: np.sort(np.asarray(rows, dtype=np.int64)) for name, rows in buckets.items()}
    combined = np.concatenate(list(result.values()))
    if not np.array_equal(np.sort(combined), np.arange(len(pairs))):
        raise ValueError("Partitions must be disjoint and exhaustive")
    return result
