"""Validation-only evaluation; deterministic thresholds and paired user bootstrap."""

import numpy as np

METRICS = ("precision", "recall", "AP")


def checked(y, scores):
    y, scores = np.asarray(y), np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.shape != y.shape or not len(y):
        raise ValueError("Expected nonempty, aligned vectors")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Labels must be binary and scores finite")
    return y.astype(np.float64), scores


class RankedMetrics:
    """Pre-sort once; ties evaluated together, including in weighted bootstrap AP."""

    def __init__(self, y, scores):
        self.y, self.scores = checked(y, scores)
        self.order = np.argsort(-self.scores, kind="stable")
        self.sorted_y = self.y[self.order]
        self.ends = np.r_[np.flatnonzero(np.diff(self.scores[self.order]) != 0), len(y) - 1]

    def curve(self, weights=None):
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, dtype=float)
        if w.shape != self.y.shape or not np.isfinite(w).all() or (w < 0).any():
            raise ValueError("Invalid sample weights")
        w = w[self.order]
        tp = np.cumsum(w * self.sorted_y)[self.ends]
        count = np.cumsum(w)[self.ends]
        precision = np.divide(tp, count, out=np.zeros_like(tp), where=count > 0)
        recall = tp / tp[-1] if tp[-1] else np.full_like(tp, np.nan)
        thresholds = self.scores[self.order][self.ends]
        return thresholds, precision, recall

    def evaluate(self, threshold, weights=None):
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        w = np.ones(len(self.y)) if weights is None else np.asarray(weights, dtype=float)
        _, precision_curve, recall_curve = self.curve(w)
        predicted = self.scores >= threshold
        tp = float(np.sum(w * self.y * predicted))
        n_predicted, positives = float(np.sum(w * predicted)), float(np.sum(w * self.y))
        ap = float(np.sum(np.diff(np.r_[0, recall_curve]) * precision_curve)) if positives else np.nan
        return np.array([tp / n_predicted if n_predicted else np.nan,
                         tp / positives if positives else np.nan, ap])


def choose_threshold(y, scores, min_recall=0.5):
    if not 0 < min_recall <= 1:
        raise ValueError("Recall constraint must be in (0, 1]")
    ranked = RankedMetrics(y, scores)
    if not ranked.y.sum():
        raise ValueError("Cannot select recall-constrained threshold without positives")
    thresholds, precision, recall = ranked.curve()
    eligible = np.flatnonzero(recall >= min_recall)
    # Exact largest precision, then largest recall, then largest threshold.
    best = max(eligible, key=lambda i: (precision[i], recall[i], thresholds[i]))
    return float(thresholds[best])


def bootstrap_users(y, scores, user_ids, threshold, replicates=2000, seed=20261108, confidence=0.95):
    ranked = RankedMetrics(y, scores)
    ids = np.asarray(user_ids)
    if ids.shape != ranked.y.shape or replicates < 2 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap arguments")
    users, inverse = np.unique(ids, return_inverse=True)
    rng = np.random.default_rng(seed)
    samples = np.empty((replicates, len(METRICS)), dtype=np.float64)
    # Same seed, sorted user universe, and draw count give paired samples across models.
    for b in range(replicates):
        draw = rng.integers(0, len(users), size=len(users))
        multiplicities = np.bincount(draw, minlength=len(users))
        samples[b] = ranked.evaluate(threshold, multiplicities[inverse])
    alpha = (1 - confidence) / 2
    intervals = {}
    for i, name in enumerate(METRICS):
        valid = samples[np.isfinite(samples[:, i]), i]
        intervals[name] = {"lower": float(np.quantile(valid, alpha)) if len(valid) else None,
                           "upper": float(np.quantile(valid, 1 - alpha)) if len(valid) else None,
                           "valid_replicates": len(valid), "undefined_replicates": replicates - len(valid)}
    return samples, intervals, users
