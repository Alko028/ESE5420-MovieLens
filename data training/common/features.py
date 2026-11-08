"""Nine history-only features and the matching movie graph (no target labels)."""

from dataclasses import dataclass

import numpy as np
from scipy import sparse


FEATURE_NAMES = (
    "user_mean_rating", "user_high_rating_rate", "user_rating_count",
    "movie_mean_rating", "movie_high_rating_rate", "movie_rating_count",
    "neighbor_weighted_mean_rating", "neighbor_weighted_high_rating_rate",
    "neighbor_observed_count",
)


@dataclass
class HistoryState:
    ratings: np.ndarray
    mask: np.ndarray
    user_stats: np.ndarray
    movie_stats: np.ndarray
    adjacency: sparse.csr_matrix  # A[target_movie, source_neighbor] = cosine weight


def build_history(history_pairs, history_ratings, n_users, n_movies, neighbor_k=20):
    if neighbor_k < 1:
        raise ValueError("neighbor_k must be positive")
    if len(np.unique(history_pairs, axis=0)) != len(history_pairs):
        raise ValueError("Duplicate history pairs")
    r = np.zeros((n_users, n_movies), dtype=np.float64)
    u, m = history_pairs[:, 0] - 1, history_pairs[:, 1] - 1
    if (u < 0).any() or (u >= n_users).any() or (m < 0).any() or (m >= n_movies).any():
        raise ValueError("History IDs outside the catalog")
    if not np.isin(history_ratings, np.arange(1, 6)).all():
        raise ValueError("Invalid history rating")
    r[u, m] = history_ratings
    mask = r > 0
    high = r >= 4
    user_count, movie_count = mask.sum(axis=1), mask.sum(axis=0)
    if (user_count == 0).any():
        raise ValueError("Existing-user task requires history for every user")
    # Inactive movie statistics are zeros and their target pairs will be excluded.
    user_stats = np.column_stack((r.sum(axis=1) / user_count,
                                  high.sum(axis=1) / user_count, user_count))
    movie_stats = np.column_stack((r.sum(axis=0) / np.maximum(movie_count, 1),
                                   high.sum(axis=0) / np.maximum(movie_count, 1), movie_count))
    norms = np.linalg.norm(r, axis=0)
    normalized = r / np.where(norms > 0, norms, 1)
    similarity = np.clip(normalized.T @ normalized, 0, 1)
    np.fill_diagonal(similarity, 0)
    target, source, weights = [], [], []
    for movie in range(n_movies):
        candidates = np.flatnonzero(similarity[movie] > 0)
        # Descending similarity, ties resolved by smaller official movie ID.
        order = np.lexsort((candidates, -similarity[movie, candidates]))[:neighbor_k]
        neighbors = candidates[order]
        target.extend([movie] * len(neighbors))
        source.extend(neighbors.tolist())
        weights.extend(similarity[movie, neighbors].tolist())
    adjacency = sparse.csr_matrix((weights, (target, source)), shape=(n_movies, n_movies))
    adjacency.sort_indices()
    return HistoryState(r, mask, user_stats, movie_stats, adjacency)


def make_features(state, pairs):
    """Take IDs only; never accept a target rating or binary target label."""
    pairs = np.asarray(pairs)
    u, m = pairs[:, 0] - 1, pairs[:, 1] - 1
    if (u < 0).any() or (u >= state.ratings.shape[0]).any() or (m < 0).any() or (m >= state.ratings.shape[1]).any():
        raise ValueError("Target IDs outside the catalog")
    if state.mask[u, m].any():
        raise ValueError("A target pair appears in observed history: leakage")
    if (state.movie_stats[m, 2] == 0).any():
        raise ValueError("Filter movies without history before feature construction")
    a = state.adjacency
    mask = state.mask.astype(np.float64)
    denominator = (a @ mask.T).T
    weighted_rating = (a @ state.ratings.T).T
    weighted_high = (a @ (state.ratings >= 4).astype(np.float64).T).T
    connections = a.copy()
    connections.data[:] = 1
    counts = (connections @ mask.T).T
    d = denominator[u, m]
    neighbor_mean = state.user_stats[u, 0].copy()
    neighbor_rate = state.user_stats[u, 1].copy()
    np.divide(weighted_rating[u, m], d, out=neighbor_mean, where=d > 0)
    np.divide(weighted_high[u, m], d, out=neighbor_rate, where=d > 0)
    features = np.column_stack((state.user_stats[u], state.movie_stats[m],
                                neighbor_mean, neighbor_rate, counts[u, m]))
    if features.shape != (len(pairs), 9) or not np.isfinite(features).all():
        raise ValueError("Expected nine finite features")
    return features.astype(np.float32)


def fit_scaler(train_features):
    """Population standard deviation (ddof=0), fit on eligible training inputs only."""
    x = np.asarray(train_features, dtype=np.float64)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Cannot fit scaler on empty/nonfinite training inputs")
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale == 0] = 1
    return mean, scale


def transform(features, mean, scale):
    result = ((features.astype(np.float64) - mean) / scale).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite standardized features")
    return result
