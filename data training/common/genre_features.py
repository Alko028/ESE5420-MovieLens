"""Append fixed movie genres and history-only preferences to the nine inputs."""

import hashlib
import json

import numpy as np
from threadpoolctl import threadpool_limits

from .dataset import DEFAULT_OUTPUT, load_development, load_graph
from .features import FEATURE_NAMES
from .preprocessing import sha256


def genre_configuration():
    path = DEFAULT_OUTPUT.parents[1] / "configs/genre.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1, "genre_fields": 19, "base_features": 9,
        "feature_order": ["base_history", "movie_genre", "user_genre_high_rating_rate", "target_genre_x_user_preference"],
        "user_preference": "history_high_count_per_genre/history_observed_count_per_genre",
        "empty_genre_fallback": "user_overall_history_high_rating_rate",
        "multi_genre_rule": "one history rating counts once in each indicated genre",
        "positive_rating_min": 4,
        "standardization": "fit independently inside each training fold; refit on all training targets after CV",
        "hyperparameter_grid": "unchanged from the corresponding ratings-only baseline",
    }
    if config != required:
        raise ValueError("Unsupported/unrecorded genre feature definition")
    return config


def build_genre_features(graph, batch):
    """Return raw 9+3G inputs and names; target labels are never read here.

    Multi-genre history movies contribute to each indicated genre. A user's
    unseen genre uses that user's overall history high-rating fraction.
    """
    rating = np.asarray(graph["history_rating"], dtype=np.float64)
    mask = np.asarray(graph["history_mask"])
    genres = np.asarray(graph["movie_genres"])
    names = np.asarray(graph["genre_names"])
    if rating.ndim != 2 or mask.shape != rating.shape or not np.array_equal(mask, rating > 0):
        raise ValueError("History rating/mask mismatch")
    if not np.isfinite(rating).all() or not np.isin(rating, [0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Invalid history ratings")
    if genres.ndim != 2 or genres.shape[0] != rating.shape[1] or not genres.shape[1]:
        raise ValueError("Genre metadata/catalog shape mismatch")
    if not np.isin(genres, [0, 1]).all() or names.shape != (genres.shape[1],):
        raise ValueError("Genres must be binary with aligned genre names")
    if len(set(names.tolist())) != len(names) or any(not isinstance(name, str) or not name for name in names.tolist()):
        raise ValueError("Genre names must be unique nonempty strings")
    users, movies = np.asarray(batch["user_index"]), np.asarray(batch["movie_index"])
    raw = np.asarray(batch["X_raw"])
    if users.ndim != 1 or movies.shape != users.shape or not len(users) or raw.shape != (len(users), 9):
        raise ValueError("Invalid aligned base inputs")
    if tuple(batch["feature_names"]) != FEATURE_NAMES or not np.isfinite(raw).all():
        raise ValueError("Expected original nine finite history features in their original order")
    if not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(movies.dtype, np.integer):
        raise ValueError("Target indices must be integers")
    if (users < 0).any() or (users >= rating.shape[0]).any() or (movies < 0).any() or (movies >= rating.shape[1]).any():
        raise ValueError("Target index outside catalog")
    for key, indices, catalog in (("user_id", users, "user_ids"), ("movie_id", movies, "movie_ids")):
        if key in batch and not np.array_equal(np.asarray(graph[catalog])[indices], batch[key]):
            raise ValueError("Official catalog IDs do not match target indices")
    if mask[users, movies].any():
        raise ValueError("A target pair appears in observed history: leakage")
    if (mask.sum(axis=0)[movies] == 0).any():
        raise ValueError("Target movie is outside the common history eligibility rule")
    user_count = mask.sum(axis=1)
    if (user_count == 0).any():
        raise ValueError("Existing-user task requires observed history")
    high = rating >= 4
    overall = high.sum(axis=1) / user_count
    with threadpool_limits(limits=4):
        genre_count = mask.astype(np.float64) @ genres.astype(np.float64)
        genre_high = high.astype(np.float64) @ genres.astype(np.float64)
    preference = np.broadcast_to(overall[:, None], genre_count.shape).copy()
    np.divide(genre_high, genre_count, out=preference, where=genre_count > 0)
    movie_genre = genres[movies].astype(np.float32)
    user_genre = preference[users].astype(np.float32)
    interaction = movie_genre * user_genre
    result = np.column_stack((raw, movie_genre, user_genre, interaction)).astype(np.float32)
    feature_names = np.asarray([*FEATURE_NAMES,
        *(f"movie_genre:{name}" for name in names),
        *(f"user_genre_high_rating_rate:{name}" for name in names),
        *(f"target_genre_x_user_preference:{name}" for name in names)])
    if not np.isfinite(result).all() or not np.array_equal(result[:, :9], raw):
        raise ValueError("Genre extension altered base features or produced nonfinite values")
    return result, feature_names


def load_genre_development():
    """Reuse fixed development rows; never load test inputs/labels or fit a scaler."""
    config = genre_configuration()
    train, validation = load_development()
    graph = load_graph()
    if np.asarray(graph["movie_genres"]).shape[1] != config["genre_fields"]:
        raise ValueError("Official MovieLens genre metadata must contain 19 fields")
    if np.intersect1d(train["row_id"], validation["row_id"]).size:
        raise ValueError("Training and validation row IDs overlap")
    batches, fingerprints = [], {}
    feature_names = None
    for partition, batch in (("train", train), ("validation", validation)):
        raw, names = build_genre_features(graph, batch)
        if len(np.unique(batch["row_id"])) != len(batch["row_id"]):
            raise ValueError("Duplicate target row IDs")
        if feature_names is not None and not np.array_equal(feature_names, names):
            raise ValueError("Training/validation feature order differs")
        feature_names = names
        # Drop baseline's pre-scaled X: every CV fold must fit a fresh scaler.
        extended = {key: value for key, value in batch.items() if key not in ("X", "X_raw", "feature_names")}
        extended.update(X_raw=raw, feature_names=names)
        for key in ("row_id", "user_id", "movie_id", "user_index", "movie_index", "y"):
            np.testing.assert_array_equal(extended[key], batch[key])
        batches.append(extended)
        fingerprints[partition] = hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest()
    metadata = {
        "graph_sha256": sha256(DEFAULT_OUTPUT / "graph.npz"),
        "genre_config_sha256": sha256(DEFAULT_OUTPUT.parents[1] / "configs/genre.json"),
        "genre_feature_definition": config,
        "n_features": len(feature_names), "genre_feature_names": feature_names.tolist(),
        "base_features_unchanged": True, "development_alignment_verified": True,
        "feature_matrix_sha256": fingerprints,
        "genre_information_source": "public movie genre metadata and observed-history ratings only",
    }
    return *batches, metadata
