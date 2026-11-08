"""Build reproducible model inputs; no classifier fitting or test evaluation."""

import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import scipy

from .features import FEATURE_NAMES, build_history, fit_scaler, make_features, transform
from .raw_data import load_movielens
from .splitting import make_splits


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_config(path):
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    fixed = {"schema_version": 1, "positive_rating_min": 4,
             "similarity": "cosine_raw_history_zero_filled",
             "neighbor_fallback": "user_history_statistics"}
    for key, value in fixed.items():
        if config.get(key) != value:
            raise ValueError(f"Unsupported proposal setting: {key}")
    for key in ("test_seed", "development_seed", "neighbor_k"):
        if type(config.get(key)) is not int or config[key] < (1 if key == "neighbor_k" else 0):
            raise ValueError(f"Invalid configuration: {key}")
    name = config.get("split_name")
    if not isinstance(name, str) or not name or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in name):
        raise ValueError("split_name must be a safe alphanumeric name")
    return config


def prepare(data_dir, output_dir, split_dir, config_path):
    config = read_config(config_path)
    data_dir, output_dir, split_dir = map(Path, (data_dir, output_dir, split_dir))
    # No silent overwrites: a changed split must have an explicit new artifact path.
    for path in (output_dir, split_dir):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing artifacts: {path}")
    data = load_movielens(data_dir)
    splits = make_splits(data.pairs, config["test_seed"], config["development_seed"])
    h = splits["history"]
    state = build_history(data.pairs[h], data.ratings[h], data.n_users, data.n_movies,
                          config["neighbor_k"])
    eligible, excluded, features = {}, {}, {}
    for name in ("train", "validation", "test"):
        rows = splits[name]
        keep = state.movie_stats[data.pairs[rows, 1] - 1, 2] > 0
        eligible[name], excluded[name] = rows[keep], rows[~keep]
        if not keep.any():
            raise ValueError(f"No eligible {name} samples")
        features[name] = make_features(state, data.pairs[rows[keep]])
    mean, scale = fit_scaler(features["train"])
    output_dir.mkdir(parents=True)
    split_dir.mkdir(parents=True)
    sealed = output_dir / "sealed"
    sealed.mkdir()
    assignments = np.empty(len(data.pairs), dtype="U10")
    for name, rows in splits.items():
        assignments[rows] = name
    # Membership only. No ratings/timestamps or test prevalence in public metadata.
    with (split_dir / "assignments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["row_id", "user_id", "movie_id", "partition", "eligible"])
        for row, (user, movie) in enumerate(data.pairs):
            writer.writerow([row, int(user), int(movie), assignments[row],
                             int(state.movie_stats[movie - 1, 2] > 0)])
    with (split_dir / "per_user_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["user_id", "history", "train", "validation", "test"])
        counts = {name: np.bincount(data.pairs[rows, 0], minlength=data.n_users + 1)
                  for name, rows in splits.items()}
        for user in range(1, data.n_users + 1):
            writer.writerow([user] + [int(counts[name][user]) for name in splits])
    np.savez_compressed(split_dir / "indices.npz", **splits,
                        **{f"{name}_eligible": rows for name, rows in eligible.items()},
                        **{f"{name}_excluded": rows for name, rows in excluded.items()})
    for name, rows in eligible.items():
        pairs = data.pairs[rows]
        values = dict(row_id=rows, user_id=pairs[:, 0], movie_id=pairs[:, 1],
                      user_index=pairs[:, 0] - 1, movie_index=pairs[:, 1] - 1,
                      X_raw=features[name], X=transform(features[name], mean, scale),
                      frequency_score=features[name][:, 4],
                      feature_names=np.asarray(FEATURE_NAMES))
        if name != "test":
            values["y"] = (data.ratings[rows] >= 4).astype(np.int8)
        np.savez_compressed(output_dir / ("test_inputs.npz" if name == "test" else f"{name}.npz"), **values)
    # The only artifact holding test outcomes; development loaders never open it.
    test_rows = eligible["test"]
    np.savez_compressed(sealed / "test_labels.npz", row_id=test_rows,
                        y=(data.ratings[test_rows] >= 4).astype(np.int8))
    np.savez_compressed(output_dir / "scaler.npz", mean=mean, scale=scale,
                        feature_names=np.asarray(FEATURE_NAMES))
    edges = state.adjacency.tocoo()
    np.savez_compressed(output_dir / "graph.npz",
                        edge_index=np.vstack((edges.col, edges.row)).astype(np.int64),
                        edge_weight=edges.data.astype(np.float32),
                        history_rating=state.ratings.astype(np.float32),
                        history_mask=state.mask,
                        movie_genres=data.movie_genres,
                        genre_names=np.asarray(data.genre_names),
                        user_ids=np.arange(1, data.n_users + 1, dtype=np.int32),
                        movie_ids=np.arange(1, data.n_movies + 1, dtype=np.int32))
    manifest = {
        "schema_version": 1,
        "config": config,
        "input_directory": str(data_dir.resolve()),
        "split_directory": str(split_dir.resolve()),
        "source_sha256": {name: sha256(data_dir / name)
                          for name in ("u.data", "u.item", "u.genre", "u.info")},
        "code_sha256": {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "dataset": {"ratings": len(data.pairs), "users": data.n_users, "movies": data.n_movies,
                    "genre_fields": len(data.genre_names)},
        "partition_counts": {name: len(rows) for name, rows in splits.items()},
        "rounding_rule": "floor(N/10) test, floor(N/10) validation, floor(N/5) train; remainder history, per user",
        "eligible_counts": {name: len(rows) for name, rows in eligible.items()},
        "excluded_counts": {name: len(rows) for name, rows in excluded.items()},
        "development_positive_counts": {name: int((data.ratings[eligible[name]] >= 4).sum())
                                        for name in ("train", "validation")},
        "feature_names": list(FEATURE_NAMES),
        "scaler_fit_partition": "eligible training targets only; ddof=0; zero std -> 1",
        "history_movies": int((state.movie_stats[:, 2] > 0).sum()),
        "graph": {"edges": int(state.adjacency.nnz), "directed": True,
                  "edge_index_convention": "[source_neighbor, target_movie], zero-based",
                  "self_loops": False, "edge_weight": "positive raw cosine similarity, not row-normalized",
                  "tie_break": "smaller movie_id first", "genre_used_in_graph": False},
        "no_neighbor_counts": {name: int((features[name][:, 8] == 0).sum()) for name in features},
        "test_policy": "labels separated, not encrypted; development loaders do not expose test inputs or labels",
    }
    manifest["artifact_sha256"] = {str(p.relative_to(output_dir)).replace("\\", "/"): sha256(p)
                                   for p in sorted(output_dir.rglob("*.npz"))}
    manifest["split_sha256"] = {p.name: sha256(p) for p in sorted(split_dir.iterdir())}
    # Completion marker written last; loaders refuse an incomplete run.
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
