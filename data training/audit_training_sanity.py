"""Audit all fixed development models; optionally run the two genre models once.

Real test files are blocked. Original baseline/GNN results are checked without
refitting them; new genre fits can be monitored at every actual scaler.fit call.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, configuration
from common.dataset import DEFAULT_OUTPUT, load_development, load_graph
from common.features import build_history, make_features
from common.genre_features import load_genre_development
from common.preprocessing import sha256
from compare_gnn import load_run
import logistic_regression_genre
import knn_genre


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(output_dir, run_genre=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    root = DEFAULT_OUTPUT.resolve()
    opened = set()

    def guard(event, values):
        if event == "open" and isinstance(values[0], (str, bytes)):
            path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
            if root in path.parents:
                relative = path.relative_to(root).as_posix()
                if relative == "test_inputs.npz" or relative.startswith("sealed/"):
                    raise PermissionError("Training sanity audit blocked real test access")
                opened.add(relative)

    sys.addaudithook(guard)
    original_train, validation = load_development()
    train, genre_validation, metadata = load_genre_development()
    graph = load_graph()
    manifest = json.loads((root / "manifest.json").read_text())
    inputs_before = {name: sha256(root / name) for name in ("train.npz", "validation.npz", "graph.npz", "manifest.json")}
    for batch, original in ((train, original_train), (genre_validation, validation)):
        for key in ("row_id", "user_id", "movie_id", "user_index", "movie_index", "y"):
            np.testing.assert_array_equal(batch[key], original[key])
        np.testing.assert_array_equal(batch["X_raw"][:, :9], original["X_raw"])
        assert batch["X_raw"].shape == (len(original["y"]), 66)
        assert not graph["history_mask"][batch["user_index"], batch["movie_index"]].any()
    # Rebuild history statistics, validate frozen graph weights/top-k choices.
    # BLAS kernels can order near-exact cosine ties differently. Do not rebuild
    # the authoritative fixed graph, or silently tolerate changed neighbor counts.
    u, m = np.nonzero(graph["history_mask"])
    with threadpool_limits(limits=4):
        state = build_history(np.column_stack((u + 1, m + 1)), graph["history_rating"][u, m],
                              len(graph["user_ids"]), len(graph["movie_ids"]), manifest["config"]["neighbor_k"])
        edges = state.adjacency.tocoo()
        rebuilt_edges = set(zip(edges.col.tolist(), edges.row.tolist()))
        frozen_edges = set(zip(*graph["edge_index"].tolist()))
        norms = np.linalg.norm(state.ratings, axis=0)
        normalized = state.ratings / np.where(norms > 0, norms, 1)
        similarity = np.clip(normalized.T @ normalized, 0, 1)
        np.fill_diagonal(similarity, 0)
    source, target = graph["edge_index"]
    assert (source != target).all() and (graph["edge_weight"] > 0).all()
    np.testing.assert_allclose(similarity[target, source], graph["edge_weight"], rtol=1e-7, atol=1e-8)
    for movie in range(similarity.shape[0]):
        selected = source[target == movie]
        assert len(selected) == min(manifest["config"]["neighbor_k"], np.count_nonzero(similarity[movie] > 0))
        if len(selected):
            unselected = np.ones(similarity.shape[1], dtype=bool)
            unselected[selected] = False
            assert similarity[movie, unselected].max() <= similarity[movie, selected].min() + 1e-12
    topology_difference = {"new_environment_only": sorted(rebuilt_edges - frozen_edges),
                           "frozen_graph_only": sorted(frozen_edges - rebuilt_edges),
                           "top_k_tie_tolerance": 1e-12}
    # Frozen edge weights were stored as float32; continuous-feature rebuilding
    # can differ by <1e-6. All count features must still match exactly.
    state.adjacency = sparse.csr_matrix((graph["edge_weight"].astype(np.float64), (target, source)), shape=similarity.shape)
    feature_max_error = {}
    with threadpool_limits(limits=4):
        for partition, batch in (("train", original_train), ("validation", validation)):
            rebuilt = make_features(state, np.column_stack((batch["user_id"], batch["movie_id"])))
            np.testing.assert_allclose(rebuilt, batch["X_raw"], rtol=0, atol=1e-6)
            np.testing.assert_array_equal(rebuilt[:, [2, 5, 8]], batch["X_raw"][:, [2, 5, 8]])
            feature_max_error[partition] = float(np.max(np.abs(rebuilt - batch["X_raw"])))
    # Membership files contain no ratings. Verify all methods retain one test split.
    split_root = BASE / "splits/baseline_v1"
    for filename, digest in manifest["split_sha256"].items():
        assert sha256(split_root / filename) == digest
    assignments = csv_rows(split_root / "assignments.csv")
    history_pairs = {(int(row["user_id"]), int(row["movie_id"])) for row in assignments if row["partition"] == "history"}
    assert history_pairs == set(zip((u + 1).tolist(), (m + 1).tolist()))
    for partition, batch in (("train", original_train), ("validation", validation)):
        eligible_rows = [int(row["row_id"]) for row in assignments if row["partition"] == partition and int(row["eligible"])]
        np.testing.assert_array_equal(eligible_rows, batch["row_id"])
    cfg = configuration()
    folds = list(StratifiedKFold(cfg["cv_folds"], shuffle=True, random_state=cfg["cv_seed"]).split(original_train["X_raw"], original_train["y"]))
    expected_ids = np.empty(len(train["y"]), dtype=int)
    for fold, (_, heldout) in enumerate(folds):
        expected_ids[heldout] = fold
    actual_fit_checks = {}
    if run_genre:
        X, y = train["X_raw"].astype(np.float64), train["y"]
        for name, module, candidates in (
            ("logistic_regression_genre", logistic_regression_genre, len(cfg["C_grid"])),
            ("knn_genre", knn_genre, len(knn_genre.knn_configuration()["n_neighbors_grid"]) * len(knn_genre.knn_configuration()["weights_grid"]))):
            expected = [fit for _ in range(candidates) for fit, _ in folds] + [np.arange(len(y))]
            checked = []
            original_fit = StandardScaler.fit

            def checked_fit(scaler, values, labels=None, **kwargs):
                indices = expected[len(checked)]
                np.testing.assert_array_equal(values, X[indices])
                np.testing.assert_array_equal(labels, y[indices])
                fitted = original_fit(scaler, values, labels, **kwargs)
                np.testing.assert_allclose(scaler.mean_, X[indices].mean(axis=0), atol=1e-12)
                checked.append({"n_fit": len(indices), "n_features": values.shape[1]})
                return fitted

            with patch.object(StandardScaler, "fit", checked_fit):
                module.run()
            assert len(checked) == len(expected)
            actual_fit_checks[name] = checked
            print(f"PASS {name}: checked {len(checked)} actual fold/final scaler fits", flush=True)
    names = ("movie_high_rating_rate", "logistic_regression", "knn", "gnn", "logistic_regression_genre", "knn_genre")
    runs = {name: load_run(name, BASE / "outputs" / name) for name in names}
    reference = runs["logistic_regression"]
    metric_checks = {}
    for name, run in runs.items():
        result = run["result"]
        assert result["input_sha256"] == reference["result"]["input_sha256"]
        assert run["keys"] == reference["keys"]
        for key in ("bootstrap_seed", "bootstrap_replicates", "confidence_level", "minimum_validation_recall"):
            assert result["config"][key] == cfg[key]
        np.testing.assert_array_equal(run["bootstrap"]["user_ids"], reference["bootstrap"]["user_ids"])
        # Code files recorded by older runs are checked; newly added modules need
        # not retroactively appear in their source manifests.
        for filename, digest in result["code_sha256"].items():
            assert sha256(BASE / filename) == digest, (name, filename)
        for filename, digest in result.get("shared_training_code_sha256", {}).items():
            assert sha256(BASE / filename) == digest, (name, filename)
        rows = csv_rows(run["root"] / "validation_predictions.csv")
        score = np.array([float(row["score"]) for row in rows])
        predicted = score >= result["threshold"]
        values = [precision_score(validation["y"], predicted), recall_score(validation["y"], predicted), average_precision_score(validation["y"], score)]
        np.testing.assert_allclose(values, [result["metrics"][key] for key in ("precision", "recall", "AP")], rtol=0, atol=1e-14)
        p, r, t = precision_recall_curve(validation["y"], score)
        feasible = np.flatnonzero(r[:-1] >= cfg["minimum_validation_recall"])
        best = max(feasible, key=lambda i: (p[i], r[i], t[i]))
        assert t[best] == result["threshold"]
        users, inverse = np.unique(validation["user_id"], return_inverse=True)
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        for replicate in range(20):
            weight = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))[inverse]
            point = [precision_score(validation["y"], predicted, sample_weight=weight),
                     recall_score(validation["y"], predicted, sample_weight=weight),
                     average_precision_score(validation["y"], score, sample_weight=weight)]
            np.testing.assert_allclose(point, run["bootstrap"]["samples"][replicate], atol=1e-14, rtol=0)
        if name != "movie_high_rating_rate":
            saved = csv_rows(run["root"] / "cv_assignments.csv")
            np.testing.assert_array_equal([int(row["row_id"]) for row in saved], train["row_id"])
            np.testing.assert_array_equal([int(row["fold"]) for row in saved], expected_ids)
            assert result["config"]["cv_folds"] == 5 and result["config"]["cv_seed"] == 42
        metric_checks[name] = {"rows_and_labels_match": True, "input_hashes_match": True,
                               "independent_metrics_and_threshold_match": True,
                               "independent_bootstrap_replicates": 20, "saved_source_hashes_match": True}
    for base, extra in (("logistic_regression", "logistic_regression_genre"), ("knn", "knn_genre")):
        assert runs[base]["result"]["config"] == runs[extra]["result"]["config"]
    assert inputs_before == {filename: sha256(root / filename) for filename in inputs_before}
    report = {"status": "PASS", "development_only": True, "test_accessed": False,
              "test_access_blocked": True, "preprocessed_files_opened": sorted(opened),
              "input_sha256": inputs_before, "split_hashes_verified": True,
              "test_seed": manifest["config"]["test_seed"], "development_seed": manifest["config"]["development_seed"],
              "all_five_fitted_models_use_same_cv": True, "cv_folds": 5, "cv_seed": 42,
              "floor_has_no_training_or_cv": True,
              "train_shape_baseline": list(original_train["X_raw"].shape),
              "train_shape_genre": list(train["X_raw"].shape), "validation_rows": len(validation["y"]),
              "frozen_graph_weights_and_top_k_verified_from_history": True,
              "history_features_rebuilt_with_frozen_graph": True,
              "continuous_feature_rebuild_max_abs_error": feature_max_error,
              "count_features_rebuilt_exactly": True,
              "cross_blas_near_tie_topology_difference": topology_difference,
              "genre_feature_definition": metadata["genre_feature_definition"],
              "genre_actual_scaler_fits_checked_this_run": actual_fit_checks,
              "saved_model_checks": metric_checks,
              "limitations": ["Validation thresholds are selected and scored on the same rows; scores are development evidence.",
                              "Bootstrap intervals condition on fitted models and thresholds, excluding selection/training uncertainty.",
                              "GNN and tabular feature representations differ; comparisons do not isolate graph propagation.",
                              "Test identity is checked using frozen split metadata/hashes, never test outcomes.",
                              "Original baseline/GNN training has prior full-run checks and current source/artifact audits; this command does not refit them."]}
    (output / "sanity.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Sanity audit passed: " + str(output / "sanity.json"), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs/genre_sanity")
    parser.add_argument("--run-genre", action="store_true", help="Run each genre model once with actual scaler-fit guards; outputs must not exist")
    args = parser.parse_args()
    audit(args.output_dir, args.run_genre)
