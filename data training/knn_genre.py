"""History-plus-genre kNN, using the original baseline's tuning protocol."""

import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, make_output, save_result, write_csv
from common.dataset import DEFAULT_OUTPUT
from common.genre_features import load_genre_development
from common.preprocessing import sha256
from knn import knn_configuration, tune_and_fit


def run(output_dir=None):
    config = knn_configuration()
    train, validation, metadata = load_genre_development()
    output = make_output("knn_genre", output_dir)
    input_sha = {name: sha256(DEFAULT_OUTPUT / name)
                 for name in ("train.npz", "validation.npz", "manifest.json")}
    protocol = {
        "model": "knn_genre",
        "config": config,
        "genre_metadata": metadata,
        "input_sha256": input_sha,
        "training_scope": "Existing training targets only; original stratified 5-fold assignments; fold-local scaling; mean AP selection",
        "base_features_unchanged": metadata["base_features_unchanged"],
        "development_alignment_verified": metadata["development_alignment_verified"],
        "n_input_features": int(train["X_raw"].shape[1]),
        "shared_training_code_sha256": {"knn.py": sha256(BASE / "knn.py")},
        "test_accessed": False,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    start = perf_counter()
    model, selected, records, summaries, fold_id = tune_and_fit(train, config)
    fit_seconds = perf_counter() - start
    write_csv(output / "cv_folds.csv", ["k", "weights", "fold", "n_fit", "n_heldout", "AP"], records)
    write_csv(output / "cv_summary.csv", ["k", "weights", "mean_AP", "fold_AP_std"], summaries)
    write_csv(output / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    scaler = model[0]
    raw_train = train["X_raw"].astype(np.float64)
    np.savez_compressed(
        output / "model.npz", mean=scaler.mean_, scale=scaler.scale_,
        X_fit=scaler.transform(raw_train), y_fit=train["y"],
        feature_names=np.asarray(train["feature_names"], dtype=str),
        n_neighbors=np.asarray(selected["n_neighbors"]),
        weights=np.asarray(selected["weights"]), metric=np.asarray(config["metric"]))
    prediction_start = perf_counter()
    raw_validation = validation["X_raw"].astype(np.float64)
    with threadpool_limits(limits=config["thread_limit"]):
        scores = model.predict_proba(raw_validation)[:, 1]
        with np.load(output / "model.npz", allow_pickle=False) as saved:
            restored = KNeighborsClassifier(
                n_neighbors=int(saved["n_neighbors"]), weights=str(saved["weights"]),
                metric=str(saved["metric"]), algorithm=config["algorithm"], n_jobs=1)
            restored.fit(saved["X_fit"], saved["y_fit"])
            replay = restored.predict_proba((raw_validation - saved["mean"]) / saved["scale"])[:, 1]
            np.testing.assert_array_equal(saved["feature_names"], train["feature_names"])
        np.testing.assert_allclose(scores, replay, atol=1e-14, rtol=1e-14)
    return save_result("knn_genre", validation, scores, config, output, {
        "selected_parameters": selected,
        "cv_mean_AP": next(row[2] for row in summaries if row[:2] == [selected["n_neighbors"], selected["weights"]]),
        "fit_and_cv_seconds": fit_seconds,
        "prediction_and_restore_check_seconds": perf_counter() - prediction_start,
        "cv": "5-fold stratified shuffled training-target rows; fold-local scaling; mean AP selection",
        "limitation": "validation selects threshold; k and weights selected by training-only CV; intervals exclude training/selection uncertainty",
        "device": "CPU",
        "model_restore_verified": True,
        "n_input_features": int(train["X_raw"].shape[1]),
        "shared_training_code_sha256": protocol["shared_training_code_sha256"],
        **metadata,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({key: result[key] for key in (
        "selected_parameters", "cv_mean_AP", "metrics", "intervals", "threshold", "n_input_features", "test_accessed")}, indent=2))
