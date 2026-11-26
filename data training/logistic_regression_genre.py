"""History-plus-genre L2 LR, using the original baseline's tuning protocol."""

import argparse
import json
from time import perf_counter

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, configuration, make_output, save_result, write_csv
from common.dataset import DEFAULT_OUTPUT
from common.genre_features import load_genre_development
from common.preprocessing import sha256
from logistic_regression import tune_and_fit


def run(output_dir=None):
    config = configuration()
    train, validation, metadata = load_genre_development()
    output = make_output("logistic_regression_genre", output_dir)
    input_sha = {name: sha256(DEFAULT_OUTPUT / name)
                 for name in ("train.npz", "validation.npz", "manifest.json")}
    protocol = {
        "model": "logistic_regression_genre",
        "config": config,
        "genre_metadata": metadata,
        "input_sha256": input_sha,
        "training_scope": "Existing training targets only; original stratified 5-fold assignments; fold-local scaling; mean AP selection",
        "base_features_unchanged": metadata["base_features_unchanged"],
        "development_alignment_verified": metadata["development_alignment_verified"],
        "n_input_features": int(train["X_raw"].shape[1]),
        "training_thread_limit": 4,
        "shared_training_code_sha256": {"logistic_regression.py": sha256(BASE / "logistic_regression.py")},
        "test_accessed": False,
    }
    # Persist the fixed inputs/configuration before fitting or validation scoring.
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    start = perf_counter()
    with threadpool_limits(limits=4):
        model, C, records, means, fold_id = tune_and_fit(train, config)
    elapsed = perf_counter() - start
    write_csv(output / "cv_folds.csv", ["C", "fold", "n_fit", "n_heldout", "AP", "iterations"], records)
    write_csv(output / "cv_summary.csv", ["C", "mean_AP", "fold_AP_std"], means)
    write_csv(output / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    scaler, classifier = model[0], model[-1]
    np.savez_compressed(
        output / "model.npz", mean=scaler.mean_, scale=scaler.scale_,
        coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_,
        feature_names=np.asarray(train["feature_names"], dtype=str), C=np.asarray(C))
    raw_validation = validation["X_raw"].astype(np.float64)
    with threadpool_limits(limits=4):
        scores = model.predict_proba(raw_validation)[:, 1]
    # Check the representation actually written to disk, not in-memory fitted values.
    with np.load(output / "model.npz", allow_pickle=False) as saved:
        restored = expit(((raw_validation - saved["mean"]) / saved["scale"]) @ saved["coef"][0] + saved["intercept"][0])
        np.testing.assert_array_equal(saved["feature_names"], train["feature_names"])
    np.testing.assert_allclose(restored, scores, rtol=1e-12, atol=1e-12)
    return save_result("logistic_regression_genre", validation, scores, config, output, {
        "fit_and_cv_seconds": elapsed,
        "selected_C": C,
        "cv_mean_AP": next(row[1] for row in means if row[0] == C),
        "cv": "5-fold stratified shuffled training-target rows; fold-local scaling; mean AP selection",
        "final_fit_iterations": int(classifier.n_iter_[0]),
        "convergence_warnings": 0,
        "device": "CPU",
        "training_thread_limit": 4,
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
        "selected_C", "cv_mean_AP", "metrics", "intervals", "threshold", "n_input_features", "test_accessed")}, indent=2))
