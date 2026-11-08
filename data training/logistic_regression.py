"""L2 logistic regression: train-only 5-fold AP tuning, untouched validation scoring."""

import argparse
import json
import warnings
from time import perf_counter

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common.baseline_run import configuration, load_development, make_output, save_result, write_csv


def model_for(C, config):
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=C, penalty="l2", solver=config["solver"], max_iter=config["max_iter"], tol=config["tol"]))


def tune_and_fit(train, config):
    # Use RAW features so each CV fold fits its own scaler. Do not use pre-scaled X.
    X, y = train["X_raw"].astype(np.float64), train["y"]
    splitter = StratifiedKFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["cv_seed"])
    folds = list(splitter.split(X, y))
    records, means = [], []
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for C in sorted(config["C_grid"]):
            values = []
            for fold, (fit, heldout) in enumerate(folds):
                model = model_for(C, config)
                model.fit(X[fit], y[fit])
                ap = average_precision_score(y[heldout], model.predict_proba(X[heldout])[:, 1])
                values.append(float(ap))
                records.append([C, fold, len(fit), len(heldout), float(ap), int(model[-1].n_iter_[0])])
            means.append([C, float(np.mean(values)), float(np.std(values, ddof=1))])
        # Stable exact ties favor smaller C (stronger regularization).
        best_C = max(means, key=lambda row: (row[1], -row[0]))[0]
        final = model_for(best_C, config)
        final.fit(X, y)
    fold_id = np.empty(len(y), dtype=np.int8)
    for i, (_, heldout) in enumerate(folds):
        fold_id[heldout] = i
    return final, best_C, records, means, fold_id


def run(output_dir=None):
    config = configuration()
    train, validation = load_development()
    output = make_output("logistic_regression", output_dir)
    start = perf_counter()
    model, C, records, means, fold_id = tune_and_fit(train, config)
    elapsed = perf_counter() - start
    write_csv(output / "cv_folds.csv", ["C", "fold", "n_fit", "n_heldout", "AP", "iterations"], records)
    write_csv(output / "cv_summary.csv", ["C", "mean_AP", "fold_AP_std"], means)
    write_csv(output / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    scaler, classifier = model[0], model[-1]
    np.savez_compressed(output / "model.npz", mean=scaler.mean_, scale=scaler.scale_,
                        coef=classifier.coef_, intercept=classifier.intercept_, classes=classifier.classes_,
                        feature_names=train["feature_names"], C=np.asarray(C))
    scores = model.predict_proba(validation["X_raw"].astype(np.float64))[:, 1]
    # Check the portable, pickle-free model representation reproduces its predictions.
    from scipy.special import expit
    restored = expit(((validation["X_raw"] - scaler.mean_) / scaler.scale_) @ classifier.coef_[0] + classifier.intercept_[0])
    np.testing.assert_allclose(restored, scores, rtol=1e-12, atol=1e-12)
    return save_result("logistic_regression", validation, scores, config, output,
                       {"fit_and_cv_seconds": elapsed, "selected_C": C,
                        "cv_mean_AP": next(row[1] for row in means if row[0] == C),
                        "cv": "5-fold stratified shuffled training-target rows; fold-local scaling; mean AP selection",
                        "final_fit_iterations": int(classifier.n_iter_[0]), "convergence_warnings": 0})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    print(json.dumps(run(args.output_dir), indent=2))
