"""One bounded, symmetric training-only kNN grid extension; never open test.

The nine-feature and genre versions reuse knn.tune_and_fit unchanged. Original
inputs, methods and results are hashed before/after; outputs refuse overwrite.
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, make_output, save_result, write_csv
from common.dataset import DEFAULT_OUTPUT, load_development
from common.genre_features import load_genre_development
from common.preprocessing import sha256
from knn import knn_configuration, tune_and_fit


def extension_configuration():
    config = knn_configuration()
    plan = json.loads((BASE / "configs/knn_grid_extension.json").read_text(encoding="utf-8"))
    if config["n_neighbors_grid"] != [5, 15, 31, 63, 127] or plan["additional_neighbors"] != [255, 511, 1023]:
        raise ValueError("Unrecorded change to the bounded kNN grid")
    if config["cv_folds"] != 5 or config["cv_seed"] != 42:
        raise ValueError("The fixed five training folds must be retained")
    return {**config, "n_neighbors_grid": config["n_neighbors_grid"] + plan["additional_neighbors"]}, plan


def install_test_guard():
    root, opened = DEFAULT_OUTPUT.resolve(), set()

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if root in path.parents:
                relative = path.relative_to(root).as_posix()
                if relative == "test_inputs.npz" or relative.startswith("sealed/"):
                    raise PermissionError("kNN grid extension blocked real test access")
                opened.add(relative)

    sys.addaudithook(guard)
    return opened


def protected_files():
    paths = set()
    for name in ("movie_high_rating_rate", "logistic_regression", "knn", "gnn", "gnn_repeat_verified",
                 "logistic_regression_genre", "knn_genre", "gnn_extensions", "gnn_comparison", "genre_comparison"):
        paths.update(p for p in (BASE / "outputs" / name).rglob("*") if p.is_file())
    paths.update(DEFAULT_OUTPUT / name for name in ("train.npz", "validation.npz", "graph.npz", "manifest.json"))
    for directory, pattern in ((BASE, "*.py"), (BASE / "common", "*.py"), (BASE / "configs", "*.json")):
        paths.update(directory.glob(pattern))
    return {p.relative_to(BASE).as_posix(): sha256(p) for p in sorted(paths)}


def fit_with_checks(train, config):
    """Monitor every actual scaler fit without changing the existing estimator."""
    X, y = train["X_raw"].astype(np.float64), train["y"]
    folds = list(StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["cv_seed"]).split(X, y))
    expected = [fit for _ in config["n_neighbors_grid"] for _ in config["weights_grid"] for fit, _ in folds]
    expected.append(np.arange(len(y)))
    checked = []
    original_fit = StandardScaler.fit

    def monitored_fit(scaler, values, labels=None, **kwargs):
        if len(checked) >= len(expected):
            raise AssertionError("Unexpected scaler fit beyond the fixed CV/refit plan")
        indices = expected[len(checked)]
        np.testing.assert_array_equal(values, X[indices])
        np.testing.assert_array_equal(labels, y[indices])
        fitted = original_fit(scaler, values, labels, **kwargs)
        np.testing.assert_allclose(scaler.mean_, X[indices].mean(axis=0), rtol=0, atol=1e-12)
        checked.append({"n_fit": len(indices), "n_features": values.shape[1]})
        return fitted

    with patch.object(StandardScaler, "fit", monitored_fit):
        result = tune_and_fit(train, config)
    if len(checked) != len(expected):
        raise AssertionError("Missing scaler fits from the fixed CV/refit plan")
    return *result, checked


def legacy_cv_check(summaries, path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        old = list(csv.DictReader(stream))
    lookup = {(int(row[0]), row[1]): row[2] for row in summaries}
    records = [[int(row["k"]), row["weights"], float(row["mean_AP"]),
                lookup[int(row["k"]), row["weights"]]] for row in old]
    np.testing.assert_allclose([row[2] for row in records], [row[3] for row in records], rtol=0, atol=1e-12)
    return records


def run_representation(name, train, validation, config, metadata, root):
    output = make_output(name, root / name)
    old_name = "knn" if name == "history" else "knn_genre"
    protocol = {"representation": name, "config": config, "n_input_features": train["X_raw"].shape[1],
                "genre_metadata": metadata, "test_accessed": False,
                "selection": "Mean training-only five-fold AP; exact ties smaller k then uniform; no validation selection of k"}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    start = perf_counter()
    model, selected, records, summaries, fold_id, scaler_fits = fit_with_checks(train, config)
    seconds = perf_counter() - start
    write_csv(output / "cv_folds.csv", ["k", "weights", "fold", "n_fit", "n_heldout", "AP"], records)
    write_csv(output / "cv_summary.csv", ["k", "weights", "mean_AP", "fold_AP_std"], summaries)
    write_csv(output / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    if (output / "cv_assignments.csv").read_bytes() != (BASE / "outputs" / old_name / "cv_assignments.csv").read_bytes():
        raise ValueError("Expanded kNN training fold assignments differ from the original")
    legacy = legacy_cv_check(summaries, BASE / "outputs" / old_name / "cv_summary.csv")
    write_csv(output / "legacy_cv_check.csv", ["k", "weights", "original_mean_AP", "rerun_mean_AP"], legacy)
    scaler = model[0]
    np.savez_compressed(output / "model.npz", mean=scaler.mean_, scale=scaler.scale_,
                        X_fit=scaler.transform(train["X_raw"].astype(np.float64)), y_fit=train["y"],
                        feature_names=train["feature_names"], n_neighbors=np.asarray(selected["n_neighbors"]),
                        weights=np.asarray(selected["weights"]), metric=np.asarray(config["metric"]))
    with threadpool_limits(limits=config["thread_limit"]):
        scores = model.predict_proba(validation["X_raw"].astype(np.float64))[:, 1]
        with np.load(output / "model.npz", allow_pickle=False) as saved:
            restored = KNeighborsClassifier(n_neighbors=int(saved["n_neighbors"]), weights=str(saved["weights"]),
                                            metric=str(saved["metric"]), algorithm=config["algorithm"], n_jobs=config["n_jobs"])
            restored.fit(saved["X_fit"], saved["y_fit"])
            replay = restored.predict_proba((validation["X_raw"].astype(np.float64) - saved["mean"]) / saved["scale"])[:, 1]
            np.testing.assert_array_equal(saved["feature_names"], train["feature_names"])
        np.testing.assert_allclose(scores, replay, rtol=1e-14, atol=1e-14)
    (output / "scaler_fit_checks.json").write_text(json.dumps(scaler_fits, indent=2) + "\n", encoding="utf-8")
    result = save_result("knn_grid_extension", validation, scores, config, output, {
        **metadata, "representation": name, "n_input_features": train["X_raw"].shape[1],
        "selected_parameters": selected, "cv_mean_AP": max(row[2] for row in summaries),
        "fit_and_cv_seconds": seconds, "device": "CPU", "model_restore_verified": True,
        "legacy_CV_grid_reproduced": True, "actual_scaler_fits_verified": len(scaler_fits),
        "shared_training_code_sha256": {"knn.py": sha256(BASE / "knn.py")},
        "config_sha256": sha256(BASE / "configs/knn_grid_extension.json"),
        "limitation": "One exploratory bounded extension after original validation inspection; training-only CV selects k/weights; validation selects thresholds; conditional bootstrap omits training/selection uncertainty."})
    print(f"COMPLETE {name}: {selected}, CV AP={result['cv_mean_AP']:.8f}; {len(scaler_fits)} scaler fits verified", flush=True)
    return result


def run(output_dir=None):
    root = Path(output_dir) if output_dir else BASE / "outputs/knn_grid_extension"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {root}")
    opened = install_test_guard()
    config, plan = extension_configuration()
    before = protected_files()
    root.mkdir(parents=True)
    (root / "preservation_before.json").write_text(json.dumps(before, indent=2) + "\n", encoding="utf-8")
    (root / "experiment_plan.json").write_text(json.dumps({"config": config, **plan}, indent=2) + "\n", encoding="utf-8")
    results = {}
    try:
        train, validation = load_development()
        genre_train, genre_validation, metadata = load_genre_development()
        for basic, genre in ((train, genre_train), (validation, genre_validation)):
            for key in ("row_id", "user_id", "movie_id", "y"):
                np.testing.assert_array_equal(basic[key], genre[key])
            np.testing.assert_array_equal(basic["X_raw"], genre["X_raw"][:, :9])
        for name, fitting, scoring, extra in (("history", train, validation, {}),
                                              ("genre", genre_train, genre_validation, metadata)):
            print(f"START {name}: {fitting['X_raw'].shape}, grid={config['n_neighbors_grid']}", flush=True)
            results[name] = run_representation(name, fitting, scoring, config, extra, root)
    finally:
        after = protected_files()
        evidence = {"status": "PASS" if before == after else "FAIL", "original_files_preserved": before == after,
                    "protected_file_count": len(before), "before": before, "after": after,
                    "test_accessed": False, "preprocessed_files_opened": sorted(opened)}
        (root / "preservation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if before != after:
            raise AssertionError("Protected experiment/code files changed during the extension")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    results = run(args.output_dir)
    print(json.dumps({name: {k: result[k] for k in ("selected_parameters", "cv_mean_AP", "metrics")}
                      for name, result in results.items()}, indent=2))
