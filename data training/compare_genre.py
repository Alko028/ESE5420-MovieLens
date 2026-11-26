"""Compare paired history-only and history-plus-genre validation runs.

Read saved validation artifacts only. The primary ablations keep each method's
training-only hyperparameter grid unchanged, so genre is the feature change.
All thresholds are already fixed by validation and are not reselected here.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.evaluation import METRICS
from compare_gnn import BOOTSTRAP_CONFIG, _integer, _interval, _read_csv, load_run


BASE = Path(__file__).resolve().parent
MODELS = ("logistic_regression", "knn", "logistic_regression_genre", "knn_genre")
INPUTS = ("train.npz", "validation.npz", "manifest.json")
GENRE_MODELS = MODELS[2:]
PAIRS = {
    "lr_genre_minus_lr": ("logistic_regression_genre", "logistic_regression"),
    "knn_genre_minus_knn": ("knn_genre", "knn"),
    "knn_genre_minus_lr_genre": ("knn_genre", "logistic_regression_genre"),
    "knn_genre_minus_lr": ("knn_genre", "logistic_regression"),
}
HPO_KEYS = {
    "logistic_regression": ("C_grid", "solver", "max_iter", "tol"),
    "knn": ("n_neighbors_grid", "weights_grid", "metric", "algorithm", "n_jobs", "thread_limit"),
}


def _sha256(value, field):
    if (not isinstance(value, str) or len(value) != 64 or
            any(character not in "0123456789abcdef" for character in value)):
        raise ValueError(f"Invalid genre provenance: {field}")
    return value


def _check_genre_metadata(runs):
    reference = runs[GENRE_MODELS[0]]["result"]
    for name in GENRE_MODELS:
        result = runs[name]["result"]
        if type(result.get("n_input_features")) is not int or result["n_input_features"] != 66:
            raise ValueError(f"Expected 66 aligned input features for {name}")
        for field in ("model_restore_verified", "base_features_unchanged", "development_alignment_verified"):
            if result.get(field) is not True:
                raise ValueError(f"Required genre check failed: {name}/{field}")
        for field in ("graph_sha256", "genre_config_sha256"):
            _sha256(result.get(field), field)
            if result[field] != reference[field]:
                raise ValueError(f"Genre feature provenance differs: {name}/{field}")
        definition = result.get("genre_feature_definition")
        if not isinstance(definition, (str, list, dict)) or not definition:
            raise ValueError(f"Missing genre feature definition for {name}")
        if definition != reference.get("genre_feature_definition"):
            raise ValueError(f"Genre feature definitions differ: {name}")
    return {field: reference[field] for field in (
        "n_input_features", "graph_sha256", "genre_config_sha256", "genre_feature_definition")}


def _check_method_configuration(runs):
    for base_name in MODELS[:2]:
        base = runs[base_name]["result"]["config"]
        genre = runs[base_name + "_genre"]["result"]["config"]
        for key in HPO_KEYS[base_name]:
            if key not in base or key not in genre or genre[key] != base[key]:
                raise ValueError(f"Method HPO configuration differs: {base_name}/{key}")
        # Also reject unnoticed changes to other saved baseline settings. New
        # feature provenance belongs outside config and can be stored separately.
        for key, value in base.items():
            if key not in genre or genre[key] != value:
                raise ValueError(f"Method configuration differs: {base_name}/{key}")


def compare_runs(run_directories, output_dir):
    """Validate all pairing and write a new comparison, refusing any overwrite."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output_dir}")
    if set(run_directories) != set(MODELS):
        raise ValueError(f"Expected exactly these models: {MODELS}")
    runs = {name: load_run(name, run_directories[name]) for name in MODELS}
    reference = runs["logistic_regression"]
    config = reference["result"]["config"]
    _integer(config["cv_folds"], "CV folds", 2)
    _integer(config["cv_seed"], "CV seed")
    reference_hashes = {key: reference["result"]["input_sha256"][key] for key in INPUTS}
    fold_reference = None
    for name, run in runs.items():
        result = run["result"]
        if any(result["input_sha256"][key] != reference_hashes[key] for key in INPUTS):
            raise ValueError(f"Development input hashes differ: {name}")
        if run["keys"] != reference["keys"]:
            raise ValueError(f"Unpaired validation rows or row ordering: {name}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall", "cv_folds", "cv_seed"):
            if result["config"].get(key) != config[key]:
                raise ValueError(f"Unpaired evaluation or CV configuration: {name}/{key}")
        for key in ("metric_names", "user_ids"):
            if not np.array_equal(run["bootstrap"][key], reference["bootstrap"][key]):
                raise ValueError(f"Unpaired bootstrap {key}: {name}")
        rows = _read_csv(run["root"] / "cv_assignments.csv", ("row_id", "fold"))
        try:
            folds = [(int(row["row_id"]), int(row["fold"])) for row in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid training CV assignments: {name}") from exc
        if (len({row[0] for row in folds}) != len(folds) or
                any(row[0] < 0 or not 0 <= row[1] < config["cv_folds"] for row in folds) or
                {row[1] for row in folds} != set(range(config["cv_folds"]))):
            raise ValueError(f"Invalid training CV assignments: {name}")
        if fold_reference is None:
            fold_reference = folds
        elif folds != fold_reference:
            raise ValueError(f"Training CV assignments differ: {name}")
    _check_method_configuration(runs)
    feature_evidence = _check_genre_metadata(runs)
    differences = {}
    for label, (left_name, right_name) in PAIRS.items():
        left, right = runs[left_name], runs[right_name]
        delta = left["bootstrap"]["samples"] - right["bootstrap"]["samples"]
        differences[label] = {
            metric: {
                "estimate": float(left["result"]["metrics"][metric] - right["result"]["metrics"][metric]),
                **_interval(delta[:, i], config["confidence_level"]),
            } for i, metric in enumerate(METRICS)
        }
    evidence = {
        "evaluation_partition": "validation",
        "primary_comparisons": ["lr_genre_minus_lr", "knn_genre_minus_knn"],
        "primary_metric": "precision",
        "paired_differences": differences,
        "validation_rows_aligned": True,
        "validation_sample_count": len(reference["keys"]),
        "training_target_count": len(fold_reference),
        "training_cv_assignments_identical": True,
        "input_hashes_identical": True,
        "input_sha256": reference_hashes,
        "within_method_configuration_unchanged": True,
        "genre_feature_provenance_identical": True,
        "genre_features": feature_evidence,
        "bootstrap_configuration": {key: config[key] for key in BOOTSTRAP_CONFIG},
        "interval_method": "paired user-bootstrap percentile intervals; each model's fixed validation-chosen threshold",
        "limitation": "Validation selects thresholds; intervals condition on fitted models and exclude training/selection uncertainty.",
        "test_accessed": False,
    }
    # Strict serialization precedes mutation, so rejected runs leave no output.
    encoded = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("model", "metric", "estimate", "ci_lower", "ci_upper", "threshold", "partition"))
        for name in MODELS:
            result = runs[name]["result"]
            for metric in METRICS:
                interval = result["intervals"][metric]
                writer.writerow((name, metric, result["metrics"][metric], interval["lower"], interval["upper"],
                                 result["threshold"], "validation"))
    (output_dir / "paired_comparison.json").write_text(encoded, encoding="utf-8")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr-dir", type=Path, default=BASE / "outputs/logistic_regression")
    parser.add_argument("--knn-dir", type=Path, default=BASE / "outputs/knn")
    parser.add_argument("--lr-genre-dir", type=Path, default=BASE / "outputs/logistic_regression_genre")
    parser.add_argument("--knn-genre-dir", type=Path, default=BASE / "outputs/knn_genre")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs/genre_comparison")
    args = parser.parse_args()
    directories = dict(zip(MODELS, (args.lr_dir, args.knn_dir, args.lr_genre_dir, args.knn_genre_dir)))
    print(json.dumps(compare_runs(directories, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
