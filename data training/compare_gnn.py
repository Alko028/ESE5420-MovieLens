"""Compare aligned saved validation runs without loading any dataset or test set.

All methods keep their own validation-chosen threshold. Bootstrap differences
pair the same sampled users; they do not include model-fitting/selection
uncertainty and are not an independent final-test estimate.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.evaluation import METRICS, RankedMetrics


BASE = Path(__file__).resolve().parent
MODELS = ("movie_high_rating_rate", "logistic_regression", "knn", "gnn")
KEYS = ("row_id", "user_id", "movie_id", "y")
BOOTSTRAP_CONFIG = ("bootstrap_seed", "bootstrap_replicates", "confidence_level")


def _reject_constant(value):
    raise ValueError(f"Non-standard JSON number: {value}")


def _read_csv(path, required):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(required).issubset(reader.fieldnames):
            raise ValueError(f"Missing CSV columns in {path}: {required}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV: {path}")
    return rows


def _finite_number(value, description):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"Expected finite {description}")
    return float(value)


def _integer(value, description, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid {description}")
    return value


def _interval(samples, confidence):
    valid = samples[np.isfinite(samples)]
    alpha = (1 - confidence) / 2
    return {
        "lower": float(np.quantile(valid, alpha)) if len(valid) else None,
        "upper": float(np.quantile(valid, 1 - alpha)) if len(valid) else None,
        "valid_replicates": int(len(valid)),
        "undefined_replicates": int(len(samples) - len(valid)),
    }


def load_run(name, root):
    """Load and validate only saved validation artifacts for one model."""
    root = Path(root)
    result = json.loads((root / "result.json").read_text(encoding="utf-8"),
                        parse_constant=_reject_constant)
    if result.get("model") != name:
        raise ValueError(f"Model name mismatch for {name}")
    if result.get("evaluation_partition") != "validation" or result.get("test_accessed") is not False:
        raise ValueError(f"{name} is not an explicitly test-unaccessed validation run")
    config = result["config"]
    _integer(config["bootstrap_replicates"], "bootstrap replicates", 2)
    _integer(config["bootstrap_seed"], "bootstrap seed")
    confidence = _finite_number(config["confidence_level"], "confidence level")
    if not 0 < confidence < 1:
        raise ValueError("Confidence level must be in (0, 1)")
    hashes = result.get("input_sha256")
    if not isinstance(hashes, dict) or not {"train.npz", "validation.npz", "manifest.json"}.issubset(hashes):
        raise ValueError(f"Missing development input hashes for {name}")
    if any(not isinstance(h, str) or len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
           for h in hashes.values()):
        raise ValueError(f"Invalid development input hash for {name}")
    threshold = _finite_number(result["threshold"], "threshold")
    rows = _read_csv(root / "validation_predictions.csv", (*KEYS, "score", "prediction"))
    try:
        keys = [tuple(int(row[key]) for key in KEYS) for row in rows]
        scores = np.asarray([float(row["score"]) for row in rows], dtype=np.float64)
        prediction = np.asarray([int(row["prediction"]) for row in rows])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid prediction values for {name}") from exc
    if len({row[0] for row in keys}) != len(keys):
        raise ValueError(f"Duplicate prediction row IDs for {name}")
    if any(row[0] < 0 or row[1] < 1 or row[2] < 1 or row[3] not in (0, 1) for row in keys):
        raise ValueError(f"Invalid prediction keys for {name}")
    if not np.isfinite(scores).all() or not np.array_equal(prediction, (scores >= threshold).astype(int)):
        raise ValueError(f"Nonfinite scores or inconsistent threshold predictions for {name}")
    users = np.asarray([row[1] for row in keys])
    if result.get("n_samples") != len(keys) or result.get("n_users") != len(np.unique(users)):
        raise ValueError(f"Saved prediction counts differ from result for {name}")
    actual = RankedMetrics(np.asarray([row[3] for row in keys]), scores).evaluate(threshold)
    for i, metric in enumerate(METRICS):
        estimate = _finite_number(result["metrics"][metric], f"{name} {metric}")
        if not np.isclose(estimate, actual[i], rtol=1e-10, atol=1e-12):
            raise ValueError(f"Saved {metric} disagrees with predictions for {name}")
    with np.load(root / "bootstrap.npz", allow_pickle=False) as values:
        if not {"samples", "metric_names", "user_ids"}.issubset(values.files):
            raise ValueError(f"Missing bootstrap arrays for {name}")
        bootstrap = {key: values[key] for key in ("samples", "metric_names", "user_ids")}
    if not np.array_equal(bootstrap["metric_names"], np.asarray(METRICS)):
        raise ValueError(f"Unexpected bootstrap metric names for {name}")
    if not np.array_equal(bootstrap["user_ids"], np.unique(users)):
        raise ValueError(f"Bootstrap user universe disagrees with predictions for {name}")
    samples = bootstrap["samples"]
    if (not np.issubdtype(samples.dtype, np.floating) or
            samples.shape != (config["bootstrap_replicates"], len(METRICS)) or np.isinf(samples).any()):
        raise ValueError(f"Invalid bootstrap sample shape or values for {name}")
    for i, metric in enumerate(METRICS):
        expected = _interval(samples[:, i], confidence)
        saved = result["intervals"][metric]
        for key, value in expected.items():
            if value is None:
                agrees = saved.get(key) is None
            elif key.endswith("replicates"):
                agrees = saved.get(key) == value
            else:
                agrees = saved.get(key) is not None and np.isclose(saved[key], value, rtol=1e-10, atol=1e-12)
            if not agrees:
                raise ValueError(f"Saved bootstrap interval disagrees with samples for {name}: {metric}/{key}")
    if name == "gnn":
        for field in ("all_training_losses_finite", "model_restore_verified"):
            if result.get(field) is not True:
                raise ValueError(f"GNN validation run failed the required check: {field}")
    return {"result": result, "keys": keys, "bootstrap": bootstrap, "root": root}


def compare_runs(run_directories, output_dir):
    """Validate pairing and persist a new comparison; never overwrite a run."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output_dir}")
    if set(run_directories) != set(MODELS):
        raise ValueError(f"Expected exactly these models: {MODELS}")
    runs = {name: load_run(name, run_directories[name]) for name in MODELS}
    reference = runs["logistic_regression"]
    config = reference["result"]["config"]
    for name, run in runs.items():
        result = run["result"]
        if result["input_sha256"] != reference["result"]["input_sha256"]:
            raise ValueError(f"Development input hashes differ: {name}")
        if run["keys"] != reference["keys"]:
            raise ValueError(f"Unpaired validation rows or row ordering: {name}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall"):
            if result["config"][key] != config[key]:
                raise ValueError(f"Unpaired evaluation configuration: {name}/{key}")
        for key in ("metric_names", "user_ids"):
            if not np.array_equal(run["bootstrap"][key], reference["bootstrap"][key]):
                raise ValueError(f"Unpaired bootstrap {key}: {name}")
    assignments = {}
    for name in ("logistic_regression", "knn", "gnn"):
        result = runs[name]["result"]
        if any(result["config"][key] != config[key] for key in ("cv_seed", "cv_folds")):
            raise ValueError(f"Training CV configuration differs: {name}")
        rows = _read_csv(runs[name]["root"] / "cv_assignments.csv", ("row_id", "fold"))
        try:
            fold_rows = [(int(row["row_id"]), int(row["fold"])) for row in rows]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid CV assignments for {name}") from exc
        if (len({row[0] for row in fold_rows}) != len(fold_rows) or
                any(row[0] < 0 or not 0 <= row[1] < config["cv_folds"] for row in fold_rows) or
                {row[1] for row in fold_rows} != set(range(config["cv_folds"]))):
            raise ValueError(f"Invalid CV assignments for {name}")
        assignments[name] = rows
        if rows != assignments["logistic_regression"]:
            raise ValueError(f"Training CV assignments differ: {name}")
    differences = {}
    gnn = runs["gnn"]
    for other in ("logistic_regression", "knn", "movie_high_rating_rate"):
        delta = gnn["bootstrap"]["samples"] - runs[other]["bootstrap"]["samples"]
        differences["gnn_minus_" + other] = {
            metric: {
                "estimate": float(gnn["result"]["metrics"][metric] - runs[other]["result"]["metrics"][metric]),
                **_interval(delta[:, i], config["confidence_level"]),
            }
            for i, metric in enumerate(METRICS)
        }
    evidence = {
        "evaluation_partition": "validation",
        "primary_comparison": "gnn_minus_logistic_regression",
        "primary_metric": "precision",
        "paired_differences": differences,
        "validation_rows_aligned": True,
        "validation_sample_count": len(reference["keys"]),
        "training_cv_assignments_identical": True,
        "input_hashes_identical": True,
        "input_sha256": reference["result"]["input_sha256"],
        "bootstrap_configuration": {key: config[key] for key in BOOTSTRAP_CONFIG},
        "gnn_training_checks": {key: gnn["result"][key] for key in (
            "all_training_losses_finite", "model_restore_verified")},
        "interval_method": "paired user-bootstrap percentile intervals; each model's fixed validation-chosen threshold",
        "limitation": "Validation selects thresholds; intervals condition on fitted models and exclude training/selection uncertainty.",
        "test_accessed": False,
    }
    # Serialize strictly before creating output, so invalid values leave no artifact.
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
    parser.add_argument("--gnn-dir", type=Path, default=BASE / "outputs/gnn")
    parser.add_argument("--logistic-regression-dir", type=Path, default=BASE / "outputs/logistic_regression")
    parser.add_argument("--knn-dir", type=Path, default=BASE / "outputs/knn")
    parser.add_argument("--movie-high-rating-rate-dir", type=Path, default=BASE / "outputs/movie_high_rating_rate")
    parser.add_argument("--output-dir", type=Path, default=BASE / "outputs/gnn_comparison")
    args = parser.parse_args()
    directories = {name: getattr(args, name + "_dir") for name in MODELS}
    print(json.dumps(compare_runs(directories, args.output_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
