"""Shared result persistence and strict development-only input loading."""

import csv
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
import sklearn

from .dataset import DEFAULT_OUTPUT, load_development
from .evaluation import METRICS, RankedMetrics, bootstrap_users, choose_threshold
from .preprocessing import sha256

BASE = Path(__file__).resolve().parents[1]


def configuration():
    return json.loads((BASE / "configs" / "baseline.json").read_text(encoding="utf-8"))


def write_csv(path, columns, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def save_result(name, validation, scores, config, output, extra=None):
    start = perf_counter()
    threshold = choose_threshold(validation["y"], scores, config["minimum_validation_recall"])
    ranked = RankedMetrics(validation["y"], scores)
    point = dict(zip(METRICS, ranked.evaluate(threshold).tolist()))
    samples, intervals, users = bootstrap_users(
        validation["y"], scores, validation["user_id"], threshold,
        config["bootstrap_replicates"], config["bootstrap_seed"], config["confidence_level"])
    prediction = (scores >= threshold).astype(np.int8)
    write_csv(output / "validation_predictions.csv", ["row_id", "user_id", "movie_id", "y", "score", "prediction"],
              zip(validation["row_id"], validation["user_id"], validation["movie_id"],
                  validation["y"], scores, prediction))
    np.savez_compressed(output / "bootstrap.npz", samples=samples, metric_names=np.asarray(METRICS), user_ids=users)
    t, p, r = ranked.curve()
    write_csv(output / "validation_pr_curve.csv", ["threshold", "precision", "recall"], zip(t, p, r))
    result = {"model": name, "evaluation_partition": "validation", "n_samples": len(scores),
              "n_users": len(users), "threshold": threshold, "metrics": point, "intervals": intervals,
              "positive_predictions": int(prediction.sum()),
              "user_coverage": len(np.unique(validation["user_id"][prediction == 1])) / len(users),
              "config": config, "threshold_rule": "score >= threshold; maximize precision subject to recall >= 0.5; ties: larger recall then larger threshold",
              "bootstrap_method": "paired users, percentile 95% CI; fixed fitted model and threshold; pooled rating-level metrics",
              "limitation": "validation selects threshold; C is selected by training-only CV; intervals exclude training and selection uncertainty",
              "test_accessed": False, "evaluation_seconds": perf_counter() - start,
              "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__},
              "input_sha256": {file: sha256(DEFAULT_OUTPUT / file) for file in ("train.npz", "validation.npz", "manifest.json")},
              "code_sha256": {str(p.relative_to(BASE)).replace("\\", "/"): sha256(p)
                              for p in [BASE / f"{name}.py", *sorted((BASE / "common").glob("*.py"))]},
              **(extra or {})}
    (output / "result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


def make_output(name, path=None):
    output = Path(path) if path else BASE / "outputs" / name
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite run: {output}; use --output-dir for an independent reproduction")
    output.mkdir(parents=True)
    return output
