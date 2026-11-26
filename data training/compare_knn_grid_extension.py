"""Compare a bounded kNN grid extension from saved validation artifacts only."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.baseline_run import write_csv
from common.evaluation import METRICS
from compare_gnn import BOOTSTRAP_CONFIG, _interval, _read_csv, load_run

BASE = Path(__file__).resolve().parent
NAMES = ("lr", "old_history", "new_history", "old_genre", "new_genre")
LABELS = {"lr": "Logistic regression", "old_history": "kNN original grid", "new_history": "kNN expanded grid",
          "old_genre": "kNN + genre original grid", "new_genre": "kNN + genre expanded grid"}
PAIRS = {"expanded_history_minus_original": ("new_history", "old_history"),
         "expanded_genre_minus_original": ("new_genre", "old_genre"),
         "genre_minus_history_expanded_grid": ("new_genre", "new_history"),
         "expanded_history_minus_LR": ("new_history", "lr"),
         "expanded_genre_minus_LR": ("new_genre", "lr")}
LIMITATION = ("Validation results after one bounded exploratory grid extension; training-only CV selects k/weights. "
              "Validation chooses each operating threshold. Paired user-bootstrap intervals condition on fitted "
              "models/thresholds, exclude selection uncertainty, and are unadjusted for multiple comparisons. "
              "The sealed test remains untouched; stop after the recorded maximum even if selected.")


def validate_alignment(runs):
    reference = runs["lr"]
    config = reference["result"]["config"]
    expected_folds = _read_csv(reference["root"] / "cv_assignments.csv", ("row_id", "fold"))
    pairs = [(int(row["row_id"]), int(row["fold"])) for row in expected_folds]
    if (len({row[0] for row in pairs}) != len(pairs) or any(row[0] < 0 for row in pairs) or
            {row[1] for row in pairs} != set(range(config["cv_folds"]))):
        raise ValueError("Invalid reference CV assignments")
    for name, run in runs.items():
        result = run["result"]
        if result["input_sha256"] != reference["result"]["input_sha256"] or run["keys"] != reference["keys"]:
            raise ValueError(f"Unpaired validation inputs/rows: {name}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall", "cv_folds", "cv_seed"):
            if result["config"].get(key) != config[key]:
                raise ValueError(f"CV/bootstrap configuration differs: {name}/{key}")
        for key in ("metric_names", "user_ids"):
            if not np.array_equal(run["bootstrap"][key], reference["bootstrap"][key]):
                raise ValueError(f"Unpaired bootstrap {key}: {name}")
        if _read_csv(run["root"] / "cv_assignments.csv", ("row_id", "fold")) != expected_folds:
            raise ValueError(f"Training fold assignments differ: {name}")
    old = runs["old_history"]["result"]["config"]
    expected = {**old, "n_neighbors_grid": [5, 15, 31, 63, 127, 255, 511, 1023]}
    if old["n_neighbors_grid"] != [5, 15, 31, 63, 127] or runs["old_genre"]["result"]["config"] != old:
        raise ValueError("Original kNN grids/settings differ")
    for name, features in (("new_history", 9), ("new_genre", 66)):
        result = runs[name]["result"]
        if result["config"] != expected or result.get("n_input_features") != features:
            raise ValueError(f"Expanded grids/settings/features differ: {name}")
        if result.get("representation") != ("history" if features == 9 else "genre"):
            raise ValueError(f"Representation metadata differs: {name}")
        for key in ("model_restore_verified", "legacy_CV_grid_reproduced"):
            if result.get(key) is not True:
                raise ValueError(f"Training check failed: {name}/{key}")
        if result.get("actual_scaler_fits_verified") != 81:
            raise ValueError(f"Actual scaler fit checks missing: {name}")
    for key in ("graph_sha256", "genre_config_sha256", "genre_feature_definition", "feature_matrix_sha256"):
        if (not runs["old_genre"]["result"].get(key) or
                runs["new_genre"]["result"].get(key) != runs["old_genre"]["result"].get(key)):
            raise ValueError(f"Genre provenance differs: {key}")
    for key in ("base_features_unchanged", "development_alignment_verified"):
        if runs["new_genre"]["result"].get(key) is not True or runs["old_genre"]["result"].get(key) is not True:
            raise ValueError(f"Genre alignment check failed: {key}")
    return len(expected_folds)


def paired_differences(runs):
    confidence = runs["lr"]["result"]["config"]["confidence_level"]
    differences = {}
    for label, (left, right) in PAIRS.items():
        samples = runs[left]["bootstrap"]["samples"] - runs[right]["bootstrap"]["samples"]
        differences[label] = {metric: {
            "estimate": runs[left]["result"]["metrics"][metric] - runs[right]["result"]["metrics"][metric],
            **_interval(samples[:, i], confidence)} for i, metric in enumerate(METRICS)}
    return differences


def _plots(runs, evidence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.1, 4.3))
    for name, color in (("new_history", "#2471a3"), ("new_genre", "#b03a2e")):
        rows = _read_csv(runs[name]["root"] / "cv_summary.csv", ("k", "weights", "mean_AP"))
        for weights, style in (("uniform", "--"), ("distance", "-")):
            values = [row for row in rows if row["weights"] == weights]
            ax.plot([int(r["k"]) for r in values], [float(r["mean_AP"]) for r in values],
                    style, marker="o", color=color, label=f"{'9 history inputs' if name == 'new_history' else '66 inputs with genre'}, {weights}")
    ax.axvline(127, color="gray", linestyle=":", label="Original upper bound (127)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Number of neighbors k (log scale)")
    ax.set_ylabel("Mean training-only 5-fold AP")
    ax.set_title("One bounded kNN grid extension")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"training_cv_AP.{suffix}", dpi=180)
    plt.close(fig)
    selected = tuple(PAIRS)[:4]
    fig, ax = plt.subplots(figsize=(7.8, 4.3))
    labels = ["Expanded - original: 9 inputs", "Expanded - original: genre", "Genre - history: expanded grid", "Expanded history - LR"]
    for index, name in enumerate(selected):
        item = evidence["paired_differences"][name]["precision"]
        ax.errorbar(100 * item["estimate"], index,
                    xerr=[[100 * (item["estimate"] - item["lower"])], [100 * (item["upper"] - item["estimate"])]], fmt="o", capsize=4)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_yticks(range(len(selected)), labels)
    ax.invert_yaxis()
    ax.set_xlabel("Validation precision difference (percentage points), paired 95% CI")
    ax.set_title("Fixed validation-selected thresholds; exploratory comparisons")
    ax.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"paired_precision_differences.{suffix}", dpi=180)
    plt.close(fig)


def compare_runs(root=None, output_dir=None):
    root = Path(root) if root else BASE / "outputs/knn_grid_extension"
    output = Path(output_dir) if output_dir else root / "comparison"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    preservation = json.loads((root / "preservation.json").read_text(encoding="utf-8"))
    if (preservation.get("status") != "PASS" or preservation.get("original_files_preserved") is not True or
            preservation.get("before") != preservation.get("after") or preservation.get("test_accessed") is not False):
        raise ValueError("Original artifacts were not preserved")
    runs = {"lr": load_run("logistic_regression", BASE / "outputs/logistic_regression"),
            "old_history": load_run("knn", BASE / "outputs/knn"),
            "old_genre": load_run("knn_genre", BASE / "outputs/knn_genre"),
            "new_history": load_run("knn_grid_extension", root / "history"),
            "new_genre": load_run("knn_grid_extension", root / "genre")}
    n_train = validate_alignment(runs)
    config = runs["lr"]["result"]["config"]
    evidence = {"evaluation_partition": "validation", "test_accessed": False,
                "training_targets": n_train, "validation_targets": len(runs["lr"]["keys"]),
                "same_inputs_rows_folds_and_bootstrap": True, "legacy_CV_grid_reproduced": True,
                "original_artifacts_preserved": True, "protected_file_count": preservation["protected_file_count"],
                "bootstrap_configuration": {key: config[key] for key in BOOTSTRAP_CONFIG},
                "paired_differences": paired_differences(runs), "limitation": LIMITATION}
    encoded = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    table = ["# Bounded kNN grid extension — validation", "",
             "All intervals are 95% paired-user bootstrap percentile intervals. Each method retains its own validation-selected threshold.", "",
             "| Model | Selected k/weights | CV AP | Precision [95% CI] | Recall | AP [95% CI] |",
             "|---|---|---:|---:|---:|---:|"]
    rows = []
    for name in NAMES:
        r = runs[name]["result"]
        selected = r["selected_parameters"] if name != "lr" else None
        settings = f"{selected['n_neighbors']}/{selected['weights']}" if selected else "C=0.1"
        def interval(metric):
            q = r["intervals"][metric]
            return f"{r['metrics'][metric]:.6f} [{q['lower']:.6f}, {q['upper']:.6f}]"
        table.append(f"| {LABELS[name]} | {settings} | {r['cv_mean_AP']:.6f} | {interval('precision')} | {r['metrics']['recall']:.6f} | {interval('AP')} |")
        for metric in METRICS:
            rows.append([name, metric, r["metrics"][metric], r["intervals"][metric]["lower"],
                         r["intervals"][metric]["upper"], r["threshold"], settings, r["cv_mean_AP"]])
    table.extend(["", LIMITATION, ""])
    delta = ["# Paired differences — validation", "", "Positive values favor the left model; precision/recall differences are percentage points.", "",
             "| Comparison | Metric | Difference | 95% CI |", "|---|---|---:|---:|"]
    for label, metrics in evidence["paired_differences"].items():
        for metric, item in metrics.items():
            scale = 1 if metric == "AP" else 100
            delta.append(f"| {label} | {metric} | {scale*item['estimate']:+.6f} | [{scale*item['lower']:+.6f}, {scale*item['upper']:+.6f}] |")
    delta.extend(["", LIMITATION, ""])
    output.mkdir(parents=True, exist_ok=False)
    (output / "paired_differences.json").write_text(encoded, encoding="utf-8")
    (output / "comparison.md").write_text("\n".join(table), encoding="utf-8")
    (output / "paired_differences.md").write_text("\n".join(delta), encoding="utf-8")
    write_csv(output / "comparison.csv", ["model", "metric", "estimate", "ci_lower", "ci_upper", "threshold", "parameters", "cv_AP"], rows)
    _plots(runs, evidence, output)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    report = compare_runs(args.runs_root, args.output_dir)
    print(json.dumps({key: report[key] for key in ("training_targets", "validation_targets", "paired_differences")}, indent=2))
