"""Compare three prespecified exploratory GNN extensions using saved validation runs.

No training or data/test loading occurs here. Each model keeps its saved
validation-chosen threshold. These exploratory, unadjusted comparisons condition
on fitted models and cannot select a variant using validation as fresh evidence.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from common.evaluation import METRICS, RankedMetrics
from compare_gnn import BOOTSTRAP_CONFIG, _integer, _interval, _read_csv, load_run


BASE = Path(__file__).resolve().parent
VARIANTS = ("wide64", "wide128", "deep2x32")
COUNTS = {"wide64": 769, "wide128": 1537, "deep2x32": 3361}
INPUTS = ("train.npz", "validation.npz", "manifest.json")
GNN_NAMES = ("gnn", *VARIANTS)
TABLE_NAMES = ("logistic_regression", *GNN_NAMES)
LABELS = {"gnn": "Original GNN", "wide64": "Wide GNN (64)",
          "wide128": "Wide GNN (128)", "deep2x32": "Deep GNN (2 x 32)",
          "logistic_regression": "Logistic regression"}
LIMITATION = (
    "Exploratory extension comparisons after inspecting the original validation result; "
    "Paired user-bootstrap percentile intervals are unadjusted for multiple comparisons. "
    "Validation already selects each operating threshold. Intervals condition on fitted "
    "models/thresholds and exclude fitting, hyperparameter and architecture-selection "
    "uncertainty. Do not choose a variant from these validation scores and describe it as "
    "independently confirmed. The sealed test remains untouched."
)


def _cv_rows(run, folds):
    rows = _read_csv(run["root"] / "cv_assignments.csv", ("row_id", "fold"))
    try:
        pairs = [(int(row["row_id"]), int(row["fold"])) for row in rows]
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid training CV assignments") from exc
    if (len({row[0] for row in pairs}) != len(pairs) or
            any(row[0] < 0 or not 0 <= row[1] < folds for row in pairs) or
            {row[1] for row in pairs} != set(range(folds))):
        raise ValueError("Invalid training CV assignments")
    return pairs


def _original_parameters(result):
    """Recover the legacy count from its saved one-layer architecture, not weights."""
    channels = result["config"].get("hidden_channels")
    taps = result["selected_parameters"].get("taps")
    if channels != 32 or taps != 5 or result.get("input_channels") != ["history_rating/5", "history_mask"]:
        raise ValueError("Original GNN architecture differs from the fixed shallow reference")
    count = 2 * taps * channels + channels + channels + 1
    if result.get("parameter_count", count) != count:
        raise ValueError("Original GNN parameter count differs")
    return count


def _validate_metadata(runs):
    reference = runs["gnn"]["result"]
    graph = reference.get("graph_sha256")
    if (not isinstance(graph, str) or len(graph) != 64 or
            any(character not in "0123456789abcdef" for character in graph)):
        raise ValueError("Missing original GNN graph hash")
    counts = {"gnn": _original_parameters(reference), "logistic_regression": None}
    for name in GNN_NAMES:
        result = runs[name]["result"]
        for key in ("all_training_losses_finite", "model_restore_verified"):
            if result.get(key) is not True:
                raise ValueError(f"GNN training check failed: {name}/{key}")
        if result.get("graph_sha256") != graph:
            raise ValueError(f"History graph hashes differ: {name}")
        if result.get("input_channels") != reference["input_channels"] or result.get("genre_used") is not False:
            raise ValueError(f"GNN input representation differs: {name}")
        for field in ("learning_rate", "weight_decay", "batch_size", "gradient_clip_norm",
                      "training_seed", "thread_limit", "epoch_grid"):
            if result["config"].get(field) != reference["config"].get(field):
                raise ValueError(f"GNN training configuration differs: {name}/{field}")
        if name != "gnn":
            if result.get("variant_name") != name:
                raise ValueError(f"Extension variant name differs: {name}")
            if result.get("parameter_count") != COUNTS[name]:
                raise ValueError(f"Extension parameter count differs: {name}")
            if result.get("max_hops") != 4:
                raise ValueError(f"Extension maximum propagation depth differs: {name}")
            counts[name] = COUNTS[name]
        epochs = result.get("selected_parameters", {}).get("epochs")
        if epochs not in (20, 40) or type(epochs) is not int:
            raise ValueError(f"Unexpected selected epoch checkpoint: {name}")
        for field in ("fit_and_cv_seconds", "evaluation_seconds", "cv_mean_AP"):
            value = result.get(field)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Missing or invalid {field}: {name}")
    return counts, graph


def _rows(runs, counts):
    for name in TABLE_NAMES:
        result = runs[name]["result"]
        for metric in METRICS:
            interval = result["intervals"][metric]
            yield {"model": name, "metric": metric, "estimate": result["metrics"][metric],
                   "ci_lower": interval["lower"], "ci_upper": interval["upper"],
                   "threshold": result["threshold"], "parameter_count": counts[name],
                   "cv_mean_AP": result.get("cv_mean_AP"),
                   "selected_epochs": result.get("selected_parameters", {}).get("epochs"),
                   "fit_and_cv_seconds": result.get("fit_and_cv_seconds"),
                   "evaluation_seconds": result.get("evaluation_seconds"), "partition": "validation"}


def _metric_text(result, metric):
    interval = result["intervals"][metric]
    def format_value(value):
        if value is None:
            return "undefined"
        return f"{100 * value:.2f}%" if metric != "AP" else f"{value:.4f}"
    return f"{format_value(result['metrics'][metric])} [{format_value(interval['lower'])}, {format_value(interval['upper'])}]"


def _comparison_markdown(runs, counts, confidence):
    lines = ["# Exploratory GNN capacity extensions — validation", "",
             f"Intervals are {100 * confidence:g}% user-bootstrap percentile intervals. "
             "Precision and recall refer to each saved threshold selected for validation recall >= 0.5.", "",
             "| Model | Precision [CI] | Recall [CI] | AP [CI] | Threshold | Parameters | CV AP | Epochs | Fit + CV (s) | Evaluation (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name in TABLE_NAMES:
        result = runs[name]["result"]
        epochs = result.get("selected_parameters", {}).get("epochs", "—")
        parameters = counts[name] if counts[name] is not None else "—"
        lines.append(f"| {LABELS[name]} | {_metric_text(result, 'precision')} | {_metric_text(result, 'recall')} | "
                     f"{_metric_text(result, 'AP')} | {result['threshold']:.6f} | {parameters} | "
                     f"{result['cv_mean_AP']:.4f} | {epochs} | {result['fit_and_cv_seconds']:.3f} | "
                     f"{result['evaluation_seconds']:.3f} |")
    lines.extend(["", "Original GNN parameter count (385) is recovered from its saved 2-input, "
                  "5-tap, 32-hidden-channel architecture. All four GNNs have at most four graph hops. "
                  "The original GNN searched 3/5 taps and 20/40 epochs; extension architectures fix taps "
                  "and tune only 20/40 epochs on the same training folds. "
                  "Timing is measured on the saved runs; it is hardware/load dependent and not a controlled runtime benchmark.",
                  "", LIMITATION, ""])
    return "\n".join(lines)


def _differences_markdown(evidence):
    confidence = evidence["bootstrap_configuration"]["confidence_level"]
    lines = ["# Paired exploratory differences — validation", "",
             f"Positive differences favor the extension. Intervals: {100 * confidence:g}% paired user bootstrap.", "",
             "| Comparison | Metric | Difference | Interval |", "|---|---|---:|---:|"]
    for label, metrics in evidence["paired_differences"].items():
        for metric, value in metrics.items():
            scale, suffix, digits = (1, "", 4) if metric == "AP" else (100, " pp", 2)
            point = f"{scale * value['estimate']:+.{digits}f}{suffix}"
            bounds = ["undefined" if value[key] is None else f"{scale * value[key]:+.{digits}f}" for key in ("lower", "upper")]
            lines.append(f"| {label} | {metric} | {point} | [{bounds[0]}, {bounds[1]}]{suffix} |")
    lines.extend(["", LIMITATION, ""])
    return "\n".join(lines)


def _plot(runs, evidence, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axis = plt.subplots(figsize=(7.5, 5.1))
    colors = plt.get_cmap("tab10").colors
    for index, name in enumerate(GNN_NAMES):
        run = runs[name]
        rows = _read_csv(run["root"] / "validation_predictions.csv", ("score",))
        scores = np.asarray([float(row["score"]) for row in rows])
        labels = np.asarray([row[3] for row in run["keys"]])
        _, precision, recall = RankedMetrics(labels, scores).curve()
        result = run["result"]
        # Recompute curves from the validated predictions, rather than trusting an unrelated CSV.
        axis.step(recall, precision, where="pre", color=colors[index],
                  label=f"{LABELS[name]} (AP {result['metrics']['AP']:.4f})")
        axis.scatter(result["metrics"]["recall"], result["metrics"]["precision"],
                     color=colors[index], edgecolor="black", linewidth=.5, zorder=4, s=50)
    axis.axvline(.5, linestyle=":", color="gray", linewidth=1)
    axis.set(xlabel="Recall (fraction of high ratings retrieved)",
             ylabel="Precision (fraction of predicted positives rated 4–5)", xlim=(0, 1), ylim=(0, 1.02),
             title="Validation PR curves — exploratory GNN extensions")
    axis.legend(loc="lower left", fontsize=8)
    axis.grid(alpha=.2)
    fig.text(.12, .015, "Dots: saved validation-chosen operating points; test not accessed.", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_pr_curves.{extension}", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.3, 4.1))
    for index, name in enumerate(VARIANTS):
        value = evidence["paired_differences"][name + "_minus_gnn"]["precision"]
        estimate = 100 * value["estimate"]
        axis.scatter(estimate, index, color=colors[index + 1], zorder=3, s=50)
        if value["lower"] is not None and value["upper"] is not None:
            # Draw interval directly: a percentile interval need not contain the point estimate.
            axis.hlines(index, 100 * value["lower"], 100 * value["upper"], color=colors[index + 1], linewidth=2)
            axis.plot([100 * value["lower"], 100 * value["upper"]], [index, index], "|", color=colors[index + 1])
    axis.axvline(0, color="black", linestyle="--", linewidth=1)
    axis.set_yticks(range(len(VARIANTS)), [LABELS[name] for name in VARIANTS])
    axis.invert_yaxis()
    axis.set(xlabel="Precision difference vs original GNN (percentage points)",
             title="Paired user-bootstrap differences — validation")
    axis.grid(axis="x", alpha=.2)
    confidence = 100 * evidence["bootstrap_configuration"]["confidence_level"]
    fig.text(.09, .015, f"Unadjusted exploratory {confidence:g}% intervals; thresholds fixed after validation selection.", fontsize=8)
    fig.tight_layout(rect=(0, .035, 1, 1))
    for extension in ("png", "pdf"):
        fig.savefig(output / f"paired_precision_differences.{extension}", dpi=180)
    plt.close(fig)


def compare_runs(runs_root, output_dir=None, original_dir=None, lr_dir=None):
    """Check every pairing before creating new output; never overwrite existing artifacts."""
    runs_root = Path(runs_root)
    output = Path(output_dir) if output_dir is not None else runs_root / "comparison"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite comparison: {output}")
    original_dir = Path(original_dir) if original_dir is not None else BASE / "outputs/gnn"
    lr_dir = Path(lr_dir) if lr_dir is not None else BASE / "outputs/logistic_regression"
    for protected in (original_dir, lr_dir, *(runs_root / name for name in VARIANTS)):
        if output.resolve() == protected.resolve() or protected.resolve() in output.resolve().parents:
            raise ValueError("Comparison must not be written inside a model run")
    preservation = None
    preservation_file = runs_root / "preservation.json"
    if preservation_file.exists():
        preservation = json.loads(preservation_file.read_text(encoding="utf-8"))
        if preservation.get("status") != "PASS" or preservation.get("original_files_preserved") is not True:
            raise ValueError("Original artifact preservation check did not pass")
        if preservation.get("test_accessed", False) is not False:
            raise ValueError("Preservation report indicates test access")
        if "before" in preservation or "after" in preservation:
            if (not isinstance(preservation.get("before"), dict) or
                    preservation.get("before") != preservation.get("after") or
                    preservation.get("protected_file_count") != len(preservation["before"])):
                raise ValueError("Original artifact preservation hashes disagree")
    runs = {"gnn": load_run("gnn", original_dir),
            "logistic_regression": load_run("logistic_regression", lr_dir)}
    runs.update({name: load_run("gnn_extensions", runs_root / name) for name in VARIANTS})
    reference = runs["gnn"]
    config = reference["result"]["config"]
    _integer(config["cv_folds"], "CV folds", 2)
    _integer(config["cv_seed"], "CV seed")
    expected_hashes = {key: reference["result"]["input_sha256"][key] for key in INPUTS}
    reference_folds = _cv_rows(reference, config["cv_folds"])
    for name, run in runs.items():
        result = run["result"]
        if any(result["input_sha256"][key] != expected_hashes[key] for key in INPUTS):
            raise ValueError(f"Development input hashes differ: {name}")
        if run["keys"] != reference["keys"]:
            raise ValueError(f"Unpaired validation rows or row ordering: {name}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall", "cv_folds", "cv_seed"):
            if result["config"].get(key) != config[key]:
                raise ValueError(f"Unpaired evaluation or CV configuration: {name}/{key}")
        for key in ("metric_names", "user_ids"):
            if not np.array_equal(run["bootstrap"][key], reference["bootstrap"][key]):
                raise ValueError(f"Unpaired bootstrap {key}: {name}")
        if _cv_rows(run, config["cv_folds"]) != reference_folds:
            raise ValueError(f"Training CV assignments differ: {name}")
    counts, graph_hash = _validate_metadata(runs)
    differences = {}
    for name in VARIANTS:
        left = runs[name]
        for other in ("gnn", "logistic_regression"):
            right = runs[other]
            delta = left["bootstrap"]["samples"] - right["bootstrap"]["samples"]
            differences[name + "_minus_" + other] = {
                metric: {"estimate": float(left["result"]["metrics"][metric] - right["result"]["metrics"][metric]),
                         **_interval(delta[:, index], config["confidence_level"])}
                for index, metric in enumerate(METRICS)}
    evidence = {"evaluation_partition": "validation", "test_accessed": False,
                "variants": list(VARIANTS), "paired_differences": differences,
                "validation_rows_aligned": True, "validation_sample_count": len(reference["keys"]),
                "training_target_count": len(reference_folds), "training_cv_assignments_identical": True,
                "input_hashes_identical": True, "input_sha256": expected_hashes,
                "graph_sha256": graph_hash, "parameter_counts": counts, "max_hops_all_gnns": 4,
                "bootstrap_configuration": {key: config[key] for key in BOOTSTRAP_CONFIG},
                "interval_method": "paired user-bootstrap percentile; each model's fixed validation-chosen threshold",
                "multiple_comparisons_adjusted": False, "exploratory": True,
                "validation_must_not_select_variant_as_independent_evidence": True,
                "original_artifacts_preserved": preservation.get("status") == "PASS" if preservation is not None else None,
                "preservation_evidence": preservation, "limitation": LIMITATION}
    # Force strict serialization before any write; malformed saved runs leave no output.
    encoded = json.dumps(evidence, indent=2, allow_nan=False) + "\n"
    comparison_text = _comparison_markdown(runs, counts, config["confidence_level"])
    differences_text = _differences_markdown(evidence)
    output.mkdir(parents=True, exist_ok=False)
    rows = list(_rows(runs, counts))
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.md").write_text(comparison_text, encoding="utf-8")
    (output / "paired_differences.json").write_text(encoded, encoding="utf-8")
    (output / "paired_differences.md").write_text(differences_text, encoding="utf-8")
    _plot(runs, evidence, output)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=BASE / "outputs/gnn_extensions")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--gnn-dir", type=Path, default=BASE / "outputs/gnn")
    parser.add_argument("--lr-dir", type=Path, default=BASE / "outputs/logistic_regression")
    args = parser.parse_args()
    print(json.dumps(compare_runs(args.runs_root, args.output_dir, args.gnn_dir, args.lr_dir), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
