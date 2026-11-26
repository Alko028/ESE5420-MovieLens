"""Build validation-only aggregate tables and scientific figures from saved runs.

This does not train a model, reselect a threshold, or read any test artifact.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.dataset import DEFAULT_OUTPUT, load_development
from common.evaluation import METRICS, RankedMetrics
from common.preprocessing import sha256
from compare_genre import PAIRS, _check_genre_metadata, _check_method_configuration
from compare_gnn import BOOTSTRAP_CONFIG, _interval, _read_csv, _reject_constant, load_run


BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
MODELS = ("movie_high_rating_rate", "logistic_regression", "knn", "gnn",
          "logistic_regression_genre", "knn_genre")
LABELS = {"movie_high_rating_rate": "Movie high-rating rate", "logistic_regression": "LR (history)",
          "knn": "kNN (history)", "gnn": "Shallow GNN", "logistic_regression_genre": "LR + genre",
          "knn_genre": "kNN + genre"}
INPUTS = ("train.npz", "validation.npz", "manifest.json")
CODE_URL = "https://github.com/Alko028/ESE5420-MovieLens"


def _dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _selection(result):
    name, config = result["model"], result["config"]
    if name == "movie_high_rating_rate":
        return "No fitted hyperparameter", "No hyperparameter grid", 1, "History movie aggregate score"
    if name.startswith("logistic_regression"):
        return (f'C={result["selected_C"]:g}', f'C_grid={config["C_grid"]}',
                66 if name.endswith("_genre") else 9, "History + genre table" if name.endswith("_genre") else "History table")
    selected = result["selected_parameters"]
    if name.startswith("knn"):
        return (f'k={selected["n_neighbors"]}; weights={selected["weights"]}',
                f'k_grid={config["n_neighbors_grid"]}; weights_grid={config["weights_grid"]}',
                66 if name.endswith("_genre") else 9, "History + genre table" if name.endswith("_genre") else "History table")
    return (f'K={selected["taps"]}; epochs={selected["epochs"]}; hidden={config["hidden_channels"]}',
            f'K_grid={config["taps_grid"]}; epochs_grid={config["epoch_grid"]}',
            2, "History-only graph input channels (not tabular features)")


def _validated_inputs():
    runs = {name: load_run(name, BASE / "outputs" / name) for name in MODELS}
    reference = runs["logistic_regression"]
    config = reference["result"]["config"]
    if config["confidence_level"] != .95:
        raise ValueError("This report is specified for 95% intervals")
    cv_reference = None
    for name, run in runs.items():
        if run["keys"] != reference["keys"]:
            raise ValueError(f"Unpaired validation rows: {name}")
        if any(run["result"]["input_sha256"][key] != reference["result"]["input_sha256"][key] for key in INPUTS):
            raise ValueError(f"Development input hashes differ: {name}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall"):
            if run["result"]["config"].get(key) != config[key]:
                raise ValueError(f"Unpaired evaluation configuration: {name}/{key}")
        for key in ("user_ids", "metric_names"):
            if not np.array_equal(run["bootstrap"][key], reference["bootstrap"][key]):
                raise ValueError(f"Unpaired bootstrap universe: {name}/{key}")
        if name != "movie_high_rating_rate":
            if any(run["result"]["config"].get(key) != config[key] for key in ("cv_folds", "cv_seed")):
                raise ValueError(f"Training CV configuration differs: {name}")
            fold_rows = _read_csv(run["root"] / "cv_assignments.csv", ("row_id", "fold"))
            folds = [(int(row["row_id"]), int(row["fold"])) for row in fold_rows]
            if (len({row[0] for row in folds}) != len(folds) or
                    any(row[0] < 0 or not 0 <= row[1] < config["cv_folds"] for row in folds) or
                    {row[1] for row in folds} != set(range(config["cv_folds"]))):
                raise ValueError(f"Invalid training CV assignments: {name}")
            if cv_reference is None:
                cv_reference = folds
            elif folds != cv_reference:
                raise ValueError(f"Training CV assignments differ: {name}")
    _check_method_configuration(runs)
    genre_features = _check_genre_metadata(runs)
    # Verify the comparison file is the current paired evidence, not a stale table.
    paired_path = BASE / "outputs/genre_comparison/paired_comparison.json"
    paired = json.loads(paired_path.read_text(encoding="utf-8"), parse_constant=_reject_constant)
    if (paired.get("evaluation_partition") != "validation" or paired.get("test_accessed") is not False or
            paired.get("input_sha256") != {key: reference["result"]["input_sha256"][key] for key in INPUTS} or
            paired.get("bootstrap_configuration") != {key: config[key] for key in BOOTSTRAP_CONFIG} or
            paired.get("genre_features") != genre_features or
            paired.get("validation_sample_count") != len(reference["keys"]) or
            paired.get("training_target_count") != len(cv_reference)):
        raise ValueError("Genre comparison provenance differs from saved model runs")
    for field in ("validation_rows_aligned", "training_cv_assignments_identical", "input_hashes_identical",
                  "within_method_configuration_unchanged", "genre_feature_provenance_identical"):
        if paired.get(field) is not True:
            raise ValueError(f"Genre comparison did not verify: {field}")
    for label, (left, right) in PAIRS.items():
        delta = runs[left]["bootstrap"]["samples"] - runs[right]["bootstrap"]["samples"]
        for i, metric in enumerate(METRICS):
            expected = {"estimate": float(runs[left]["result"]["metrics"][metric] - runs[right]["result"]["metrics"][metric]),
                        **_interval(delta[:, i], .95)}
            if paired["paired_differences"][label][metric] != expected:
                raise ValueError(f"Stale paired comparison: {label}/{metric}")
    train, validation = load_development()
    for key in INPUTS:
        if sha256(DEFAULT_OUTPUT / key) != reference["result"]["input_sha256"][key]:
            raise ValueError(f"Current development input differs from model runs: {key}")
    keys = list(zip(validation["row_id"].tolist(), validation["user_id"].tolist(),
                    validation["movie_id"].tolist(), validation["y"].tolist()))
    if keys != reference["keys"]:
        raise ValueError("Development labels do not align with saved validation predictions")
    return runs, paired, train, validation


def _format_interval(estimate, interval):
    lower, upper = interval["lower"], interval["upper"]
    bounds = "undefined" if lower is None or upper is None else f"[{lower:.4f}, {upper:.4f}]"
    return f"{estimate:.4f} {bounds}"


def _tables(runs, paired, output):
    rows = []
    markdown = ["# Validation model results", "", "All quantities are fractions. Brackets are paired-user-bootstrap 95% percentile intervals for each fixed fitted model and threshold.", "",
                "| Model | Precision [95% CI] | Recall [95% CI] | AP [95% CI] | Threshold | Inputs | Training CV selection |",
                "|---|---|---|---|---|---|---|"]
    for name, run in runs.items():
        result = run["result"]
        selected, grid, feature_count, representation = _selection(result)
        row = {"model": name, "threshold": result["threshold"], "cv_selected": selected, "cv_grid": grid,
               "n_features_or_channels": feature_count, "representation": representation, "partition": "validation"}
        cells = []
        for metric in METRICS:
            interval = result["intervals"][metric]
            row[metric] = result["metrics"][metric]
            for key in ("lower", "upper", "valid_replicates", "undefined_replicates"):
                row[f"{metric}_{key}"] = interval[key]
            cells.append(_format_interval(result["metrics"][metric], interval))
        rows.append(row)
        markdown.append(f"| {LABELS[name]} | {' | '.join(cells)} | {result['threshold']:.6f} | {feature_count}: {representation} | {selected} |")
    markdown.extend(["", "LR and kNN use 9 history features; genre versions use the same 9 plus 57 genre-derived features (66 total). The GNN uses two history graph-signal channels, a different representation rather than two tabular covariates. The frequency floor is one movie aggregate score.", "",
                     "Hyperparameters are chosen by mean AP in the same training-only five folds; learned scaling is fit inside each fold. The frequency floor has no fitted hyperparameter. Each operating threshold maximizes validation precision subject to recall >= 0.5; this is development evidence, not a final-test estimate.", "",
                     "## Hyperparameter grids", ""])
    markdown.extend(f"- **{LABELS[row['model']]}**: {row['cv_grid']}" for row in rows)
    markdown.extend(["", f"Code: {CODE_URL}", ""])
    _write_csv(output / "all_model_results.csv", list(rows[0]), rows)
    (output / "all_model_results.md").write_text("\n".join(markdown), encoding="utf-8")
    differences = []
    lines = ["# Paired validation differences", "", "Positive differences favor the first method. Values are fractions, not percentage points.", "",
             "| Comparison | Metric | Difference [95% CI] | Valid bootstrap replicates | Undefined replicates |", "|---|---|---|---|---|"]
    for label, metrics in paired["paired_differences"].items():
        for metric in METRICS:
            estimate = metrics[metric]
            differences.append({"comparison": label, "metric": metric, **estimate, "partition": "validation"})
            lines.append(f"| {label} | {metric} | {_format_interval(estimate['estimate'], estimate)} | {estimate['valid_replicates']} | {estimate['undefined_replicates']} |")
    lines.extend(["", "The two primary feature ablations are lr_genre_minus_lr and knn_genre_minus_knn. The additional cross-method comparisons are descriptive. Same sampled users are paired across models; intervals hold fitted models and selected validation thresholds fixed and exclude fitting/selection uncertainty.", ""])
    _write_csv(output / "paired_differences.csv", list(differences[0]), differences)
    (output / "paired_differences.md").write_text("\n".join(lines), encoding="utf-8")
    _dump(output / "paired_differences.json", paired)


def _figures(runs, paired, prevalence, output):
    colors = ["#777777", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axis = plt.subplots(figsize=(8, 5.3))
    for (name, run), color in zip(runs.items(), colors):
        rows = _read_csv(run["root"] / "validation_predictions.csv", ("y", "score"))
        y = np.asarray([int(row["y"]) for row in rows])
        scores = np.asarray([float(row["score"]) for row in rows])
        _, precision, recall = RankedMetrics(y, scores).curve()
        result = run["result"]
        axis.step(np.r_[0., recall], np.r_[1., precision], where="pre", color=color, linewidth=1.4,
                  label=f"{LABELS[name]} (AP={result['metrics']['AP']:.4f})")
        axis.scatter(result["metrics"]["recall"], result["metrics"]["precision"], color=color,
                     edgecolors="white", linewidths=.7, s=58, zorder=5)
    axis.axhline(prevalence, color="black", linestyle=":", linewidth=1.2,
                 label=f"Validation prevalence ({prevalence:.4f})")
    axis.set(xlabel="Recall (fraction)", ylabel="Precision (fraction)", xlim=(0, 1), ylim=(0, 1),
             title="Held-out validation precision-recall curves")
    axis.grid(alpha=.2)
    axis.legend(loc="lower left", fontsize=8.5)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"validation_pr_curves.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    primary = (("lr_genre_minus_lr", "LR + genre - LR"), ("knn_genre_minus_knn", "kNN + genre - kNN"))
    for axis, metric in zip(axes, ("precision", "AP")):
        axis.axvline(0., color="black", linestyle="--", linewidth=1)
        for y_position, (label, _) in enumerate(primary):
            interval = paired["paired_differences"][label][metric]
            if interval["lower"] is not None and interval["upper"] is not None:
                axis.hlines(y_position, interval["lower"], interval["upper"], linewidth=2, color="#0072B2")
                axis.vlines([interval["lower"], interval["upper"]], y_position - .06, y_position + .06,
                            color="#0072B2", linewidth=1.4)
            axis.scatter(interval["estimate"], y_position, s=45, color="#D55E00", zorder=3)
        axis.set(yticks=[0, 1], yticklabels=[item[1] for item in primary], ylim=(-.5, 1.5),
                 xlabel=f"Delta {metric} (fraction)", title=f"Paired genre effect: {metric}")
        axis.grid(axis="x", alpha=.2)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output / f"paired_genre_effects.{extension}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_report(output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite report: {output}")
    runs, paired, train, validation = _validated_inputs()
    prevalence = {"evaluation_partition": "development only", "test_accessed": False,
                  "method": "Analytic descriptive proportions and constant-score reference; no bootstrap interval computed."}
    for label, batch in (("train", train), ("validation", validation)):
        count, positive = len(batch["y"]), int(np.sum(batch["y"]))
        fraction = positive / count
        prevalence[label] = {"N": count, "positive_count": positive, "positive_prevalence": fraction,
                             "all_positive_precision": fraction, "all_positive_recall": 1.0,
                             "constant_score_AP": fraction}
    # Strict JSON validation before creating any artifacts.
    json.dumps(prevalence, allow_nan=False)
    output.mkdir(parents=True, exist_ok=False)
    _tables(runs, paired, output)
    _dump(output / "prevalence.json", prevalence)
    text = ["# High-rating prevalence and analytic reference", "", "Positive means a 4-5 star observed rating. Counts refer to eligible model targets, not the full raw dataset.", "",
            "| Partition | N | Positive count | High-rating fraction | All-positive precision | All-positive recall | Constant-score AP |",
            "|---|---|---|---|---|---|---|"]
    for name in ("train", "validation"):
        item = prevalence[name]
        text.append(f"| {name} | {item['N']} | {item['positive_count']} | {item['positive_prevalence']:.4f} | {item['all_positive_precision']:.4f} | 1.0000 | {item['constant_score_AP']:.4f} |")
    text.extend(["", "This reference always predicts high rating, hence recall 1 and precision equal to positive prevalence. Giving every target the same score also gives AP equal to prevalence. These are exact descriptive calculations on the observed development labels; no bootstrap interval was computed for this reference. It differs from the learned-threshold movie high-rating-rate floor. The final test was not read.", ""])
    (output / "prevalence.md").write_text("\n".join(text), encoding="utf-8")
    _figures(runs, paired, prevalence["validation"]["positive_prevalence"], output)
    (output / "captions.md").write_text(
        "# Figure captions\n\n"
        "**Figure 1 (validation_pr_curves).** Precision-recall curves on the same eligible held-out validation ratings for the movie high-rating-rate floor, nine-history-feature L2 logistic regression (LR) and kNN baselines, the shallow history-only GNN, and LR/kNN augmented with 57 genre-derived features (66 total). The GNN uses two graph-signal channels, so its representation differs from the tabular baselines. Legends give AP. Filled points show each model's actual validation-selected threshold maximizing precision subject to recall at least 0.5; the dotted horizontal line is validation positive prevalence. Both axes use fractions from 0 to 1. Thresholds were selected on these same validation labels, so operating-point performance may be optimistic and is not an independent final-test estimate.\n\n"
        "**Figure 2 (paired_genre_effects).** Genre-minus-history-only differences in validation precision (left) and AP (right) for LR and kNN. Points are observed differences and horizontal segments are 95% percentile intervals from the same paired user-bootstrap draws. Zero indicates no difference; positive values favor genre features. Values are fractions. Dataset, target rows, folds and within-method tuning grids are held fixed. Models retain their own fixed validation-selected thresholds; the bootstrap conditions on those fits and thresholds and excludes model-training and selection uncertainty. These are development comparisons, not final-test evidence.\n\n"
        f"Code: {CODE_URL}\n", encoding="utf-8")
    (output / "feedback_coverage.md").write_text(
        "# Early-draft feedback coverage from this experiment\n\n"
        "## Available evidence\n\n"
        "- LR + genre and kNN + genre use the same fixed targets and training-only CV folds as the original nine-feature baselines. Original grids are kept fixed to isolate feature augmentation.\n"
        "- all_model_results.csv/md supplies precision, actual recall and AP with user-bootstrap 95% intervals, thresholds, selected hyperparameters, grids and representations for all six methods.\n"
        "- paired_differences.csv/md/json and paired_genre_effects.png/pdf supply the primary within-method genre contrasts and paired uncertainty.\n"
        "- prevalence.json/md supplies training/validation high-rating fractions and the analytic all-positive/constant-score reference. It does not claim bootstrap uncertainty for that reference.\n"
        "- validation_pr_curves.png/pdf shows all six PR curves and actual selected operating points. captions.md records selection bias and interval limitations.\n\n"
        "## Remaining paper and experiment work\n\n"
        "- Interpret the actual genre results and GNN comparison; update abstract, methods, results and the question/title consistently.\n"
        "- Related Work, substantive Limitations, and moving the AI statement into Acknowledgments still require editing the paper.\n"
        "- kNN's original grid upper boundary remains unresolved. This run deliberately preserves that grid for an unconfounded genre comparison; any later expansion must be applied symmetrically and selected in training-only CV.\n"
        "- Freeze all models, hyperparameters, validation-chosen thresholds and analysis rules before evaluating sealed test exactly once. Final-test evaluation is still pending.\n"
        "- Final paper/video, solo-author contribution description, and solo-project reproducibility requirement remain in the project progress checklist.\n\n"
        "## Next task\n\n"
        "Review these aggregate figures and paired intervals, decide whether to symmetrically expand the kNN grid using training-only CV, and finalize the analysis protocol before opening the test. No test data was accessed to produce this report.\n\n"
        f"Code: {CODE_URL}\n", encoding="utf-8")
    return {"output_dir": str(output), "validation_samples": len(validation["y"]),
            "models": list(MODELS), "test_accessed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "report/genre_experiment")
    args = parser.parse_args()
    print(json.dumps(build_report(args.output_dir), indent=2, allow_nan=False))
