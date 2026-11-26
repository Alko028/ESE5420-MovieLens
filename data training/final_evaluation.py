"""Freeze a nine-method protocol by validation replay, then evaluate test once.

`freeze` never opens real test artifacts. `run-test` is a separate, explicit
command; it verifies the frozen provenance and consumes a protocol-specific
exclusive marker before loading test. No model/scaler/threshold is learned here.
"""

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, write_csv
from common.dataset import DEFAULT_OUTPUT, load_development, load_graph, load_final_test
from common.evaluation import METRICS, RankedMetrics, bootstrap_users
from common.genre_features import build_genre_features
from common.preprocessing import sha256
from compare_gnn import BOOTSTRAP_CONFIG, _interval, _read_csv, load_run

DATA_ROOT = DEFAULT_OUTPUT
DEFAULT_PROTOCOL = BASE / "protocols/final_v1"
MODEL_DEFINITIONS = (
    ("movie_rate", "Movie high-rating rate", "reference", "frequency", "movie_high_rating_rate", "movie_high_rating_rate"),
    ("lr", "L2 logistic regression", "baseline", "logistic", "logistic_regression", "logistic_regression"),
    ("knn", "kNN (511, distance)", "baseline", "knn", "knn_grid_extension/history", "knn_grid_extension"),
    ("gnn", "Original GNN (1 x 32)", "primary", "shallow_gnn", "gnn", "gnn"),
    ("lr_genre", "LR + genre", "genre", "logistic", "logistic_regression_genre", "logistic_regression_genre"),
    ("knn_genre", "kNN + genre (127, distance)", "genre", "knn", "knn_grid_extension/genre", "knn_grid_extension"),
    ("wide64", "GNN (1 x 64)", "exploratory", "extension", "gnn_extensions/wide64", "gnn_extensions"),
    ("wide128", "GNN (1 x 128)", "exploratory", "extension", "gnn_extensions/wide128", "gnn_extensions"),
    ("deep2x32", "GNN (2 x 32)", "exploratory", "extension", "gnn_extensions/deep2x32", "gnn_extensions"),
)
COMPARISONS = [
    {"id": "gnn_minus_lr", "left": "gnn", "right": "lr", "role": "primary"},
    {"id": "knn_minus_lr", "left": "knn", "right": "lr", "role": "secondary"},
    {"id": "lr_genre_minus_lr", "left": "lr_genre", "right": "lr", "role": "genre"},
    {"id": "knn_genre_minus_knn", "left": "knn_genre", "right": "knn", "role": "genre"},
    {"id": "movie_rate_minus_lr", "left": "movie_rate", "right": "lr", "role": "reference"},
    *[{"id": f"{name}_minus_{other}", "left": name, "right": other, "role": "exploratory"}
      for name in ("wide64", "wide128", "deep2x32") for other in ("gnn", "lr")],
]
INTERPRETATION = (
    "Primary endpoint: original GNN minus LR precision at fixed validation-chosen thresholds. "
    "An interval entirely above/below zero supports improvement/decrease; crossing zero is inconclusive, not equivalence. "
    "Test recall may be below 0.5; never adjust a threshold on test. Secondary/exploratory intervals are unadjusted "
    "for multiplicity. User-bootstrap intervals condition on fitted models/thresholds and omit training/split/selection "
    "uncertainty. No refit on validation, architecture choice, feature change or additional search follows test."
)


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write_json(path, value, *, exclusive=False):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as stream:
        stream.write(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def install_access_guard(root):
    root = Path(root).resolve()
    state = {"allow_test": False, "opened": set(), "test_open_events": 0, "blocked_test_events": 0}

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if root in path.parents:
                relative = path.relative_to(root).as_posix()
                if relative == "test_inputs.npz" or relative.startswith("sealed/"):
                    if not state["allow_test"]:
                        state["blocked_test_events"] += 1
                        raise PermissionError("Final-evaluation preparation blocked real test access")
                    state["test_open_events"] += 1
                state["opened"].add(relative)

    sys.addaudithook(guard)
    return state


def environment():
    import gnn
    return {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
            "sklearn": sklearn.__version__, "torch": str(gnn.torch.__version__),
            "cuda": gnn.torch.version.cuda}


def source_hashes():
    paths = [*BASE.glob("*.py"), *(BASE / "common").glob("*.py"),
             *(BASE / "configs").glob("*.json"), BASE / "tests/test_final_evaluation.py"]
    return {p.relative_to(BASE).as_posix(): sha256(p) for p in sorted(paths)}


def collect_models():
    models, runs = [], {}
    reference, reference_folds = None, None
    for model_id, label, role, kind, relative, saved_name in MODEL_DEFINITIONS:
        directory = BASE / "outputs" / relative
        run = load_run(saved_name, directory)
        result = run["result"]
        if reference is None:
            reference = run
        if run["keys"] != reference["keys"] or result["input_sha256"] != reference["result"]["input_sha256"]:
            raise ValueError(f"Unpaired development inputs/rows: {model_id}")
        for key in (*BOOTSTRAP_CONFIG, "minimum_validation_recall"):
            if result["config"][key] != reference["result"]["config"][key]:
                raise ValueError(f"Evaluation configuration mismatch: {model_id}/{key}")
        if kind != "frequency":
            folds = _read_csv(directory / "cv_assignments.csv", ("row_id", "fold"))
            if reference_folds is None:
                reference_folds = folds
            if folds != reference_folds or result["config"]["cv_folds"] != 5 or result["config"]["cv_seed"] != 42:
                raise ValueError(f"Training fold mismatch: {model_id}")
        filename = None if kind == "frequency" else "model.pt" if kind in ("shallow_gnn", "extension") else "model.npz"
        files = ["result.json", "validation_predictions.csv", "bootstrap.npz"]
        if filename:
            files += [filename, "cv_assignments.csv", "cv_summary.csv"]
        spec = {"id": model_id, "label": label, "role": role, "kind": kind,
                "run_directory": directory.relative_to(BASE).as_posix(), "saved_model_name": saved_name,
                "model_file": filename, "threshold": result["threshold"],
                "selected_parameters": result.get("selected_parameters", {"C": result["selected_C"]} if "selected_C" in result else {}),
                "genre": model_id in ("lr_genre", "knn_genre"),
                "files_sha256": {name: sha256(directory / name) for name in files}}
        if kind == "shallow_gnn":
            spec["hidden_channels"] = result["config"]["hidden_channels"]
        if kind == "extension":
            spec["variant"] = result["variant"]
        models.append(spec)
        runs[model_id] = run
    by_id = {s["id"]: s for s in models}
    if (by_id["knn"]["selected_parameters"] != {"n_neighbors": 511, "weights": "distance"} or
            by_id["knn_genre"]["selected_parameters"] != {"n_neighbors": 127, "weights": "distance"} or
            by_id["gnn"]["selected_parameters"] != {"taps": 5, "epochs": 40}):
        raise ValueError("Unexpected change to the agreed final model matrix")
    return models, runs


def predict_frozen(spec, batch, graph, device="cuda"):
    """Inference only. Remove labels before any feature or model operation."""
    inputs = {k: v for k, v in batch.items() if k != "y"}
    directory = BASE / spec["run_directory"]
    kind = spec["kind"]
    if kind == "frequency":
        scores = np.asarray(inputs["frequency_score"], dtype=np.float64)
    elif kind in ("logistic", "knn"):
        raw, names = build_genre_features(graph, inputs) if spec["genre"] else (inputs["X_raw"], inputs["feature_names"])
        raw = raw.astype(np.float64)
        with np.load(directory / spec["model_file"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["feature_names"], names)
            if (saved["mean"].shape != (raw.shape[1],) or not np.isfinite(saved["mean"]).all() or
                    not np.isfinite(saved["scale"]).all() or (saved["scale"] <= 0).any()):
                raise ValueError("Invalid frozen scaler")
            scaler = StandardScaler()
            scaler.mean_, scaler.scale_, scaler.n_features_in_ = saved["mean"], saved["scale"], raw.shape[1]
            transformed = scaler.transform(raw)
            if kind == "logistic":
                classifier = LogisticRegression()
                classifier.coef_, classifier.intercept_, classifier.classes_ = saved["coef"], saved["intercept"], saved["classes"]
                classifier.n_features_in_ = raw.shape[1]
                np.testing.assert_array_equal(classifier.classes_, [0, 1])
                if float(saved["C"]) != spec["selected_parameters"]["C"]:
                    raise ValueError("Frozen LR C differs from its checkpoint")
            else:
                selected = spec["selected_parameters"]
                if int(saved["n_neighbors"]) != selected["n_neighbors"] or str(saved["weights"]) != selected["weights"]:
                    raise ValueError("Frozen kNN setting differs from its checkpoint")
                if saved["X_fit"].ndim != 2 or saved["X_fit"].shape[1] != raw.shape[1] or saved["y_fit"].shape != (len(saved["X_fit"]),):
                    raise ValueError("Invalid frozen kNN training arrays")
                classifier = KNeighborsClassifier(n_neighbors=selected["n_neighbors"], weights=selected["weights"],
                                                  metric=str(saved["metric"]), algorithm="brute", n_jobs=1)
                # Rebuild the index from serialized TRAINING state; never fit a
                # scaler or append validation/test examples to these arrays.
                classifier.fit(saved["X_fit"], saved["y_fit"])
                np.testing.assert_array_equal(classifier.classes_, [0, 1])
            with threadpool_limits(limits=4):
                scores = classifier.predict_proba(transformed)[:, 1]
    else:
        import gnn
        from common.deep_movie_gnn import TwoLayerMovieGNN, build_full_history_taps
        torch = gnn.torch
        gnn.seed_training(20261125, 4)
        target_device = gnn.resolve_device(device)
        saved = torch.load(directory / spec["model_file"], map_location="cpu", weights_only=True)
        if saved["graph_sha256"] != sha256(DATA_ROOT / "graph.npz"):
            raise ValueError("GNN checkpoint graph differs from frozen history graph")
        variant = spec.get("variant")
        taps = variant["taps"] if variant else spec["selected_parameters"]["taps"]
        hidden = variant["hidden_channels"] if variant else spec["hidden_channels"]
        deep = variant is not None and variant["depth"] == 2
        model = TwoLayerMovieGNN(gnn.normalized_adjacency(graph), taps, hidden) if deep else gnn.ShallowMovieGNN(taps, hidden)
        model.load_state_dict(saved["state_dict"])
        model = model.to(target_device).eval()
        if not deep:
            features = gnn.polynomial_features(graph, inputs, taps)
            scores = gnn.predict_scores(model, features, target_device, batch_size=256 if variant else 8192)
        else:
            expected = TwoLayerMovieGNN(gnn.normalized_adjacency(graph), taps, hidden).adjacency.coalesce()
            actual = model.adjacency.cpu().coalesce()
            torch.testing.assert_close(actual.indices(), expected.indices(), rtol=0, atol=0)
            torch.testing.assert_close(actual.values(), expected.values(), rtol=0, atol=0)
            history = torch.as_tensor(build_full_history_taps(graph, taps), dtype=torch.float32, device=target_device)
            values = []
            with torch.no_grad():
                for start in range(0, len(inputs["row_id"]), 256):
                    users = torch.as_tensor(inputs["user_index"][start:start+256], dtype=torch.long, device=target_device)
                    movies = torch.as_tensor(inputs["movie_index"][start:start+256], dtype=torch.long, device=target_device)
                    values.append(torch.sigmoid(model(history, users, movies)).cpu().numpy())
            scores = np.concatenate(values).astype(np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (len(inputs["row_id"]),) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError(f"Invalid frozen scores: {spec['id']}")
    return scores


def evaluate_predictions(batch, scores_by_model, protocol):
    if set(scores_by_model) != {spec["id"] for spec in protocol["models"]}:
        raise ValueError("Predictions do not match the complete frozen model matrix")
    y, users = np.asarray(batch["y"]), np.asarray(batch["user_id"])
    if (y.ndim != 1 or users.shape != y.shape or len(np.unique(batch["row_id"])) != len(y) or
            not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid aligned evaluation rows/labels")
    results, bootstraps, universe = {}, {}, None
    config = protocol["evaluation"]
    for spec in protocol["models"]:
        name, threshold = spec["id"], spec["threshold"]
        scores = np.asarray(scores_by_model[name], dtype=np.float64)
        if not np.isfinite(threshold) or scores.shape != y.shape:
            raise ValueError("Invalid frozen threshold or score alignment")
        point = RankedMetrics(y, scores).evaluate(threshold)
        samples, intervals, user_ids = bootstrap_users(y, scores, users, threshold,
            config["bootstrap_replicates"], config["bootstrap_seed"], config["confidence_level"])
        if universe is not None:
            np.testing.assert_array_equal(universe, user_ids)
        universe = user_ids
        predicted = scores >= threshold
        results[name] = {"label": spec["label"], "role": spec["role"], "threshold": threshold,
                         "selected_parameters": spec["selected_parameters"], "metrics": dict(zip(METRICS, point)),
                         "intervals": intervals, "positive_predictions": int(predicted.sum()),
                         "user_coverage": len(np.unique(users[predicted])) / len(user_ids)}
        bootstraps[name] = samples
    differences = {}
    for pair in protocol["comparisons"]:
        left, right = pair["left"], pair["right"]
        delta = bootstraps[left] - bootstraps[right]
        differences[pair["id"]] = {"role": pair["role"], "left": left, "right": right,
            "metrics": {metric: {"estimate": results[left]["metrics"][metric] - results[right]["metrics"][metric],
                         **_interval(delta[:, i], config["confidence_level"])} for i, metric in enumerate(METRICS)}}
    return {"n_samples": len(y), "n_users": len(universe), "results": results, "paired_differences": differences,
            "prevalence": {"positive_count": int(y.sum()), "eligible_count": len(y), "positive_share": float(y.mean()),
                           "all_positive_precision": float(y.mean()), "all_positive_recall": 1.,
                           "constant_score_AP": float(y.mean()), "reference_type": "descriptive analytic reference"},
            "evaluation": config, "interpretation": protocol["interpretation"]}, bootstraps


def write_artifacts(output, batch, scores, protocol, summary, bootstraps, partition):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "results.json", {"partition": partition, **summary})
    write_json(output / "paired_differences.json", {"partition": partition, **summary["paired_differences"]})
    write_json(output / "prevalence.json", {"partition": partition, **summary["prevalence"]})
    names = [s["id"] for s in protocol["models"]]
    np.savez_compressed(output / "predictions.npz", model_names=np.asarray(names), scores=np.column_stack([scores[n] for n in names]),
                        row_id=batch["row_id"], user_id=batch["user_id"], movie_id=batch["movie_id"], y=batch["y"])
    rows = []
    for name in names:
        result = summary["results"][name]
        for metric in METRICS:
            interval = result["intervals"][metric]
            rows.append([name, metric, result["metrics"][metric], interval["lower"], interval["upper"], result["threshold"]])
        np.savez_compressed(output / f"{name}_bootstrap.npz", samples=bootstraps[name], metric_names=np.asarray(METRICS), user_ids=np.unique(batch["user_id"]))
        t, p, r = RankedMetrics(batch["y"], scores[name]).curve()
        write_csv(output / f"{name}_pr_curve.csv", ["threshold", "precision", "recall"], zip(t, p, r))
    write_csv(output / "comparison.csv", ["model", "metric", "estimate", "ci_lower", "ci_upper", "frozen_threshold"], rows)
    lines = [f"# Frozen-model comparison — {partition}", "", "Thresholds are fixed from validation; intervals are user-bootstrap percentile intervals.", "",
             "| Model | Precision [95% CI] | Recall [95% CI] | AP [95% CI] |", "|---|---:|---:|---:|"]
    for name in names:
        result = summary["results"][name]
        def cell(metric):
            value, interval = result["metrics"][metric], result["intervals"][metric]
            return f"{value:.6f} [{interval['lower']:.6f}, {interval['upper']:.6f}]" if np.isfinite(value) and interval["lower"] is not None else "undefined"
        lines.append(f"| {result['label']} | {cell('precision')} | {cell('recall')} | {cell('AP')} |")
    lines += ["", protocol["interpretation"], "", "The all-positive/constant-score reference is descriptive; see prevalence.json.", ""]
    (output / "comparison.md").write_text("\n".join(lines), encoding="utf-8")


def freeze(protocol_directory=DEFAULT_PROTOCOL, device="cuda"):
    root = Path(protocol_directory)
    if root.exists():
        raise FileExistsError(f"Refusing to replace an existing protocol: {root}")
    state = install_access_guard(DATA_ROOT)
    models, runs = collect_models()
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    protected = {s["id"]: dict(s["files_sha256"]) for s in models}
    inputs = {name: sha256(DATA_ROOT / name) for name in ("train.npz", "validation.npz", "graph.npz", "manifest.json")}
    code = source_hashes()
    shared = runs["lr"]["result"]["config"]
    evaluation = {key: shared[key] for key in BOOTSTRAP_CONFIG}
    evaluation.update(metric_names=list(METRICS), aggregation="pooled rating-level", bootstrap_unit="users",
                      interval="percentile", decision_rule="score >= frozen validation threshold", thresholds_reselected=False)
    protocol = {"schema_version": 1, "protocol_id": "final_v1", "client_date": "2026-11-25",
                "created_at_utc": datetime.now(timezone.utc).isoformat(), "models": models,
                "comparisons": COMPARISONS, "primary_comparison": "gnn_minus_lr", "primary_metric": "precision",
                "evaluation": evaluation, "interpretation": INTERPRETATION,
                "data_root": DATA_ROOT.relative_to(BASE).as_posix(), "development_sha256": inputs,
                "test_expected_sha256": {name: manifest["artifact_sha256"][name] for name in ("test_inputs.npz", "sealed/test_labels.npz")},
                "test_hash_source": "Existing manifest only; no test bytes opened during freeze",
                "source_sha256": code, "environment": environment(), "gnn_device": device, "thread_limit": 4,
                "refit": False, "test_evaluation_runs_allowed": 1, "multiple_comparison_adjustment": "none",
                "eligibility": "Fixed preprocessing: existing users, movies observed in history; identical test targets for all methods"}
    _, validation = load_development(DATA_ROOT)
    graph = load_graph(DATA_ROOT)
    scores, checks = {}, {}
    for spec in models:
        name = spec["id"]
        values = predict_frozen(spec, validation, graph, device)
        old = _read_csv(runs[name]["root"] / "validation_predictions.csv", ("score", "prediction"))
        old_values = np.asarray([float(row["score"]) for row in old])
        np.testing.assert_allclose(values, old_values, rtol=0, atol=1e-12)
        np.testing.assert_array_equal(values >= spec["threshold"], [int(row["prediction"]) for row in old])
        scores[name] = values
        checks[name] = {"maximum_score_error": float(np.max(np.abs(values-old_values))), "frozen_predictions_identical": True}
        print(f"Validation replay PASS: {name}; max score error={checks[name]['maximum_score_error']:.3g}", flush=True)
    summary, bootstraps = evaluate_predictions(validation, scores, protocol)
    for spec in models:
        name = spec["id"]
        np.testing.assert_allclose(bootstraps[name], runs[name]["bootstrap"]["samples"], rtol=0, atol=2e-12, equal_nan=True)
        for metric in METRICS:
            np.testing.assert_allclose(summary["results"][name]["metrics"][metric], runs[name]["result"]["metrics"][metric], rtol=0, atol=2e-12)
        for filename, digest in protected[name].items():
            if sha256(runs[name]["root"] / filename) != digest:
                raise ValueError("A selected model artifact changed during freeze")
    if code != source_hashes() or any(sha256(DATA_ROOT / name) != digest for name, digest in inputs.items()):
        raise ValueError("Code/development data changed during freeze")
    if state["test_open_events"] or state["blocked_test_events"]:
        raise AssertionError("Test access attempted during freeze")
    root.mkdir(parents=True, exist_ok=False)
    write_artifacts(root / "validation_replay", validation, scores, protocol, summary, bootstraps, "validation_replay")
    preflight = {"status": "PASS", "model_checks": checks, "all_bootstrap_arrays_reproduced": True,
                 "selected_model_artifacts_unchanged": True, "source_and_development_hashes_unchanged": True,
                 "test_accessed": False, "preprocessed_files_opened": sorted(state["opened"])}
    write_json(root / "preflight.json", preflight, exclusive=True)
    protocol.update(status="frozen", preflight_sha256=sha256(root / "preflight.json"))
    write_json(root / "protocol.json", protocol, exclusive=True)
    (root / "protocol.sha256").write_text(sha256(root / "protocol.json") + "\n", encoding="ascii")
    return protocol


def verify_frozen(root):
    root = Path(root)
    if sha256(root / "protocol.json") != (root / "protocol.sha256").read_text(encoding="ascii").strip():
        raise ValueError("Frozen protocol hash changed")
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("schema_version") != 1 or protocol.get("refit") is not False:
        raise ValueError("A fully frozen no-refit protocol is required")
    names = [s["id"] for s in protocol["models"]]
    if (names != [s[0] for s in MODEL_DEFINITIONS] or protocol["comparisons"] != COMPARISONS or
            protocol.get("primary_comparison") != "gnn_minus_lr" or protocol.get("primary_metric") != "precision" or
            protocol.get("test_evaluation_runs_allowed") != 1 or protocol["evaluation"].get("thresholds_reselected") is not False):
        raise ValueError("Frozen model matrix/analysis rules differ")
    if sha256(root / "preflight.json") != protocol["preflight_sha256"]:
        raise ValueError("Preflight evidence changed")
    preflight = json.loads((root / "preflight.json").read_text(encoding="utf-8"))
    if (preflight.get("status") != "PASS" or preflight.get("test_accessed") is not False or
            preflight.get("all_bootstrap_arrays_reproduced") is not True or
            set(preflight.get("model_checks", {})) != set(names) or
            any(item.get("frozen_predictions_identical") is not True for item in preflight["model_checks"].values())):
        raise ValueError("Validation preflight did not pass")
    if protocol["environment"] != environment():
        raise ValueError("Evaluation environment differs from the validated frozen environment")
    for relative, digest in protocol["source_sha256"].items():
        if sha256(BASE / relative) != digest:
            raise ValueError(f"Frozen source/config changed: {relative}")
    if (BASE / protocol["data_root"]).resolve() != DATA_ROOT.resolve():
        raise ValueError("Frozen data root differs")
    for relative, digest in protocol["development_sha256"].items():
        if sha256(DATA_ROOT / relative) != digest:
            raise ValueError(f"Frozen development/graph input changed: {relative}")
    for spec in protocol["models"]:
        for relative, digest in spec["files_sha256"].items():
            if sha256(BASE / spec["run_directory"] / relative) != digest:
                raise ValueError(f"Frozen model artifact changed: {spec['id']}/{relative}")
    manifest = json.loads((DATA_ROOT / "manifest.json").read_text(encoding="utf-8"))
    if any(manifest["artifact_sha256"][name] != digest for name, digest in protocol["test_expected_sha256"].items()):
        raise ValueError("Test artifact manifest changed")
    return protocol


def claim_test_run(root, protocol_hash):
    """This marker remains even on failure. Another output directory cannot bypass it."""
    path = Path(root) / "test_access_started.json"
    write_json(path, {"protocol_sha256": protocol_hash, "started_at_utc": datetime.now(timezone.utc).isoformat(),
                     "warning": "Consumed evaluation allowance; do not delete to rerun after observing test."}, exclusive=True)
    return path


def run_test(protocol_directory=DEFAULT_PROTOCOL, output_dir=None):
    root = Path(protocol_directory)
    output = Path(output_dir) if output_dir else BASE / "outputs/final_test_v1"
    if output.exists() or (root / "test_access_started.json").exists():
        raise FileExistsError("Final output or protocol-specific one-run marker already exists; refusing another test evaluation")
    state = install_access_guard(DATA_ROOT)
    protocol = verify_frozen(root)
    graph = load_graph(DATA_ROOT)
    # All provenance checks above are test-free. Claim before loading any labels.
    claim_test_run(root, sha256(root / "protocol.json"))
    state["allow_test"] = True
    try:
        batch = load_final_test(DATA_ROOT, protocol_frozen=True)
        if graph["history_mask"][batch["user_index"], batch["movie_index"]].any():
            raise ValueError("Test target overlaps observed history")
        scores = {spec["id"]: predict_frozen(spec, batch, graph, protocol["gnn_device"]) for spec in protocol["models"]}
        summary, bootstraps = evaluate_predictions(batch, scores, protocol)
        write_artifacts(output, batch, scores, protocol, summary, bootstraps, "test")
        write_json(output / "execution.json", {"status": "PASS", "protocol_sha256": sha256(root / "protocol.json"),
                   "final_loader_calls": 1, "test_open_events": state["test_open_events"],
                   "note": "File hash verification and np.load each open an artifact; there is one evaluation/loader invocation.",
                   "preprocessed_files_opened": sorted(state["opened"]), "thresholds_reselected": False, "refit": False})
    except BaseException as exc:
        write_json(root / "test_failure.json", {"error": repr(exc), "test_open_events": state["test_open_events"],
                    "action": "Stop and inspect; allowance remains consumed. Do not blindly reopen test or remove marker."})
        raise
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "verify", "run-test"))
    parser.add_argument("--protocol-dir", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda", help="Freeze only; stored device is used by run-test")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze(args.protocol_dir, args.device)
        print(json.dumps({"status": result["status"], "models": [s["id"] for s in result["models"]], "test_accessed": False}))
    elif args.action == "verify":
        result = verify_frozen(args.protocol_dir)
        print(json.dumps({"status": "PASS", "primary_comparison": result["primary_comparison"], "test_accessed": False}))
    else:
        result = run_test(args.protocol_dir, args.output_dir)
        print(json.dumps(clean({"n_samples": result["n_samples"], "results": result["results"]}), indent=2, allow_nan=False))
