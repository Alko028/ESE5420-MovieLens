"""Audit new fixed-input development runs and compare archived reproduction.

This never loads real test inputs/labels. Old/new results are compared for
reproducibility, never selected by which run scores better.
"""
import csv
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, precision_score, recall_score

from common.dataset import load_development, load_graph
from common.preprocessing import sha256
from gnn_extensions import install_test_guard

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
ARCHIVE = PROJECT / "archive/full_rerun_2026-11-25_01"
OLD = ARCHIVE / "data training/outputs"
NEW = BASE / "outputs"
RUNS = ["movie_high_rating_rate", "logistic_regression", "knn", "gnn", "logistic_regression_genre", "knn_genre",
        "gnn_extensions/wide64", "gnn_extensions/wide128", "gnn_extensions/deep2x32",
        "knn_grid_extension/history", "knn_grid_extension/genre"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def tensors_equal(a, b):
    assert type(a) is type(b)
    if isinstance(a, torch.Tensor):
        if a.layout == torch.sparse_coo:
            a, b = a.coalesce(), b.coalesce()
            assert a.shape == b.shape and torch.equal(a.indices(), b.indices()) and torch.equal(a.values(), b.values())
        else:
            assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            tensors_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            tensors_equal(left, right)
    else:
        assert a == b


def compare_payload(left, right, filename, ignore_legacy_knn_score=False):
    if filename.endswith(".npz"):
        with np.load(left / filename, allow_pickle=False) as a, np.load(right / filename, allow_pickle=False) as b:
            assert a.files == b.files
            for key in a.files:
                if np.issubdtype(a[key].dtype, np.floating):
                    np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-12, equal_nan=True)
                else:
                    np.testing.assert_array_equal(a[key], b[key])
    elif filename.endswith(".pt"):
        tensors_equal(torch.load(left / filename, map_location="cpu", weights_only=True),
                      torch.load(right / filename, map_location="cpu", weights_only=True))
    else:
        with (left / filename).open(newline="", encoding="utf-8") as stream:
            a = list(csv.reader(stream))
        with (right / filename).open(newline="", encoding="utf-8") as stream:
            b = list(csv.reader(stream))
        assert a[0] == b[0] and len(a) == len(b)
        for i, field in enumerate(a[0]):
            if ignore_legacy_knn_score and field == "score":
                continue
            av, bv = [r[i] for r in a[1:]], [r[i] for r in b[1:]]
            try:
                numeric = np.asarray(av, dtype=float), np.asarray(bv, dtype=float)
            except ValueError:
                assert av == bv, (filename, field)
            else:
                np.testing.assert_allclose(*numeric, rtol=0, atol=1e-12, equal_nan=True)


def main():
    opened = install_test_guard()
    state = read_json(NEW / "full_rerun_20261125/status.json")
    assert state["status"] == "PASS" and len(state["stages"]) == 15
    access = {}
    for path in (NEW / "full_rerun_20261125").glob("*_access.json"):
        row = read_json(path)
        assert not row["blocked"] and row["test_accessed"] is False
        access[path.stem] = row
    assert len(access) == 15
    protected = read_json(ARCHIVE / "protected_before.json")
    for path, digest in protected.items():
        assert sha256(PROJECT / path).lower() == digest.lower(), path
    train, validation = load_development()
    graph = load_graph()
    assert not np.intersect1d(train["row_id"], validation["row_id"]).size
    for batch in (train, validation):
        assert not graph["history_mask"][batch["user_index"], batch["movie_index"]].any()
    reference = (NEW / "logistic_regression/cv_assignments.csv").read_bytes()
    evidence = {}
    files = ("validation_predictions.csv", "bootstrap.npz", "cv_assignments.csv", "cv_folds.csv", "cv_summary.csv", "model.npz", "model.pt", "training_history.csv")
    for name in RUNS:
        root, old_root = NEW / name, OLD / name
        result, old = read_json(root / "result.json"), read_json(old_root / "result.json")
        for key in ("config", "input_sha256", "selected_parameters"):
            assert result.get(key) == old.get(key), (name, key)
        # Earlier baseline runs predated the added genre/deep common modules.
        # Require every old recorded module to match; verify all new hashes too.
        for relative, digest in old["code_sha256"].items():
            assert result["code_sha256"].get(relative) == digest, (name, relative)
        for relative, digest in result["code_sha256"].items():
            assert sha256(BASE / relative) == digest, (name, relative)
        variation = name == "gnn_extensions/deep2x32"
        numerical_changes = {key: {"old": old.get(key), "new": result.get(key)}
                             for key in ("metrics", "intervals", "threshold", "cv_mean_AP")
                             if result.get(key) != old.get(key)}
        if not variation:
            for key in ("threshold", "cv_mean_AP"):
                if key in result:
                    np.testing.assert_allclose(result[key], old[key], rtol=0, atol=1e-12)
            for key in ("metrics", "intervals"):
                assert result[key] == old[key], (name, key)
        if (root / "cv_assignments.csv").exists():
            assert (root / "cv_assignments.csv").read_bytes() == reference, name
        rows = np.genfromtxt(root / "validation_predictions.csv", delimiter=",", names=True)
        old_rows = np.genfromtxt(old_root / "validation_predictions.csv", delimiter=",", names=True)
        maximum_score_error = float(np.max(np.abs(rows["score"] - old_rows["score"])))
        if not variation:
            np.testing.assert_array_equal(rows["prediction"], old_rows["prediction"])
            if name != "knn":
                assert maximum_score_error <= 1e-12
        for key in ("row_id", "user_id", "movie_id", "y"):
            np.testing.assert_array_equal(rows[key], validation[key])
        pred = rows["score"] >= result["threshold"]
        np.testing.assert_array_equal(pred, rows["prediction"])
        metrics = [precision_score(rows["y"], pred), recall_score(rows["y"], pred), average_precision_score(rows["y"], rows["score"])]
        np.testing.assert_allclose(metrics, [result["metrics"][m] for m in ("precision", "recall", "AP")], rtol=0, atol=2e-12)
        checked = []
        different = []
        for filename in files:
            if (root / filename).exists():
                try:
                    compare_payload(root, old_root, filename, name == "knn" and filename == "validation_predictions.csv")
                except AssertionError:
                    if not variation or filename == "cv_assignments.csv":
                        raise
                    different.append(filename)
                checked.append(filename)
        evidence[name] = {"old_new_numerically_identical": not different and not numerical_changes,
                          "files_checked": checked, "different_numeric_payloads": different,
                          "numerical_changes": numerical_changes, "same_validation_rows_labels": True,
                          "maximum_old_new_score_error": maximum_score_error,
                          "roundoff_comparison_atol": 1e-12,
                          "legacy_127_knn_scores_not_required_bitwise_equal": name == "knn",
                          "newly_hash_recorded_modules": sorted(result["code_sha256"].keys() - old["code_sha256"].keys()),
                          "same_training_code_config_inputs_and_selected_parameters": True}
    for filename in files:
        if (NEW / "gnn" / filename).exists():
            compare_payload(NEW / "gnn", NEW / "gnn_repeat_verified", filename)
    sanity = read_json(NEW / "genre_sanity_verified/sanity.json")
    assert sanity["status"] == "PASS"
    for relative in ("gnn_extensions/preservation.json", "knn_grid_extension/preservation.json"):
        assert read_json(NEW / relative)["status"] == "PASS"
    probe = read_json(NEW / "full_rerun_20261125/sparse_repeat_probe.json")
    assert probe["deterministic_algorithms_enabled"] and all(item["equal"] for item in probe["forward"])
    assert any(not item["equal"] for item in probe["backward"]) and probe["test_accessed"] is False
    output = {"status": "PASS_WITH_DOCUMENTED_NUMERICAL_VARIATION", "formal_development_runs_checked": len(RUNS), "runs": evidence,
              "all_cv_assignments_identical": True, "all_current_samples_labels_aligned": True,
              "gnn_full_repeat_identical": True, "preserved_raw_data_inputs_code_files": len(protected),
              "development_process_access": access, "audit_preprocessed_files_opened": sorted(opened),
              "test_accessed": False, "test_previously_observed_in_archived_final_run": True,
              "selection_of_better_old_or_new_run": False}
    (NEW / "full_rerun_20261125/verification.json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in output.items() if key not in ("runs", "development_process_access")}, indent=2))


if __name__ == "__main__":
    main()
