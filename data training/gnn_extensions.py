"""Three prespecified exploratory GNN capacity variants, isolated from old runs.

The original GNN/inputs/results are hashed before/after this run. Each variant
selects only its epoch count using the same training-only folds, then reports
validation evidence. No test entry point is provided.
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

import gnn
from gnn import torch, nn
from common.baseline_run import BASE, configuration, make_output, save_result, write_csv
from common.dataset import DEFAULT_OUTPUT, load_development, load_graph
from common.preprocessing import sha256


def extension_configuration():
    shared = configuration()
    result = {key: shared[key] for key in (
        "minimum_validation_recall", "cv_folds", "cv_seed", "bootstrap_replicates", "bootstrap_seed", "confidence_level")}
    result.update(json.loads((BASE / "configs/gnn_extensions.json").read_text(encoding="utf-8")))
    if [v["name"] for v in result["variants"]] != ["wide64", "wide128", "deep2x32"]:
        raise ValueError("Expected the three recorded capacity variants")
    if result["epoch_grid"] != [20, 40] or result["cv_folds"] != 5 or result["cv_seed"] != 42:
        raise ValueError("Unrecorded change to the fixed CV/epoch protocol")
    return result


def protected_files():
    paths = set()
    for name in ("movie_high_rating_rate", "logistic_regression", "knn", "gnn", "gnn_repeat_verified", "logistic_regression_genre", "knn_genre"):
        root = BASE / "outputs" / name
        paths.update(p for p in root.rglob("*") if p.is_file())
    for relative in ("train.npz", "validation.npz", "graph.npz", "manifest.json"):
        paths.add(DEFAULT_OUTPUT / relative)
    paths.update(BASE / name for name in ("gnn.py", "logistic_regression.py", "knn.py", "configs/gnn.json", "configs/baseline.json", "configs/knn.json", "configs/genre.json"))
    return {str(path.relative_to(BASE)).replace("\\", "/"): sha256(path) for path in sorted(paths)}


def install_test_guard():
    opened = set()
    root = DEFAULT_OUTPUT.resolve()

    def guard(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            path = Path(args[0].decode() if isinstance(args[0], bytes) else args[0]).resolve()
            if root in path.parents:
                relative = path.relative_to(root).as_posix()
                if relative == "test_inputs.npz" or relative.startswith("sealed/"):
                    raise PermissionError("GNN extension process blocked real test access")
                opened.add(relative)

    sys.addaudithook(guard)
    return opened


def subset(batch, rows):
    return {key: batch[key][rows] for key in ("row_id", "user_index", "movie_index", "y")}


def new_model(variant, adjacency, device):
    if variant["depth"] == 1:
        model = gnn.ShallowMovieGNN(variant["taps"], variant["hidden_channels"])
    else:
        from common.deep_movie_gnn import TwoLayerMovieGNN
        model = TwoLayerMovieGNN(adjacency, taps=variant["taps"], hidden_channels=variant["hidden_channels"])
    if sum(p.numel() for p in model.parameters()) != variant["parameter_count"]:
        raise ValueError("Architecture parameter count differs from the recorded plan")
    return model.to(device)


def predict(model, data, batch, variant, device, batch_size=256):
    model.eval()
    values = []
    with torch.no_grad():
        for start in range(0, len(batch["y"]), batch_size):
            end = start + batch_size
            if variant["depth"] == 1:
                logits = model(torch.as_tensor(data[start:end], dtype=torch.float32, device=device))
            else:
                users = torch.as_tensor(batch["user_index"][start:end], dtype=torch.long, device=device)
                movies = torch.as_tensor(batch["movie_index"][start:end], dtype=torch.long, device=device)
                logits = model(data, users, movies)
            values.append(torch.sigmoid(logits).cpu().numpy())
    score = np.concatenate(values).astype(np.float64)
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite extension score")
    return score


def fit_one(batch, data, variant, config, adjacency, device, seed, epochs, heldout=None):
    """Only supplied fit targets backpropagate; heldout is an inner-CV checkpoint."""
    gnn.seed_training(seed, config["thread_limit"])
    model = new_model(variant, adjacency, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss()
    labels = torch.as_tensor(batch["y"], dtype=torch.float32, device=device)
    if variant["depth"] == 1:
        train_data = torch.as_tensor(data, dtype=torch.float32, device=device)
    else:
        train_data = data
    rng = np.random.default_rng(seed)
    scores, trace = {}, []
    for epoch in range(1, max(epochs) + 1):
        model.train()
        order = rng.permutation(len(labels))
        running = 0.
        for start in range(0, len(labels), config["batch_size"]):
            rows = order[start:start + config["batch_size"]]
            indices = torch.as_tensor(rows, dtype=torch.long, device=device)
            optimizer.zero_grad(set_to_none=True)
            if variant["depth"] == 1:
                logits = model(train_data[indices])
            else:
                users = torch.as_tensor(batch["user_index"][rows], dtype=torch.long, device=device)
                movies = torch.as_tensor(batch["movie_index"][rows], dtype=torch.long, device=device)
                logits = model(train_data, users, movies)
            loss = criterion(logits, labels[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite extension training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            running += float(loss.detach()) * len(rows)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite extension parameters")
        train_ap, heldout_ap = None, None
        if epoch in epochs:
            train_ap = float(average_precision_score(batch["y"], predict(model, data, batch, variant, device)))
            if heldout is not None:
                heldout_batch, heldout_data = heldout
                heldout_ap = float(average_precision_score(heldout_batch["y"], predict(model, heldout_data, heldout_batch, variant, device)))
                scores[epoch] = heldout_ap
        trace.append({"epoch": epoch, "train_loss": running / len(labels), "train_AP": train_ap, "heldout_AP": heldout_ap})
    return model, scores, trace


def run_variant(variant, config, train, validation, graph, root, device):
    output = make_output("gnn_extensions", Path(root) / variant["name"])
    adjacent = gnn.normalized_adjacency(graph)
    protocol = {"variant": variant, "config": config, "test_accessed": False,
                "graph_sha256": sha256(DEFAULT_OUTPUT / "graph.npz"),
                "selection": "same training-only 5-fold AP; select fixed epoch checkpoint; ties prefer fewer epochs",
                "input_channels": ["history_rating/5", "history_mask"], "genre_used": False}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    start = perf_counter()
    if variant["depth"] == 1:
        train_data = gnn.polynomial_features(graph, train, variant["taps"])
    else:
        from common.deep_movie_gnn import build_full_history_taps
        train_data = torch.as_tensor(build_full_history_taps(graph, variant["taps"]), dtype=torch.float32, device=device)
        # The same history-only tensor is legal in every fold. Only target fit
        # labels vary; there are no train/validation target values in this input.
    y = train["y"]
    folds = list(StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["cv_seed"]).split(train["X_raw"], y))
    fold_id = np.empty(len(y), dtype=np.int8)
    records, history = [], []
    for fold, (fit, heldout) in enumerate(folds):
        fold_id[heldout] = fold
        fit_data = train_data[fit] if variant["depth"] == 1 else train_data
        heldout_data = train_data[heldout] if variant["depth"] == 1 else train_data
        _, ap, trace = fit_one(subset(train, fit), fit_data, variant, config, adjacent, device,
                               config["training_seed"] + fold, config["epoch_grid"],
                               heldout=(subset(train, heldout), heldout_data))
        for epochs, value in sorted(ap.items()):
            records.append([epochs, fold, len(fit), len(heldout), value])
        history.extend({"stage": "cv", "fold": fold, **row} for row in trace)
        print(f"{variant['name']} fold {fold + 1}/5 AP {ap}", flush=True)
        write_csv(output / "cv_folds.csv", ["epochs", "fold", "n_fit", "n_heldout", "AP"], records)
    summary = []
    for epochs in config["epoch_grid"]:
        values = [row[-1] for row in records if row[0] == epochs]
        summary.append([epochs, float(np.mean(values)), float(np.std(values, ddof=1))])
    chosen = max(summary, key=lambda row: (row[1], -row[0]))
    model, _, trace = fit_one(train, train_data, variant, config, adjacent, device, config["training_seed"], [chosen[0]])
    history.extend({"stage": "final_fit", "fold": -1, **row} for row in trace)
    fit_seconds = perf_counter() - start
    write_csv(output / "cv_summary.csv", ["epochs", "mean_AP", "fold_AP_std"], summary)
    write_csv(output / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    if (output / "cv_assignments.csv").read_bytes() != (BASE / "outputs/gnn/cv_assignments.csv").read_bytes():
        raise ValueError("Extension training folds do not match the existing GNN")
    write_csv(output / "training_history.csv", ["stage", "fold", "epoch", "train_loss", "train_AP", "heldout_AP"],
              ([row[key] for key in ("stage", "fold", "epoch", "train_loss", "train_AP", "heldout_AP")] for row in history))
    validation_data = gnn.polynomial_features(graph, validation, variant["taps"]) if variant["depth"] == 1 else train_data
    score = predict(model, validation_data, validation, variant, device)
    torch.save({"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "variant": variant, "epochs": chosen[0], "config": config,
                "graph_sha256": protocol["graph_sha256"]}, output / "model.pt")
    checkpoint = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    restored = new_model(checkpoint["variant"], adjacent, device)
    restored.load_state_dict(checkpoint["state_dict"])
    replay = predict(restored, validation_data, validation, variant, device)
    np.testing.assert_array_equal(score, replay)
    return save_result("gnn_extensions", validation, score, config, output, {
        "variant_name": variant["name"], "variant": variant, "selected_parameters": {"epochs": chosen[0]},
        "parameter_count": variant["parameter_count"], "max_hops": variant["max_hops"],
        "cv_mean_AP": chosen[1], "fit_and_cv_seconds": fit_seconds,
        "graph_sha256": protocol["graph_sha256"],
        "config_sha256": sha256(BASE / "configs/gnn_extensions.json"),
        "original_gnn_sha256": sha256(BASE / "gnn.py"),
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "model_restore_verified": True,
        "all_training_losses_finite": all(np.isfinite(row["train_loss"]) for row in history),
        "final_train_AP": next(row["train_AP"] for row in reversed(history) if row["stage"] == "final_fit"),
        "genre_used": False, "input_channels": protocol["input_channels"],
        "limitation": "Exploratory follow-up after observing original validation results; three unadjusted comparisons; validation selects thresholds; bootstrap excludes training/selection uncertainty."})


def run(output_dir=None, device="auto", only=None):
    opened = install_test_guard()
    config = extension_configuration()
    root = Path(output_dir) if output_dir else BASE / "outputs/gnn_extensions"
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite extension experiment: {root}")
    snapshot = protected_files()
    root.mkdir(parents=True)
    (root / "preservation_before.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    (root / "experiment_plan.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    selected_device = gnn.resolve_device(device)
    train, validation = load_development()
    graph = load_graph()
    gnn.verify_development_alignment(train, validation, graph)
    for batch in (train, validation):
        if graph["history_mask"][batch["user_index"], batch["movie_index"]].any():
            raise ValueError("A target occurs in observed history")
    results = {}
    try:
        for variant in config["variants"]:
            if only is None or variant["name"] in only:
                results[variant["name"]] = run_variant(variant, config, train, validation, graph, root, selected_device)
    finally:
        after = protected_files()
        unchanged = after == snapshot
        evidence = {"status": "PASS" if unchanged else "FAIL", "original_files_preserved": unchanged,
                    "protected_file_count": len(snapshot), "test_accessed": False,
                    "preprocessed_files_opened": sorted(opened), "before": snapshot, "after": after}
        (root / "preservation.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        if not unchanged:
            raise AssertionError("Existing experiment files changed during extension training")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--only", nargs="+", choices=("wide64", "wide128", "deep2x32"))
    args = parser.parse_args()
    results = run(args.output_dir, args.device, args.only)
    print(json.dumps({name: {"metrics": result["metrics"], "CV_AP": result["cv_mean_AP"]} for name, result in results.items()}, indent=2))
