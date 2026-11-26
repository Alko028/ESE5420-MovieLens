"""Lab3-style shallow polynomial movie GNN on the fixed LR/kNN development data.

For one graph-filter/ReLU layer and a pointwise readout, gathering the fixed
history-only polynomial signals before the learned layer is exactly equivalent
to evaluating the graph network on all movies and gathering afterwards.
"""

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

# Set before initializing CUDA; deterministic GEMMs otherwise may be unavailable.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold

try:
    import torch
    from torch import nn
except ImportError as exc:
    raise ImportError("GNN requires PyTorch; see requirements-gnn.txt and report/gnn_work_log.md") from exc

from common.baseline_run import BASE, configuration, make_output, save_result, write_csv
from common.dataset import DEFAULT_OUTPUT, load_development, load_graph
from common.preprocessing import sha256


def gnn_configuration():
    shared = configuration()
    config = {key: shared[key] for key in (
        "minimum_validation_recall", "cv_folds", "cv_seed", "bootstrap_replicates",
        "bootstrap_seed", "confidence_level")}
    config.update(json.loads((BASE / "configs/gnn.json").read_text(encoding="utf-8")))
    validate_configuration(config)
    return config


def validate_configuration(config):
    for key in ("hidden_channels", "batch_size", "thread_limit", "cv_folds"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    for key in ("taps_grid", "epoch_grid"):
        values = config[key]
        if not values or any(type(v) is not int or v < 1 for v in values) or len(set(values)) != len(values):
            raise ValueError(f"Invalid grid: {key}")
    if config["cv_folds"] < 2 or min(config["cv_seed"], config["training_seed"]) < 0:
        raise ValueError("Invalid folds or seeds")
    if not np.isfinite([config["learning_rate"], config["weight_decay"], config["gradient_clip_norm"]]).all():
        raise ValueError("Nonfinite optimizer configuration")
    if config["learning_rate"] <= 0 or config["weight_decay"] < 0 or config["gradient_clip_norm"] <= 0:
        raise ValueError("Invalid optimizer configuration")


def resolve_device(requested="auto"):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Device must be auto, cpu or cuda")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; choose --device cpu explicitly")
    device = torch.device("cuda" if requested == "cuda" or
                          (requested == "auto" and torch.cuda.is_available()) else "cpu")
    # Check that the installed wheel actually supports the available GPU.
    torch.ones(1, device=device).add_(1)
    return device


def seed_training(seed, thread_limit):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(thread_limit)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def normalized_adjacency(graph):
    """A[target, source], incoming-weight row normalization; zero rows stay zero."""
    rating = np.asarray(graph["history_rating"])
    if rating.ndim != 2:
        raise ValueError("Expected user x movie history")
    n_movies = rating.shape[1]
    edge = np.asarray(graph["edge_index"])
    weight = np.asarray(graph["edge_weight"], dtype=np.float64)
    if edge.ndim != 2 or edge.shape[0] != 2 or weight.shape != (edge.shape[1],):
        raise ValueError("Invalid graph edge shape")
    if not np.issubdtype(edge.dtype, np.integer) or (edge < 0).any() or (edge >= n_movies).any():
        raise ValueError("Invalid zero-based graph indices")
    if not np.isfinite(weight).all() or (weight <= 0).any():
        raise ValueError("Graph weights must be finite and positive")
    if (edge[0] == edge[1]).any():
        raise ValueError("Existing history graph must have no self loops (tap 0 is identity)")
    adjacency = sparse.csr_matrix((weight, (edge[1], edge[0])), shape=(n_movies, n_movies))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inverse = np.divide(1., degree, out=np.zeros_like(degree), where=degree > 0)
    return (sparse.diags(inverse) @ adjacency).astype(np.float32).tocsr()


def polynomial_features(graph, batch, taps):
    """Gather [rating/5, mask] for each tap at the exact supplied target rows.

    No labels, fitted scaling, metadata genres, user IDs as embeddings or target
    ratings are used. Recurrence is (A @ signal.T).T, i.e. signal @ A.T.
    """
    if type(taps) is not int or taps < 1:
        raise ValueError("taps must be positive")
    rating = np.asarray(graph["history_rating"], dtype=np.float32)
    mask = np.asarray(graph["history_mask"])
    if rating.ndim != 2 or mask.shape != rating.shape or not np.array_equal(mask, rating > 0):
        raise ValueError("History rating/mask mismatch")
    if not np.isfinite(rating).all() or not np.isin(rating, [0, 1, 2, 3, 4, 5]).all():
        raise ValueError("Invalid history ratings")
    users, movies = np.asarray(batch["user_index"]), np.asarray(batch["movie_index"])
    if users.ndim != 1 or movies.shape != users.shape or not len(users):
        raise ValueError("Expected aligned nonempty user/movie indices")
    if not np.issubdtype(users.dtype, np.integer) or not np.issubdtype(movies.dtype, np.integer):
        raise ValueError("Target indices must be integers")
    if (users < 0).any() or (users >= rating.shape[0]).any() or (movies < 0).any() or (movies >= rating.shape[1]).any():
        raise ValueError("Target indices outside graph catalog")
    if mask[users, movies].any():
        raise ValueError("A target pair appears in observed history: leakage")
    if (mask.sum(axis=0)[movies] == 0).any():
        raise ValueError("Target movie is not eligible under the common history rule")
    adjacency = normalized_adjacency(graph)
    signals = [rating / np.float32(5), mask.astype(np.float32)]
    gathered = []
    for _ in range(taps):
        gathered.extend(signal[users, movies] for signal in signals)
        signals = [(adjacency @ signal.T).T for signal in signals]
    features = np.column_stack(gathered).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite polynomial history features")
    return features


class ShallowMovieGNN(nn.Module):
    """One polynomial graph-filter/ReLU layer, then shared K=1 readout."""

    def __init__(self, taps, hidden_channels=32):
        super().__init__()
        self.taps, self.hidden_channels = taps, hidden_channels
        self.hidden = nn.Linear(2 * taps, hidden_channels)
        self.readout = nn.Linear(hidden_channels, 1)

    def forward(self, features):
        return self.readout(torch.relu(self.hidden(features))).squeeze(-1)


def predict_scores(model, features, device, batch_size=8192):
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            values = torch.as_tensor(features[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(torch.sigmoid(model(values)).cpu().numpy())
    score = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(score).all():
        raise FloatingPointError("Nonfinite GNN predictions")
    return score


def train_one(X, y, taps, config, device, seed, checkpoints, heldout=None):
    """Fit only supplied targets; CV heldout labels can score fixed checkpoints."""
    seed_training(seed, config["thread_limit"])
    if X.shape != (len(y), 2 * taps) or not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Invalid GNN fit inputs")
    if not checkpoints or min(checkpoints) < 1:
        raise ValueError("Invalid epoch checkpoints")
    model = ShallowMovieGNN(taps, config["hidden_channels"]).to(device)
    inputs = torch.as_tensor(X, dtype=torch.float32, device=device)
    labels = torch.as_tensor(y, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    loss_fn = nn.BCEWithLogitsLoss()
    generator = np.random.default_rng(seed)
    ap_at_checkpoint, history = {}, []
    for epoch in range(1, max(checkpoints) + 1):
        model.train()
        permutation = generator.permutation(len(y))
        total_loss = 0.
        for start in range(0, len(y), config["batch_size"]):
            indices = torch.as_tensor(permutation[start:start + config["batch_size"]], device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(inputs[indices]), labels[indices])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite GNN training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise FloatingPointError("Nonfinite GNN parameters")
        heldout_ap = None
        if heldout is not None and epoch in checkpoints:
            heldout_ap = float(average_precision_score(heldout[1], predict_scores(model, heldout[0], device)))
            ap_at_checkpoint[epoch] = heldout_ap
        history.append({"epoch": epoch, "train_loss": total_loss / len(y), "heldout_AP": heldout_ap})
    return model, ap_at_checkpoint, history


def tune_and_fit(train, features_by_taps, config, device, output=None):
    """Same StratifiedKFold rows as LR/kNN; jointly select taps and fixed epochs."""
    validate_configuration(config)
    y = np.asarray(train["y"])
    if not np.array_equal(np.unique(y), [0, 1]) or np.bincount(y.astype(int)).min() < config["cv_folds"]:
        raise ValueError("Each class needs at least cv_folds examples")
    folds = list(StratifiedKFold(config["cv_folds"], shuffle=True, random_state=config["cv_seed"]).split(train["X_raw"], y))
    fold_id = np.empty(len(y), dtype=np.int8)
    for fold, (_, heldout) in enumerate(folds):
        fold_id[heldout] = fold
    if output is not None:
        write_csv(Path(output) / "cv_assignments.csv", ["row_id", "fold"], zip(train["row_id"], fold_id))
    records, summaries, history = [], [], []
    for taps in sorted(config["taps_grid"]):
        X = features_by_taps[taps]
        for fold, (fit, heldout) in enumerate(folds):
            _, scores, trace = train_one(X[fit], y[fit], taps, config, device,
                                         config["training_seed"] + fold, sorted(config["epoch_grid"]),
                                         heldout=(X[heldout], y[heldout]))
            for epochs, ap in sorted(scores.items()):
                records.append([taps, epochs, fold, len(fit), len(heldout), ap])
            history.extend({"stage": "cv", "taps": taps, "fold": fold, **row} for row in trace)
            print(f"GNN taps={taps}, fold={fold + 1}/{len(folds)}, checkpoint AP={scores}", flush=True)
        for epochs in sorted(config["epoch_grid"]):
            values = [row[-1] for row in records if row[0] == taps and row[1] == epochs]
            summaries.append([taps, epochs, float(np.mean(values)), float(np.std(values, ddof=1))])
    # Exact AP ties favor fewer taps, then fewer epochs; no outer validation here.
    best = max(summaries, key=lambda row: (row[2], -row[0], -row[1]))
    selected = {"taps": best[0], "epochs": best[1]}
    model, _, trace = train_one(features_by_taps[best[0]], y, best[0], config, device,
                                config["training_seed"], [best[1]])
    history.extend({"stage": "final_fit", "taps": best[0], "fold": -1, **row} for row in trace)
    return model, selected, records, summaries, fold_id, history


def verify_development_alignment(train, validation, graph):
    if np.intersect1d(train["row_id"], validation["row_id"]).size:
        raise ValueError("Training/validation targets overlap")
    for batch in (train, validation):
        if len(np.unique(batch["row_id"])) != len(batch["row_id"]):
            raise ValueError("Duplicate target rows")
        if not np.array_equal(np.asarray(graph["user_ids"])[batch["user_index"]], batch["user_id"]):
            raise ValueError("User catalog alignment mismatch")
        if not np.array_equal(np.asarray(graph["movie_ids"])[batch["movie_index"]], batch["movie_id"]):
            raise ValueError("Movie catalog alignment mismatch")


def run(output_dir=None, device="auto"):
    config = gnn_configuration()
    selected_device = resolve_device(device)
    train, validation = load_development()
    graph = load_graph()
    verify_development_alignment(train, validation, graph)
    output = make_output("gnn", output_dir)
    protocol = {"architecture": "2 history channels -> polynomial graph filter -> 32 ReLU -> pointwise logit",
                "config": config, "loss": "BCEWithLogitsLoss; y=rating>=4",
                "graph_normalization": "incoming positive cosine weights normalized to row sum one; no added loops; tap0 identity",
                "input_channels": ["history_rating/5", "history_mask"], "genre_used": False,
                "selection": "mean training-only 5-fold AP jointly over taps and fixed epoch checkpoints; final refit for selected epochs",
                "device": str(selected_device), "test_accessed": False,
                "graph_sha256": sha256(DEFAULT_OUTPUT / "graph.npz"),
                "config_sha256": sha256(BASE / "configs/gnn.json"),
                "input_sha256": {name: sha256(DEFAULT_OUTPUT / name) for name in ("train.npz", "validation.npz", "manifest.json")}}
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    start = perf_counter()
    # Only history and training target IDs are needed until configuration is selected.
    features = {k: polynomial_features(graph, train, k) for k in config["taps_grid"]}
    model, selected, records, summaries, folds, history = tune_and_fit(train, features, config, selected_device, output)
    fit_seconds = perf_counter() - start
    write_csv(output / "cv_folds.csv", ["taps", "epochs", "fold", "n_fit", "n_heldout", "AP"], records)
    write_csv(output / "cv_summary.csv", ["taps", "epochs", "mean_AP", "fold_AP_std"], summaries)
    write_csv(output / "training_history.csv", ["stage", "taps", "fold", "epoch", "train_loss", "heldout_AP"],
              ([row[key] for key in ("stage", "taps", "fold", "epoch", "train_loss", "heldout_AP")] for row in history))
    validation_features = polynomial_features(graph, validation, selected["taps"])
    scores = predict_scores(model, validation_features, selected_device)
    checkpoint = {"state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                  "selected_parameters": selected, "hidden_channels": config["hidden_channels"],
                  "config": config, "graph_sha256": protocol["graph_sha256"]}
    torch.save(checkpoint, output / "model.pt")
    # Replay actual serialized weights, not just an in-memory copy.
    saved = torch.load(output / "model.pt", map_location="cpu", weights_only=True)
    restored = ShallowMovieGNN(saved["selected_parameters"]["taps"], saved["hidden_channels"]).to(selected_device)
    restored.load_state_dict(saved["state_dict"])
    replay = predict_scores(restored, validation_features, selected_device)
    np.testing.assert_array_equal(scores, replay)
    loss_finite = all(np.isfinite(row["train_loss"]) for row in history)
    return save_result("gnn", validation, scores, config, output, {
        "selected_parameters": selected,
        "cv_mean_AP": next(row[2] for row in summaries if row[:2] == [selected["taps"], selected["epochs"]]),
        "architecture": protocol["architecture"], "graph_sha256": protocol["graph_sha256"],
        "config_sha256": protocol["config_sha256"], "device": str(selected_device),
        "device_name": torch.cuda.get_device_name(selected_device) if selected_device.type == "cuda" else "CPU",
        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
        "fit_and_cv_seconds": fit_seconds, "model_restore_verified": True, "all_training_losses_finite": loss_finite,
        "cv": protocol["selection"], "input_channels": protocol["input_channels"], "genre_used": False,
        "representation_note": "same history/split/labels and eligible rows as LR/kNN; graph representation differs from their nine tabular features",
        "limitation": "validation selects threshold; training CV selects taps/epochs; intervals exclude training/selection uncertainty; graph vs tabular comparison cannot isolate graph propagation"})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    result = run(args.output_dir, args.device)
    print(json.dumps({key: result[key] for key in (
        "selected_parameters", "cv_mean_AP", "metrics", "intervals", "threshold", "device", "test_accessed")}, indent=2))
