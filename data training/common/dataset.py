"""Shared CPU NumPy interface for frequency, LR, kNN, and future CUDA GNNs."""

import json
from pathlib import Path

import numpy as np

from .features import FEATURE_NAMES


DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "outputs" / "preprocessed"


def _read(root, filename):
    # Import locally to avoid a module initialization cycle.
    from .preprocessing import sha256

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1:
        raise ValueError("Unsupported dataset schema")
    path = root / filename
    if sha256(path) != manifest["artifact_sha256"][filename]:
        raise ValueError(f"Artifact hash mismatch: {filename}")
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _validate_batch(batch, labeled):
    n = len(batch["row_id"])
    if tuple(batch["feature_names"]) != FEATURE_NAMES:
        raise ValueError("Unexpected feature order")
    if batch["X"].shape != (n, 9) or batch["X_raw"].shape != (n, 9):
        raise ValueError("Invalid tabular shape")
    for key in ("X", "X_raw", "frequency_score"):
        if not np.isfinite(batch[key]).all():
            raise ValueError("Nonfinite model inputs")
    for key in ("user_id", "movie_id", "user_index", "movie_index", "frequency_score"):
        if batch[key].shape != (n,):
            raise ValueError(f"Invalid {key} alignment")
    if not np.array_equal(batch["user_id"] - 1, batch["user_index"]) or not np.array_equal(batch["movie_id"] - 1, batch["movie_index"]):
        raise ValueError("Official ID / tensor index mismatch")
    if labeled:
        if batch["y"].shape != (n,) or not np.isin(batch["y"], [0, 1]).all():
            raise ValueError("Invalid binary labels")
    elif "y" in batch:
        raise ValueError("Test inputs must not contain labels")
    return batch


def load_development(root=DEFAULT_OUTPUT):
    """Return (train, validation); no test artifact is accessed."""
    return tuple(_validate_batch(_read(root, f"{name}.npz"), True)
                 for name in ("train", "validation"))


def load_graph(root=DEFAULT_OUTPUT):
    """History-only graph; optional movie genres are metadata, not baseline features."""
    return _read(root, "graph.npz")


def load_final_test(root=DEFAULT_OUTPUT, *, protocol_frozen=False):
    """Explicit final-evaluation gate. This is a workflow guard, not access control."""
    if protocol_frozen is not True:
        raise PermissionError("Freeze models, hyperparameters, thresholds, and analysis before loading test labels")
    inputs = _validate_batch(_read(root, "test_inputs.npz"), False)
    labels = _read(root, "sealed/test_labels.npz")
    if not np.array_equal(inputs["row_id"], labels["row_id"]):
        raise ValueError("Test input/label alignment mismatch")
    inputs["y"] = labels["y"]
    return _validate_batch(inputs, True)
