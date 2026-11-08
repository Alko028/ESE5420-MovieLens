"""Validate real development artifacts without opening sealed test outcomes."""

import json
from pathlib import Path

import numpy as np

from common import load_development, load_final_test, load_graph
from common.dataset import DEFAULT_OUTPUT
from common.raw_data import load_movielens


def main():
    root = DEFAULT_OUTPUT
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    raw = load_movielens(Path(manifest["input_directory"]))
    train, validation = load_development(root)
    graph = load_graph(root)
    for batch in (train, validation):
        np.testing.assert_array_equal(raw.pairs[batch["row_id"]],
                                      np.column_stack((batch["user_id"], batch["movie_id"])))
        np.testing.assert_array_equal(batch["y"], (raw.ratings[batch["row_id"]] >= 4).astype(np.int8))
        assert not graph["history_mask"][batch["user_index"], batch["movie_index"]].any()
        assert np.isfinite(batch["X"]).all()
    np.testing.assert_allclose(train["X"].mean(axis=0), 0, atol=2e-5)
    np.testing.assert_allclose(train["X"].std(axis=0), 1, atol=2e-5)
    # Independently recompute real neighborhood summaries using explicit edges.
    r, mask = graph["history_rating"], graph["history_mask"]
    src, dst = graph["edge_index"]
    weights = graph["edge_weight"]
    for i in np.linspace(0, len(validation["row_id"]) - 1, 100, dtype=int):
        u, m = validation["user_index"][i], validation["movie_index"][i]
        edge = dst == m
        neighbors, w = src[edge], weights[edge]
        observed = mask[u, neighbors]
        selected, w = neighbors[observed], w[observed]
        if len(selected):
            expected = [np.average(r[u, selected], weights=w),
                        np.average(r[u, selected] >= 4, weights=w), len(selected)]
        else:
            own = r[u, mask[u]]
            expected = [own.mean(), (own >= 4).mean(), 0]
        np.testing.assert_allclose(validation["X_raw"][i, 6:], expected, rtol=2e-6, atol=1e-6)
    try:
        load_final_test(root)
    except PermissionError:
        pass
    else:
        raise AssertionError("Final test gate failed")
    result = {
        "actual_data_checks": ["raw IDs and development labels aligned", "no targets in history mask",
                               "finite nine-dimensional inputs", "training mean 0 / std 1",
                               "100 independent neighborhood checks", "test access blocked"],
        "train_shape": list(train["X"].shape),
        "validation_shape": list(validation["X"].shape),
        "graph_edges": len(weights), "history_shape": list(r.shape),
        "no_neighbor_counts": manifest["no_neighbor_counts"],
        "test_metrics_computed": False, "models_trained": False,
    }
    (root / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
