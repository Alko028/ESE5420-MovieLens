# Training code and shared inputs

See the [complete reproduction guide](../report/reproduce_final_experiments.md) for the current nine-method experiment, installation, CUDA execution, CV/freeze/evaluation and figures.

- Common parsing, splits, history features and evaluation: common/.
- Hyperparameters/seeds: configs/.
- Separate model entry points: movie_high_rating_rate.py, logistic_regression.py, knn.py, gnn.py, logistic_regression_genre.py, knn_genre.py, gnn_extensions.py, knn_grid_extension.py.
- Development sequence with real-test access guards: rerun_all_experiments.py.
- Frozen evaluation: final_evaluation.py; figure/table generation: build_final_report.py.
- Tests: tests/ (99 tests with PyTorch installed).

Inputs are generated locally by prepare_data.py and checked by verify_data.py. Never overwrite original experiment inputs or existing outputs. The same fixed history, eligible targets and training folds apply to all methods; LR/kNN scaling is fold-local. Development loaders do not read sealed test labels. Data, split memberships, fitted models and user-level outputs are excluded from Git.

The default automation requires CUDA; individual GNN scripts accept CPU too, with different numerical-reproduction guarantees. The guide explicitly distinguishes portable fresh-clone commands from verify_full_rerun.py, an archive-dependent historical audit.
