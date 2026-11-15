"""Nine-feature kNN classifier with train-only CV and validation evaluation."""

import argparse
import json
from time import perf_counter

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from common.baseline_run import BASE, configuration, load_development, make_output, save_result, write_csv


def knn_configuration():
    shared = configuration()
    config = {k: shared[k] for k in ('minimum_validation_recall', 'cv_folds', 'cv_seed',
                                    'bootstrap_replicates', 'bootstrap_seed', 'confidence_level')}
    config.update(json.loads((BASE / 'configs/knn.json').read_text(encoding='utf-8')))
    return config


def model_for(k, weights, config):
    return make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=k, weights=weights, metric=config['metric'],
        algorithm=config['algorithm'], n_jobs=config['n_jobs']))


def tune_and_fit(train, config):
    X, y = train['X_raw'].astype(np.float64), train['y']
    folds = list(StratifiedKFold(n_splits=config['cv_folds'], shuffle=True,
                                random_state=config['cv_seed']).split(X, y))
    if not np.array_equal(np.unique(y), [0, 1]):
        raise ValueError('Both binary classes are required')
    if min(config['n_neighbors_grid']) < 1 or max(config['n_neighbors_grid']) > min(len(fit) for fit, _ in folds):
        raise ValueError('Neighbor grid exceeds available fold training samples')
    records, summaries = [], []
    with threadpool_limits(limits=config['thread_limit']):
        for k in sorted(config['n_neighbors_grid']):
            for weights in config['weights_grid']:
                values = []
                for fold, (fit, heldout) in enumerate(folds):
                    # Never use X pre-scaled with statistics from all training rows.
                    model = model_for(k, weights, config)
                    model.fit(X[fit], y[fit])
                    score = model.predict_proba(X[heldout])[:, 1]
                    ap = float(average_precision_score(y[heldout], score))
                    records.append([k, weights, fold, len(fit), len(heldout), ap])
                    values.append(ap)
                summaries.append([k, weights, float(np.mean(values)), float(np.std(values, ddof=1))])
        # Exact AP ties favor smaller k, then uniform weights. No validation scores used.
        best = max(summaries, key=lambda row: (row[2], -row[0], row[1] == 'uniform'))
        model = model_for(best[0], best[1], config)
        model.fit(X, y)
    fold_id = np.empty(len(y), dtype=np.int8)
    for fold, (_, heldout) in enumerate(folds):
        fold_id[heldout] = fold
    return model, {'n_neighbors': best[0], 'weights': best[1]}, records, summaries, fold_id


def run(output_dir=None):
    config = knn_configuration()
    train, validation = load_development()
    output = make_output('knn', output_dir)
    start = perf_counter()
    model, selected, records, summaries, fold_id = tune_and_fit(train, config)
    fit_seconds = perf_counter() - start
    write_csv(output / 'cv_folds.csv', ['k', 'weights', 'fold', 'n_fit', 'n_heldout', 'AP'], records)
    write_csv(output / 'cv_summary.csv', ['k', 'weights', 'mean_AP', 'fold_AP_std'], summaries)
    write_csv(output / 'cv_assignments.csv', ['row_id', 'fold'], zip(train['row_id'], fold_id))
    scaler = model[0]
    # kNN's fitted state is the scaled training examples and their labels (local only).
    np.savez_compressed(output / 'model.npz', mean=scaler.mean_, scale=scaler.scale_,
                        X_fit=scaler.transform(train['X_raw'].astype(np.float64)), y_fit=train['y'],
                        feature_names=train['feature_names'], n_neighbors=np.asarray(selected['n_neighbors']),
                        weights=np.asarray(selected['weights']), metric=np.asarray(config['metric']))
    prediction_start = perf_counter()
    with threadpool_limits(limits=config['thread_limit']):
        scores = model.predict_proba(validation['X_raw'].astype(np.float64))[:, 1]
        with np.load(output / 'model.npz', allow_pickle=False) as saved:
            restored = KNeighborsClassifier(n_neighbors=int(saved['n_neighbors']), weights=str(saved['weights']),
                                            metric=str(saved['metric']), algorithm=config['algorithm'], n_jobs=1)
            restored.fit(saved['X_fit'], saved['y_fit'])
            replay = restored.predict_proba((validation['X_raw'].astype(float) - saved['mean']) / saved['scale'])[:, 1]
        np.testing.assert_allclose(scores, replay, atol=1e-14, rtol=1e-14)
    return save_result('knn', validation, scores, config, output, {
        'selected_parameters': selected,
        'cv_mean_AP': next(row[2] for row in summaries if row[:2] == [selected['n_neighbors'], selected['weights']]),
        'fit_and_cv_seconds': fit_seconds, 'prediction_and_restore_check_seconds': perf_counter() - prediction_start,
        'cv': '5-fold stratified shuffled training-target rows; fold-local scaling; mean AP selection',
        'limitation': 'validation selects threshold; k and weights selected by training-only CV; intervals exclude training/selection uncertainty',
        'device': 'CPU', 'model_restore_verified': True})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir')
    args = parser.parse_args()
    result = run(args.output_dir)
    print(json.dumps({k: result[k] for k in ('selected_parameters', 'cv_mean_AP', 'metrics', 'intervals', 'threshold', 'test_accessed')}, indent=2))
