"""Compare saved validation runs, with alignment checks and paired user intervals."""

import csv
import json

import numpy as np

from common.baseline_run import BASE, write_csv


def main():
    names = ['movie_high_rating_rate', 'logistic_regression', 'knn']
    results, predictions, bootstrap = {}, {}, {}
    for name in names:
        root = BASE / 'outputs' / name
        results[name] = json.loads((root / 'result.json').read_text(encoding='utf-8'))
        with (root / 'validation_predictions.csv').open() as stream:
            predictions[name] = list(csv.DictReader(stream))
        with np.load(root / 'bootstrap.npz', allow_pickle=False) as values:
            bootstrap[name] = {key: values[key] for key in values.files}
    reference = results['logistic_regression']
    keys = [(r['row_id'], r['user_id'], r['movie_id'], r['y']) for r in predictions['logistic_regression']]
    for name in names:
        r = results[name]
        if r['evaluation_partition'] != 'validation' or r['input_sha256'] != reference['input_sha256']:
            raise ValueError('Input version or evaluation partition mismatch')
        if keys != [(r['row_id'], r['user_id'], r['movie_id'], r['y']) for r in predictions[name]]:
            raise ValueError('Unpaired validation rows')
        for key in ['bootstrap_seed', 'bootstrap_replicates', 'confidence_level']:
            if r['config'][key] != reference['config'][key]:
                raise ValueError('Unpaired bootstrap configuration')
        for key in ['user_ids', 'metric_names']:
            np.testing.assert_array_equal(bootstrap[name][key], bootstrap['logistic_regression'][key])
    # Matching folds is checked by IDs as well as fold number, not only the seed.
    def folds(name):
        with (BASE / 'outputs' / name / 'cv_assignments.csv').open() as stream:
            return list(csv.DictReader(stream))
    if folds('knn') != folds('logistic_regression'):
        raise ValueError('LR and kNN training CV assignments differ')
    differences = {}
    for other in ['logistic_regression', 'movie_high_rating_rate']:
        delta = bootstrap['knn']['samples'] - bootstrap[other]['samples']
        differences['knn_minus_' + other] = {}
        for i, metric in enumerate(bootstrap['knn']['metric_names']):
            valid = delta[np.isfinite(delta[:, i]), i]
            differences['knn_minus_' + other][str(metric)] = {
                'estimate': results['knn']['metrics'][str(metric)] - results[other]['metrics'][str(metric)],
                'lower': float(np.quantile(valid, .025)), 'upper': float(np.quantile(valid, .975)),
                'valid_replicates': len(valid)}
    report = BASE.parent / 'report'
    (report / 'tables').mkdir(exist_ok=True)
    rows = []
    for name in names:
        for metric in ['precision', 'recall', 'AP']:
            r = results[name]
            rows.append([name, metric, r['metrics'][metric], r['intervals'][metric]['lower'],
                         r['intervals'][metric]['upper'], r['threshold'], 'validation'])
    write_csv(report / 'tables' / 'knn_comparison.csv',
              ['model', 'metric', 'estimate', 'ci_lower', 'ci_upper', 'threshold', 'partition'], rows)
    evidence = {'paired_differences': differences, 'validation_rows_aligned': True,
                'training_cv_assignments_identical': True, 'input_hashes_identical': True,
                'test_accessed': False, 'bootstrap_replicates': reference['config']['bootstrap_replicates']}
    (report / 'tables' / 'knn_paired_comparison.json').write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
