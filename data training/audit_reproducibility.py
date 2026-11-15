"""Repeat fixed development experiments; verify actual fits and independent metrics.

Run from any directory: python "data training/audit_reproducibility.py"
No original outputs are overwritten. Test artifact access is blocked in this process.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
from sklearn.metrics import average_precision_score, precision_score, recall_score, precision_recall_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import logistic_regression as lr
import knn
import movie_high_rating_rate as floor
from common.baseline_run import BASE, configuration
from common.dataset import DEFAULT_OUTPUT, load_development
from common.features import build_history, make_features
from common.preprocessing import sha256


def read_csv(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def compare_npz(a, b):
    with np.load(a, allow_pickle=False) as left, np.load(b, allow_pickle=False) as right:
        assert left.files == right.files
        for key in left.files:
            np.testing.assert_array_equal(left[key], right[key])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=BASE / 'outputs/audit_2026-11-14')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 2:
        raise ValueError('Use at least two repeats')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    opened = set()
    forbidden = {(DEFAULT_OUTPUT / p).resolve() for p in ('test_inputs.npz', 'sealed/test_labels.npz')}

    def guard(event, values):
        if event == 'open' and isinstance(values[0], (str, bytes)):
            path = Path(values[0].decode() if isinstance(values[0], bytes) else values[0]).resolve()
            if path in forbidden:
                raise AssertionError('Test artifact access blocked: ' + str(path))
            if path.is_relative_to(DEFAULT_OUTPUT):
                opened.add(str(path.relative_to(DEFAULT_OUTPUT)))

    sys.addaudithook(guard)
    train, val = load_development()
    manifest = json.loads((DEFAULT_OUTPUT / 'manifest.json').read_text(encoding='utf-8'))
    before = {name: sha256(DEFAULT_OUTPUT / name) for name in ('train.npz', 'validation.npz', 'manifest.json')}
    # Verify raw-file identities, then read only history and development target ratings.
    source = BASE.parent / 'Lab3/ml-100k'
    for name, digest in manifest['source_sha256'].items():
        assert sha256(source / name) == digest
    split_root = BASE / 'splits/baseline_v1'
    for name, digest in manifest['split_sha256'].items():
        assert sha256(split_root / name) == digest
    with np.load(split_root / 'indices.npz', allow_pickle=False) as z:
        splits = {key: z[key] for key in z.files}
    np.testing.assert_array_equal(np.sort(np.concatenate([splits[k] for k in ('history', 'train', 'validation', 'test')])), np.arange(100000))
    needed = set(splits['history']) | set(train['row_id']) | set(val['row_id'])
    raw = {}
    with (source / 'u.data').open(encoding='ascii') as stream:
        for row, line in enumerate(stream):
            if row in needed:
                raw[row] = tuple(map(int, line.split()[:3]))
    history = np.asarray([raw[int(row)] for row in splits['history']])
    state = build_history(history[:, :2], history[:, 2], 943, 1682, manifest['config']['neighbor_k'])
    for name, batch in [('train', train), ('validation', val)]:
        assert np.isin(batch['row_id'], splits[name]).all()
        expected_rows = []
        # Membership filtering uses movie history only, with no target ratings.
        with (source / 'u.data').open(encoding='ascii') as stream:
            members = set(splits[name])
            for row, line in enumerate(stream):
                if row in members and state.movie_stats[int(line.split()[1]) - 1, 2] > 0:
                    expected_rows.append(row)
        np.testing.assert_array_equal(batch['row_id'], expected_rows)
        targets = np.asarray([raw[int(row)] for row in batch['row_id']])
        np.testing.assert_array_equal(targets[:, :2], np.column_stack((batch['user_id'], batch['movie_id'])))
        np.testing.assert_array_equal(targets[:, 2] >= 4, batch['y'])
        np.testing.assert_array_equal(make_features(state, targets[:, :2]), batch['X_raw'])
        np.testing.assert_array_equal(batch['frequency_score'], batch['X_raw'][:, 4])

    cfg = configuration()
    X, y = train['X_raw'].astype(float), train['y']
    folds = list(StratifiedKFold(n_splits=cfg['cv_folds'], shuffle=True, random_state=cfg['cv_seed']).split(X, y))
    models = [('movie_high_rating_rate', floor, 0), ('logistic_regression', lr, len(cfg['C_grid'])),
              ('knn', knn, len(knn.knn_configuration()['n_neighbors_grid']) * len(knn.knn_configuration()['weights_grid']))]
    evidence = []
    for repeat in range(1, args.repeats + 1):
        for name, module, candidates in models:
            expected = [fit for _ in range(candidates) for fit, _ in folds]
            if candidates:
                expected.append(np.arange(len(y)))
            calls = []
            original_fit = StandardScaler.fit

            def checked_fit(self, data, labels=None, **kwargs):
                indices = expected[len(calls)]
                np.testing.assert_array_equal(data, X[indices])
                np.testing.assert_array_equal(labels, y[indices])
                result = original_fit(self, data, labels, **kwargs)
                np.testing.assert_allclose(self.mean_, X[indices].mean(axis=0), atol=1e-12)
                calls.append(len(indices))
                return result

            output = args.output_dir / f'repeat_{repeat}' / name
            with patch.object(StandardScaler, 'fit', checked_fit):
                result = module.run(output)
            assert len(calls) == len(expected)
            reference_dir = BASE / 'outputs' / name
            reference = json.loads((reference_dir / 'result.json').read_text(encoding='utf-8'))
            for key in ('metrics', 'intervals', 'threshold', 'config', 'input_sha256'):
                assert result[key] == reference[key], (name, key)
            for key in ('selected_C', 'selected_parameters', 'cv_mean_AP'):
                if key in result:
                    assert result[key] == reference[key]
            files = ['validation_predictions.csv', 'validation_pr_curve.csv']
            if candidates:
                files += ['cv_assignments.csv', 'cv_summary.csv', 'cv_folds.csv']
                compare_npz(output / 'model.npz', reference_dir / 'model.npz')
                assert read_csv(output / 'cv_assignments.csv') == read_csv(BASE / 'outputs/logistic_regression/cv_assignments.csv')
            for filename in files:
                assert (output / filename).read_bytes() == (reference_dir / filename).read_bytes(), filename
            compare_npz(output / 'bootstrap.npz', reference_dir / 'bootstrap.npz')
            rows = read_csv(output / 'validation_predictions.csv')
            for key in ('row_id', 'user_id', 'movie_id', 'y'):
                np.testing.assert_array_equal([int(row[key]) for row in rows], val[key])
            scores = np.asarray([float(row['score']) for row in rows])
            predicted = scores >= result['threshold']
            independent = [precision_score(val['y'], predicted), recall_score(val['y'], predicted), average_precision_score(val['y'], scores)]
            np.testing.assert_allclose(independent, [result['metrics'][k] for k in ('precision', 'recall', 'AP')], atol=1e-14, rtol=1e-14)
            p, r, t = precision_recall_curve(val['y'], scores)
            choices = np.flatnonzero(r[:-1] >= cfg['minimum_validation_recall'])
            best = max(choices, key=lambda i: (p[i], r[i], t[i]))
            assert t[best] == result['threshold']
            # Independent cluster resampling + sklearn weighted metrics for 20 replicas.
            users, inv = np.unique(val['user_id'], return_inverse=True)
            rng = np.random.default_rng(cfg['bootstrap_seed'])
            with np.load(output / 'bootstrap.npz', allow_pickle=False) as z:
                for b in range(20):
                    weights = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users))[inv]
                    values = [precision_score(val['y'], predicted, sample_weight=weights),
                              recall_score(val['y'], predicted, sample_weight=weights),
                              average_precision_score(val['y'], scores, sample_weight=weights)]
                    np.testing.assert_allclose(values, z['samples'][b], atol=1e-14, rtol=1e-14)
                for i, metric in enumerate(('precision', 'recall', 'AP')):
                    np.testing.assert_allclose(np.quantile(z['samples'][:, i], [.025, .975]),
                        [result['intervals'][metric]['lower'], result['intervals'][metric]['upper']], atol=1e-14)
            evidence.append({'repeat': repeat, 'model': name, 'scaler_fits_checked': calls,
                             'exact_match_to_original': True, 'metrics': result['metrics']})
            print(f'PASS repeat {repeat}: {name}; checked {len(calls)} actual scaler fits', flush=True)
    assert before == {name: sha256(DEFAULT_OUTPUT / name) for name in before}
    report = {'status': 'PASS', 'repeats_per_model': args.repeats, 'runs': evidence,
              'train_shape': list(X.shape), 'validation_shape': list(val['X_raw'].shape),
              'input_sha256': before, 'preprocessed_files_opened': sorted(opened),
              'test_artifact_access_blocked': True, 'features_rebuilt_exactly': True,
              'raw_and_split_hashes_verified': True, 'cv_assignments_identical': True,
              'independent_sklearn_bootstrap_replicates_per_run': 20,
              'all_2000_bootstrap_values_match_original': True}
    (args.output_dir / 'audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print('Audit passed: ' + str(args.output_dir / 'audit.json'))


if __name__ == '__main__':
    main()
