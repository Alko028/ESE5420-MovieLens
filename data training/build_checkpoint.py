"""Generate aggregate-only checkpoint deliverables from saved validation runs."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
REPO_URL = "https://github.com/Alko028/ESE5420-MovieLens"


def main():
    report = PROJECT / "report"
    submission = report / "checkpoint_submission"
    submission.mkdir(exist_ok=True)
    results = [json.loads((BASE / "outputs" / name / "result.json").read_text())
               for name in ("movie_high_rating_rate", "logistic_regression")]
    floor, lr = results
    # Verify row/label and bootstrap universe alignment before paired differences.
    def predictions(name):
        with (BASE / "outputs" / name / "validation_predictions.csv").open() as stream:
            return list(csv.DictReader(stream))
    left, right = [predictions(r["model"]) for r in results]
    assert [(x['row_id'], x['user_id'], x['y']) for x in left] == [(x['row_id'], x['user_id'], x['y']) for x in right]
    for key in ("bootstrap_seed", "bootstrap_replicates", "confidence_level"):
        assert floor["config"][key] == lr["config"][key]
    with np.load(BASE / "outputs/movie_high_rating_rate/bootstrap.npz") as a, np.load(BASE / "outputs/logistic_regression/bootstrap.npz") as b:
        np.testing.assert_array_equal(a['user_ids'], b['user_ids'])
        differences = b['samples'] - a['samples']
    delta = {name: {"estimate": lr['metrics'][name] - floor['metrics'][name],
                    "lower": float(np.quantile(differences[:, i], .025)),
                    "upper": float(np.quantile(differences[:, i], .975))}
             for i, name in enumerate(("precision", "recall", "AP"))}
    (report / "tables" / "paired_baseline_differences.json").write_text(json.dumps(delta, indent=2) + '\n')
    with (report / "tables" / "baseline_results.csv").open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'metric', 'estimate', 'ci_lower', 'ci_upper', 'threshold', 'partition'])
        for result in results:
            for metric in ('precision', 'recall', 'AP'):
                interval = result['intervals'][metric]
                writer.writerow([result['model'], metric, result['metrics'][metric], interval['lower'], interval['upper'], result['threshold'], 'validation'])
    def score(r, name):
        ci = r['intervals'][name]
        return f"{r['metrics'][name]:.4f} (95% CI {ci['lower']:.4f}-{ci['upper']:.4f})"
    text = f"""# High-rating prediction on MovieLens 100K
## Group
LLK; Lingkai
## Changes since the proposal
Reduced five development splits to one and prioritized baselines before GNN. Following the checkpoint instructions, C is now selected by training-only K-fold CV rather than validation AP; validation selects the operating threshold.
## Data
MovieLens 100K: 100,000 ratings, 943 users, 1,682 movies. Nine history features summarize user, movie, and neighborhood mean rating, high-rating rate, and count. Output: rating >=4 is positive; ratings 1-3 are negative; unrated pairs remain unknown. No demographic or timestamp predictors. Excluded target pairs for movies absent from history: 48 training, 21 validation, 26 test.
## Split
Random within each user: floor(10%) test, floor(10%) validation, floor(20%) training targets, remainder history. Test seed 20261106; development seed 42. Counts: 61,175 history; 19,585 eligible training; 9,575 eligible validation; 9,570 eligible test. Final test labels stay sealed. History alone supplies statistics and movie similarities. Five stratified, shuffled folds within training targets use seed 42 and fold-local scaling. Users may occur across folds because this is existing-user preference completion, not cold-start or future prediction.
## Baseline
L2 logistic regression; C grid [0.001, 0.01, 0.1, 1, 10, 100]. Mean CV AP selected C={lr['selected_C']:g} (CV AP {lr['cv_mean_AP']:.4f}). Refit on training targets only. The movie history high-rating rate is a nonpersonalized reference. Thresholds maximize validation precision subject to recall >=0.5; ties favor larger recall, then larger threshold.
## Result
On held-out validation ratings, logistic regression AP is {score(lr, 'AP')}; precision is {score(lr, 'precision')}; actual recall is {lr['metrics']['recall']:.4f} at threshold {lr['threshold']:.6f}. Reference AP is {score(floor, 'AP')}; precision is {score(floor, 'precision')}, recall {floor['metrics']['recall']:.4f}.
Intervals are percentile intervals from 2,000 paired user-bootstrap resamples of 943 users (seed 20261108), retaining all ratings per sampled user and fixing models/thresholds. Validation selected thresholds, so precision intervals are descriptive and omit selection/training uncertainty. AP does not depend on the selected threshold. No final test scores were computed.
## Issues found
No duplicate pairs or history-target overlap. Some movies lack history; zero-neighbor features fall back to user history. All CV fits converged. Random splits can use later ratings to predict earlier ones; conclusions concern held-out preference completion. Only one development split, so no split-spread claim.
## Next steps
Lingkai will implement kNN, draft the paper, and time-box a shallow GNN before November 16. If GNN training is unstable, proceed with logistic regression, kNN, and genre ablation.
## Code
{REPO_URL}
"""
    expected = ['Group', 'Changes since the proposal', 'Data', 'Split', 'Baseline', 'Result', 'Issues found', 'Next steps', 'Code']
    assert [line[3:] for line in text.splitlines() if line.startswith('## ')] == expected
    words = len(' '.join(line for line in text.splitlines() if not line.startswith('#')).split())
    assert words < 500, words
    (submission / 'baseline.txt').write_text(text, encoding='utf-8')
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
    for ax, metric, title in zip(axes, ['AP', 'precision'], ['Average precision', 'Precision at selected threshold']):
        for i, (result, color) in enumerate(zip(results, ['#697386', '#1368aa'])):
            value = result['metrics'][metric]
            ci = result['intervals'][metric]
            ax.errorbar(value, i, xerr=[[value-ci['lower']], [ci['upper']-value]], fmt='o', color=color, capsize=5, markersize=7)
            ax.text(value, i+.16, f"{value:.4f}", ha='center', fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(.68, .86)
        ax.set_ylim(-.45, 1.55)
        ax.set_xlabel('Validation score')
        ax.set_yticks([0,1], ['Movie high-rating rate', 'L2 logistic regression'])
        ax.grid(axis='x', alpha=.2)
    figure.suptitle('MovieLens 100K: baseline checkpoint', fontsize=15)
    figure.text(.5, .085, 'Points: held-out validation scores | Bars: 95% user-bootstrap intervals (2,000 draws)', ha='center', fontsize=9)
    figure.text(.5, .035, 'Precision uses validation-selected thresholds (recall >= 0.5); selection uncertainty is not included. Test sealed.', ha='center', fontsize=8)
    figure.tight_layout(rect=[0,.14,1,.94])
    figure.subplots_adjust(wspace=.22)
    figure.savefig(submission / 'baseline.png', dpi=180)
    plt.close(figure)
    (report / 'baseline_checkpoint.md').write_text(
        '# Baseline checkpoint — 2026-11-08\n\n'
        '提交要求：11 月 9 日 23:59 美东时间；每组一份，必交 baseline.txt（少于 500 词），可选 baseline.png。\n\n'
        f'已生成 [baseline.txt](checkpoint_submission/baseline.txt)，正文 {words} 词；[baseline.png](checkpoint_submission/baseline.png)。\n\n'
        '| Model | Precision [95% CI] | Recall | AP [95% CI] |\n|---|---|---|---|\n' +
        '\n'.join(f"| {r['model']} | {score(r,'precision')} | {r['metrics']['recall']:.4f} | {score(r,'AP')} |" for r in results) +
        f"\n\nLR C={lr['selected_C']}; training-only 5-fold mean AP={lr['cv_mean_AP']:.4f}.\n\n"
        f"验证集配对 ΔPrecision（LR − frequency）={delta['precision']['estimate']:.4f}，95% CI [{delta['precision']['lower']:.4f}, {delta['precision']['upper']:.4f}]。这不是 GNN 主比较，也不是最终测试结论。\n\n"
        '本次完成：两 baseline、CV 调参、用户 bootstrap、模板 TXT 和 PNG。整体剩余：Canvas 提交，以及 kNN、限时 GNN、消融和最终交付。下一任务：核对提交文件并由用户上传，再推进 early draft。\n\n'
        '方法与复现细节见 [baseline_work_log.md](baseline_work_log.md)。\n', encoding='utf-8')
    print(json.dumps({'body_words':words,'submission':str(submission),'paired_differences':delta},indent=2))


if __name__ == '__main__':
    main()
