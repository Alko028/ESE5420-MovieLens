# ESE 5420 - MovieLens high-rating prediction

Single-author project by Lingkai Lei. Predict whether an existing user's held-out MovieLens-100K rating is 4-5 rather than 1-3. This is preference completion on observed ratings, not future-behavior prediction or full-catalog recommendation.

Implemented: L2 logistic regression, kNN, a shallow polynomial movie GNN, LR/kNN genre variants, three exploratory GNN capacity variants, and a movie-rate floor. Original GNN versus LR remains the primary comparison.

## Paper and reproduction

- [Complete reproduction guide](report/reproduce_final_experiments.md): setup, official data, preprocessing, all training stages, CV, freezing, evaluation and figures/tables.
- [Paper source](report/draft.tex) and [compiled draft](report/draft.pdf), NeurIPS 2026 style. The later final submission still needs its video URL.
- [Canonical aggregate results](report/final_results/README.md), [full-precision metrics](report/final_results/paper_results.json) and [figures](report/final_results/figures/).

I completed this project individually, with no teammates. Instructions enable reproduction but do not claim independent human replication. Older work logs and report/tables are historical development records. Only report/final_results supplies the current paper's results.

## Quick start

Tested: Windows, Python 3.13.9, pinned base requirements; GNN experiments used PyTorch 2.9.1/CUDA 12.8 and an RTX 5080.

```powershell
git clone https://github.com/Alko028/ESE5420-MovieLens.git
cd ESE5420-MovieLens
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "data training/requirements.txt"
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
```

Download official [MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) separately. Put u.data, u.item, u.genre and u.info in Lab3/ml-100k/. Data are not distributed here.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "data training/tests" -v
.\.venv\Scripts\python.exe "data training/prepare_data.py"
.\.venv\Scripts\python.exe "data training/verify_data.py"
```

These commands check code and prepare inputs without model training or test-performance evaluation. Follow the complete guide for all experiments. Use a fresh clone; do not delete evaluation markers to repeat test evaluation.

## Protocol

- Same ratings, fixed within-user random split and eligible targets for every method. Test seed 20261106; development/fold seed 42. Eligible targets: 19,585 training, 9,575 validation, 9,570 test.
- Nine baseline features and graph use observed history only. Genre variants add 57 columns. Unrated pairs are not negatives.
- Same five training folds select hyperparameters by mean AP. LR/kNN scalers are fold-local. Held-out target labels never enter graph/features.
- Validation maximizes precision subject to recall >= 0.5. Models/scalers/thresholds are frozen before the reported test evaluation. Each fitted model in the reported comparison is evaluated once under the same fixed protocol. Test scores do not select models, parameters, thresholds or training runs.
- 95% paired-user percentile intervals: 2,000 draws, seed 20261108. Conditional on fitted models/thresholds; no training/split/selection uncertainty. Exploratory comparisons are unadjusted for multiplicity.
- The same test data had an earlier evaluation followed by fixed-protocol retraining/re-evaluation. This is documented provenance, not another independent test; no test-based tuning or best-of-runs selection was used.

## Current test results

Precision/recall are percentages. The complete intervals and 11 paired differences are in the aggregate JSON/CSV files.

| Method | Precision (%) [95% CI] | Recall (%) | AP |
|---|---|---|---|
| Movie-rate floor | 73.67 [71.78, 75.61] | 51.08 | 0.7261 |
| LR | 80.92 [79.53, 82.37] | 50.75 | 0.8027 |
| kNN (511) | 81.35 [79.92, 82.81] | 50.70 | 0.8011 |
| Original GNN | 77.04 [75.56, 78.64] | 49.23 | 0.7690 |
| LR + genre | 80.93 [79.58, 82.43] | 50.60 | 0.8039 |
| kNN + genre (127) | 78.73 [77.28, 80.19] | 50.85 | 0.7771 |
| GNN wide64 | 77.32 [75.85, 78.85] | 49.11 | 0.7697 |
| GNN wide128 | 77.09 [75.63, 78.62] | 48.90 | 0.7702 |
| GNN deep2x32 | 78.22 [76.79, 79.68] | 50.90 | 0.7791 |

Original GNN minus LR precision is -3.88 percentage points, 95% CI [-5.06,-2.59]. kNN versus LR is inconclusive. Genre features do not clearly help LR and reduce this kNN's precision. Capacity variants remain below LR.

## Data terms and AI assistance

F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM TIIS 5(4), Article 19. https://doi.org/10.1145/2827872

Follow the [MovieLens-100K README terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt). No GroupLens/University of Minnesota endorsement is implied. Code, documentation, manuscript and aggregates only: no raw ratings, demographics, split memberships, user-level predictions, fitted models or derived user-level arrays.

Lingkai Lei is responsible for ideas, final experimental design, interpretation and main manuscript. OpenAI Codex generated/implemented code under the author's direction and assisted with debugging, experiment execution/checks, figures, literature lookup and manuscript drafting/editing. The author retains responsibility for verification and submission.
