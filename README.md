# ESE 5420 — MovieLens high-rating prediction

Reproducible baseline checkpoint: predict whether an existing user gives a held-out movie 4–5 stars. The first course baseline is L2 logistic regression, with a nonpersonalized movie high-rating-rate reference. kNN and a shallow GNN are future work.

## Reproduce

Tested with Python 3.13.9. From a **fresh clone**:

```powershell
git clone https://github.com/Alko028/ESE5420-MovieLens.git
cd ESE5420-MovieLens
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r "data training/requirements.txt"
```

Download [MovieLens 100K from GroupLens](https://grouplens.org/datasets/movielens/100k/) yourself and extract it so that `Lab3/ml-100k/u.data`, `u.item`, `u.genre`, and `u.info` exist. Data are not included in this repository.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "data training/tests" -v
.\.venv\Scripts\python.exe "data training/prepare_data.py"
.\.venv\Scripts\python.exe "data training/verify_data.py"
.\.venv\Scripts\python.exe "data training/movie_high_rating_rate.py"
.\.venv\Scripts\python.exe "data training/logistic_regression.py"
.\.venv\Scripts\python.exe "data training/build_checkpoint.py"
```

Output directories are not overwritten. Run the sequence once per fresh clone. To repeat a model using the existing fixed data, give its script a new `--output-dir`; do not change seeds to select a better run.

## Protocol

- One random-within-user history/train/validation/test split. Test seed 20261106; development seed 42. Target ratings never enter history statistics or the graph.
- Nine inputs: user mean/high-rating-rate/count, movie mean/high-rating-rate/count, neighborhood weighted mean/high-rating-rate/count. History-only raw-rating cosine, up to 20 positive neighbors; no-neighbor fallback uses user history.
- Five stratified shuffled folds **inside training targets** (seed 42). Each fold fits its own scaler. Select C by mean CV AP from `[0.001, 0.01, 0.1, 1, 10, 100]`; selected C is 0.1.
- Refit on training targets. Validation chooses a threshold maximizing precision subject to recall >= 0.5. The final test set remains sealed.
- Percentile 95% intervals from 2,000 paired user-bootstrap draws (seed 20261108), with models and thresholds fixed. Intervals do not capture training or model/threshold selection uncertainty.

## Validation results

| Method | Precision (95% CI) | Recall | AP (95% CI) |
|---|---|---|---|
| Movie high-rating rate | 0.7399 [0.7185, 0.7599] | 0.5075 | 0.7338 [0.7128, 0.7538] |
| L2 logistic regression | 0.8193 [0.8040, 0.8339] | 0.5016 | 0.8067 [0.7914, 0.8203] |

These are development results on held-out observed ratings, not final-test or full-catalog recommendation estimates. Only one split was run; no best-seed selection or split-spread claim.

The submission files are in `report/checkpoint_submission/`. Chinese [personal results summary](report/personal_checkpoint_summary.md) and [reproduction guide](report/reproduction_guide.md) explain the workflow. Method details: [baseline work log](report/baseline_work_log.md).

## Data attribution and publication boundaries

F. Maxwell Harper and Joseph A. Konstan. 2015. *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems 5(4), Article 19. https://doi.org/10.1145/2827872

Follow the [MovieLens 100K README terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt). No endorsement by GroupLens or the University of Minnesota is implied. This repository contains code and aggregate results only: no raw ratings, demographics, per-user predictions, split memberships, or derived user-level arrays. Run scripts locally to reproduce those private artifacts.

AI assistance: OpenAI Codex assisted with implementation, debugging, experiment execution, and documentation. The author is responsible for reviewing and verifying the methods and results.
