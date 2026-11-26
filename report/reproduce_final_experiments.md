# Reproduce the current paper experiments

This guide covers all nine methods. Start in a fresh clone, separate from existing outputs. This is a single-author project: public instructions support reproduction, not a claim of independent teammate replication.

## 1. Environment and data

Use the installation commands in the root README. Automated all-model execution requires CUDA: the runner explicitly selects --device cuda. Individual GNN entry points support --device cpu, but cross-device weights/results are not promised to match bit for bit. Reference environment: Windows, Python 3.13.9, pinned base requirements, PyTorch 2.9.1/CUDA 12.8.

Download official MovieLens-100K separately. Place u.data, u.item, u.genre and u.info in Lab3/ml-100k/. Demographics are not needed. From repository root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s "data training/tests" -v
.\.venv\Scripts\python.exe "data training/prepare_data.py"
.\.venv\Scripts\python.exe "data training/verify_data.py"
```

Preprocessing creates local data training/outputs/preprocessed and data training/splits/baseline_v1. Seeds are fixed in configs/preprocessing.json; existing output is not overwritten. Expected shapes: training (19585,9), validation (9575,9); 9570 eligible test targets. Preparing sealed files does not evaluate test performance. Training uses development targets/history only.

Cross-BLAS cosine near ties can affect a top-k neighbor. The paper kept its original saved graph fixed. Those private derived arrays are not redistributed, so rebuilding inputs across platforms is not guaranteed to reproduce their exact hashes. Record your clone's manifest and keep inputs fixed across every method.

## 2. All development experiments (CUDA)

```powershell
.\.venv\Scripts\python.exe "data training/rerun_all_experiments.py"
```

The historical filename identifies the complete fixed development sequence. Its 15 stages run tests; floor; LR; original-grid kNN; original GNN and a repeat; baseline/GNN comparisons; genre training with scaler/data sanity checks; genre comparison/report; three GNN capacity variants/comparison; bounded kNN grid extension/comparison. An access guard blocks actual test inputs/labels throughout. It stops on failure, writes logs to outputs/full_rerun_20261125, and refuses existing formal model outputs. It does not automatically delete or retry them.

The runner's historical purpose text concerns the author's earlier observed test. It does not mean a new clone has already evaluated its test. Do not alter frozen training source to change provenance wording.

| Method | Entry under data training/ | Config under configs/ |
|---|---|---|
| Movie-rate floor | movie_high_rating_rate.py | baseline.json |
| LR | logistic_regression.py | baseline.json |
| Original-grid kNN | knn.py | knn.json |
| Original GNN | gnn.py | gnn.json |
| LR/kNN + genre | logistic_regression_genre.py / knn_genre.py | genre.json and baseline/kNN grids |
| GNN wide64/wide128/deep2x32 | gnn_extensions.py | gnn_extensions.json |
| Final bounded kNN grid | knn_grid_extension.py | knn_grid_extension.json |

Final kNN uses outputs/knn_grid_extension/history; genre kNN uses outputs/knn_grid_extension/genre. Do not substitute the historical original-grid kNN based on a higher development score.

verify_full_rerun.py is historical audit source comparing the author's two runs with a private local archive. That archive is not distributed, so this script is NOT a fresh-clone step. Portable tests, training sanity checks and freeze/verify below do not need the archive.

## 3. Freeze and verify without opening test

```powershell
.\.venv\Scripts\python.exe "data training/final_evaluation.py" freeze --protocol-dir "data training/protocols/reproduction_20261125" --device cuda
.\.venv\Scripts\python.exe "data training/final_evaluation.py" verify --protocol-dir "data training/protocols/reproduction_20261125"
```

Freeze restores nine methods, checks data/row/fold/model alignment, replays validation predictions/bootstrap, and binds source/config/environment/artifact hashes. Verify uses development artifacts and expected test hashes from the manifest, without opening test labels. Configurations, weights, scalers, thresholds and all 11 comparisons are fixed before evaluation.

## 4. Evaluate each fitted model once

Only after all choices are complete:

```powershell
.\.venv\Scripts\python.exe "data training/final_evaluation.py" run-test --protocol-dir "data training/protocols/reproduction_20261125" --output-dir "data training/outputs/final_test_rerun_20261125"
```

One loader invocation evaluates all nine methods on identical target rows, with no refitting or threshold selection. A protocol-specific marker remains on failure too. Do not remove it or create another protocol to tune on observed test performance. Reproducing a published test comparison does not add independent confirmation.

Expected execution.json: PASS, final_loader_calls=1, refit=false, thresholds_reselected=false. Outputs include results.json, paired_differences.json, bootstrap arrays, predictions and PR data. User-level outputs remain local and must not be uploaded.

## 5. Figures/tables from saved results

Published reference aggregates already exist in report/final_results. Write reproduced outputs elsewhere:

```powershell
.\.venv\Scripts\python.exe "data training/build_final_report.py" --results-dir "data training/outputs/final_test_rerun_20261125" --protocol-dir "data training/protocols/reproduction_20261125" --output-dir "report/reproduced_final_results"
```

This reads saved evaluation outputs, not raw/preprocessed data. It checks point metrics, saved intervals and paired differences before making five PDF/PNG figures, four LaTeX tables, CSV and full-precision JSON. It refuses an existing output directory.

## 6. Compile the paper

With LaTeX installed, from report/:

```powershell
pdflatex -interaction=nonstopmode -halt-on-error draft.tex
pdflatex -interaction=nonstopmode -halt-on-error draft.tex
```

The official neurips_2026.sty is included. Figures resolve under final_results/figures. Current draft: five main-text pages, one reference page, two appendix pages. The later final submission needs a real video URL below the author name.

## Limits

Read all models and paired intervals; never select seeds/architectures/thresholds/training runs using test. Original GNN versus LR is primary; capacity variants are exploratory with unadjusted intervals. Bootstrap is conditional on fitted models/thresholds, omitting training/split/selection uncertainty. Deep GNN CUDA sparse backward operations may vary despite deterministic settings; exact weights or scores across retraining/hardware are not guaranteed.

The original study evaluated the same test before fixed-protocol retraining/re-evaluation. No test-based tuning or best-of-runs selection was used. A clean clone does not erase that provenance.
