# Concrete Compressive Strength Prediction

A reproducible machine-learning mini-project comparing linear regression and a
random forest on the UCI Concrete Compressive Strength dataset. A constant-mean
predictor provides an additional sanity-check baseline.

## Research question

Can a nonlinear model improve strength predictions for held-out concrete mixes
relative to linear regression? Inputs are seven ingredient quantities (kg/m^3)
and age (days); the output is compressive strength (MPa).

## Installation and execution

Run commands from this directory, using Python 3.14. The pinned requirements
record the environment used for the supplied results.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/download_data.py
python src/train.py
python src/predict.py --input examples/inputs.csv --output results/example_predictions.csv
```

On macOS/Linux activate with `source .venv/bin/activate`. If PowerShell activation
is disabled, use `.\.venv\Scripts\python.exe` in place of `python` after creating
the environment. The supplied CSV and original source archive also allow training
without downloading data again. Dependencies must first be installed.

## Dataset and attribution

I-Cheng Yeh (1998), *Concrete Compressive Strength*, UCI Machine Learning
Repository, DOI: https://doi.org/10.24432/C5PK67.
Dataset page: https://archive.ics.uci.edu/dataset/165/concrete-compressive-strength.
UCI distributes this dataset under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.

The original dataset has 1,030 rows, eight features, and one target. The download
script retains the archive, renames columns, and exports values without changing
row order. Source and checksum are recorded in `data/provenance.json`. Training
removes exactly duplicated complete rows, keeping the first; original row indices
are retained in the split and prediction files. No new labels were collected.

| Column | Meaning | Unit |
|---|---|---|
| cement | Cement | kg/m^3 |
| slag | Blast furnace slag | kg/m^3 |
| fly_ash | Fly ash | kg/m^3 |
| water | Water | kg/m^3 |
| superplasticizer | Superplasticizer | kg/m^3 |
| coarse_aggregate | Coarse aggregate | kg/m^3 |
| fine_aggregate | Fine aggregate | kg/m^3 |
| age | Age | days |
| strength_mpa | Measured compressive strength (training target only) | MPa |

## Evaluation protocol

1. Check that values are finite and remove exact full-row duplicates.
2. Define mix groups using exact equality of the seven ingredient quantities,
   excluding age. Use `GroupShuffleSplit(test_size=0.20, random_state=42)`.
   This holds out 20% of groups, not necessarily 20% of rows.
3. Within the training set, use five-fold `GroupKFold`. Each mix stays entirely
   within one fold. StandardScaler for linear regression is fitted inside the
   pipeline, separately for each fold. No scaling is needed for the forest.
4. Evaluate the mean predictor and linear regression with identical CV folds.
   Tune a 300-tree forest over max_depth={None,10}, min_samples_leaf={1,3,5},
   max_features={1.0,0.7}; choose by validation MAE (12 candidates).
5. Select the model family by mean CV MAE, fit on the training partition only,
   and evaluate the three models on the untouched test partition.

No test scores are used to choose hyperparameters or the saved model. The best
forest's CV score is a tuning score, not an unbiased nested-CV estimate. The held-out
test estimates performance for unseen exact mixes from this dataset. Exact
grouping cannot establish independence between similar mixes or experimental
sources, for which identifiers are unavailable in the input table.

## Outputs and inference

Executed results (seed 42; 810 development rows, 195 test rows; 342/86 mix groups):

| Model | CV MAE mean +/- SD | Test MAE | Test RMSE | Test R-squared |
|---|---:|---:|---:|---:|
| Mean baseline | 13.22 +/- 0.71 | 12.92 | 16.16 | -0.0003 |
| Linear regression | 8.44 +/- 0.23 | 7.52 | 9.58 | 0.648 |
| Random forest | 4.52 +/- 0.55 | 4.23 | 5.54 | 0.882 |

MAE and RMSE are in MPa. The random forest reduced held-out MAE by 43.7%
relative to linear regression on this split. Full-precision values are retained
in the CSV and JSON outputs. There were 25 exact duplicate rows removed,
leaving 1,005 observations. The selected forest has 300 trees, unlimited depth,
min_samples_leaf=1 and max_features=0.7. Its validation score was used for tuning;
it is not a nested-CV estimate. The single test split has no reported confidence interval.

- `results/metrics.csv`: CV MAE mean and fold standard deviation; test MAE,
  RMSE (both MPa), and R-squared.
- `results/summary.json`: split sizes, seed, parameters, versions, and checksum.
- `results/split_assignments.csv`: original zero-based row index, group, split,
  and CV fold (-1 for test rows).
- `results/rf_cv_search.csv`: all tuning candidates and scores.
- `results/test_predictions.csv`: held-out targets and all three predictions.
- `results/predictions.png` and `.pdf`: measured-versus-predicted plot.
- `models/model.joblib`: the model selected on validation MAE, trained only on
  training rows; `metadata.json` stores schema and training ranges.
- `examples/inputs.csv`: five actual held-out examples; reference targets are
  separately provided in `examples/expected_strengths.csv`.

For a new prediction, make a CSV with the eight named input columns and run
`python src/predict.py --input your_inputs.csv --output your_predictions.csv`.
Units and column names must match the table. The script validates numeric values,
warns about individual features outside training ranges, and outputs
`predicted_strength_mpa`. Being within these ranges does not guarantee that a new
combination is represented in training. Only load trusted joblib files.

## Report and submission

`report/report.tex` is the LaTeX source; `report/report.pdf` is its compiled output.
From the report directory, compile with `pdflatex report.tex` (twice), or upload
the source and `../results/predictions.pdf` to Overleaf, adjusting the figure path
if the folder structure changes.

Before submission, review the interpretation and reflection, confirm the author
and course information, create a GitHub repository, and replace the repository
status in the report with its real URL. No GitHub repository is created by merely
extracting this archive. Upload project files, not your virtual environment.
Compile the report again after inserting the URL and confirm it remains at most
two A4 pages, including references.

The supplied PDF was compiled with Tectonic 0.17.0 (LaTeX). The Python environment
is pinned separately; a TeX installation is needed only to rebuild the report,
not to train or predict.

## Limitations and disclosures

This is an educational empirical predictor, not a replacement for materials
testing or design checks. Data are limited to one published dataset; material
provenance, curing conditions, and between-laboratory effects are not fully
represented. The model has no physical monotonicity constraints or uncertainty
intervals. It has not been validated on independent modern concrete experiments.

Generative AI (OpenAI ChatGPT/Codex) assisted with project design, code generation,
execution, documentation, and the report draft. The student must review and
understand the work and adapt the reflection to their actual learning experience.
No pretrained model is used. Models are trained with scikit-learn; numerical
processing uses NumPy, SciPy and pandas; plots use Matplotlib. No third-party
project code was copied into the training scripts.

Software reference: Pedregosa et al. (2011), Scikit-learn: Machine Learning in
Python, JMLR 12, 2825-2830. https://jmlr.org/papers/v12/pedregosa11a.html.
