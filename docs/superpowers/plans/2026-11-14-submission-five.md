# Submission five implementation plan

> Execute inline with the existing executing-plans workflow and verification-loop. User request: help with submission 5.

**Goal:** Demonstrate logistic regression regularization and classification evaluation, then prepare a checked fifth submission.

**Architecture:** Add one experiment section to the notebook using existing feature, pipeline, and export helpers. Keep all earlier experiment records and scores intact.

**Tech stack:** Existing pinned Python/scikit-learn/Jupyter environment.

**Spec:** `docs/COURSE_REQUIREMENTS.md`, template sections 7/12, and CS4320 HW5 classification-metrics exercises. Unlike the churn homework, select by accuracy for Titanic.

## Constraints and review focus

- Preserve the existing uncommitted `train.csv` change and all historical scores.
- Predeclare ten configurations: C [0.01, 0.1, 1, 10, 100] and class_weight [None, balanced]. Keep default L2 regularization, lbfgs, max_iter=1000, features, and seed-42 development folds unchanged.
- Select maximum mean development CV accuracy. Exact ties favor first grid entry: smaller C (stronger regularization), then unweighted classes.
- Include C=1/class_weight=None control; assert its fold accuracies match the original cleaned logistic baseline.
- Report accuracy, precision, recall, F1, ROC-AUC, average precision, train accuracy, and fit/score times. Define Survived=1 as positive. Use probabilities for ranking metrics and curves, class predictions for label metrics, and zero_division=0 for undefined precision/F1.
- Compare existing baselines using the same folds; after selection report diagnostics on the already-used holdout. No threshold tuning, holdout selection, or search expansion.
- Explain class-weight and regularization tradeoffs with measured results, plus confusion matrices and ROC/PR plots. Average precision and F1 must not be confused with ROC-AUC.
- Export `submission_05_logistic_regularized.csv`; validate 418 rows, schema, exact test ID order, binary values, round trip, and hash. Preserve all search records in generated metadata.
- Preparation does not count as upload. Keep Kaggle score unknown and confirmed count at four until scoring evidence arrives.

## Tasks

- [x] Add search, multimetric comparison, diagnostic plots, explanation, and export cells to `titanic.ipynb`.
- [x] Execute in a fresh kernel and record the actual result in `docs/EXPERIMENTS.md` and README.
- [x] Independently reproduce predictions; check artifacts, existing tests, dependency audit, report rendering, and review.
- [x] Prepare verified work for commit/push and document Gabe's upload instructions.

## Execution and verification record

- Implemented ten-setting search, six classification metrics, model comparisons, timing, confusion matrices, ROC/PR curves, conclusions, and checked export.
- Selected C=10/class_weight=None; C=100 tied on accuracy. CV accuracy 0.823057 versus control 0.821649. Only one test label changed; no performance improvement or completed upload is claimed.
- Fresh-kernel execution through `scripts/run_notebook.py`: all 19 code cells passed. The final prose-only conclusions were added afterward and HTML regenerated; code and saved outputs were preserved.
- `python -m unittest discover -s tests -v`: 4 tests passed. Dependency audit: no known vulnerabilities.
- Independent full-data refit reproduced every exported label and SHA-256; selected and control CV folds reproduced; holdout confusion counts and all six metrics checked independently. Prior submission scores and hashes remained intact.
- Notebook schema, syntax, execution order, and absence of error outputs checked; all three new plot outputs visually inspected. Earlier code cells remain unchanged.
- Independent final reviewer found no substantive issues and independently reproduced all 418 labels and checked selection, metrics, schema, and hashes.
- Scoped diff whitespace check passed. The unrelated preexisting train.csv line-ending change is excluded from this commit.
- Delivery: commit/push follows this review record; remote confirmation is reported in the final response. Kaggle upload remains Gabe's next step.
