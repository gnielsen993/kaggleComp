# Notebook cleanup and next submission implementation plan

> Execute inline using the executing-plans workflow, followed by verification-loop. The user has authorized implementation.

**Goal:** Make the existing Titanic notebook reproducible and explainable, then prepare an evaluated third non-dummy submission.

**Architecture:** Keep the notebook as the report and experiment driver. Small local Python modules hold reusable feature engineering, model pipelines, and submission validation. Preserve historical submission scores separately from newly computed validation scores.

**Tech stack:** Python 3.13, pandas, scikit-learn, matplotlib, Jupyter, unittest.

**Spec:** User request to clean up current work and work on the next submission; `docs/COURSE_REQUIREMENTS.md`.

## Constraints and review focus

- Preserve the existing data, screenshots, reference assets, and historical scores.
- Exclude the pre-existing `train.csv` modification from commits.
- Split before learning preprocessing statistics; all fitting stays inside pipelines.
- Test unseen categories and missing values; feature engineering must not mutate input.
- Select the next candidate using five-fold CV on the development split; the existing holdout is a reused diagnostic, not a fresh unbiased test.
- Validate submission columns, IDs, row count, binary predictions, and absence of missing values.
- Preparing a CSV does not count as a successful Kaggle upload. Do not invent a leaderboard score.

## Task 1: Clean up existing work

- [x] Add focused tests in `tests/test_titanic.py` for deterministic features, training-only imputation, unseen categories, and submission validation; observe failures before implementation.
- [x] Add `titanic_features.py` and `titanic_models.py` exposing `engineer_features`, `make_pipeline`, and `make_submission`.
- [x] Rewrite `titanic.ipynb` with problem/metric, setup, data inspection, split, preprocessing rationale, baseline comparisons, historical Kaggle evidence, and limitations.
- [x] Add pinned requirements and README run instructions.
- [x] Run tests and execute every notebook cell in a fresh kernel; review and commit verified cleanup.

## Task 2: Prepare the next submission

- [ ] Compare a modest gradient boosting experiment with existing baselines using identical development CV folds and features.
- [ ] Record configuration, CV mean/spread, diagnostic holdout accuracy, and selection rationale in the notebook and `docs/EXPERIMENTS.md`.
- [ ] Fit the selected candidate on all labeled rows and write `submission_03_*.csv` locally with validation checks.
- [ ] Reexecute the full notebook, verify CSV contents, audit dependencies, review changes, and commit/push verified work.

## Execution notes

The repo has no GSD roadmap; GSD quick is not applicable. Work proceeds inline under this plan. The user authorized the implementation, so no additional plan approval is needed. Global repo defaults authorize committing and pushing verified changes; no Kaggle upload is performed without a concrete user instruction.

Task 1 validation: four unittest cases passed; all seven code cells executed in a fresh kernel and the HTML report rendered. Reviewed the EDA plot. Development CV: dummy 0.6166, logistic regression 0.8216, random forest 0.8329. Updated bundled pip after the dependency audit flagged it.
