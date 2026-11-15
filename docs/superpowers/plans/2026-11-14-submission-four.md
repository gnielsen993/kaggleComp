# Submission four implementation plan

> Execute inline using the existing executing-plans workflow and verification-loop. User request: help through submission 4.

**Goal:** Test whether a small random forest parameter search improves local validation, then prepare a checked CSV for Gabe to upload.

**Architecture:** Extend the existing notebook with one experiment. Reuse the existing feature/pipeline/export helpers unchanged. Keep the original three upload records intact.

**Tech stack:** Existing pinned Python/scikit-learn/Jupyter environment.

**Spec:** User request and `docs/COURSE_REQUIREMENTS.md`.

## Constraints and review focus

- Preserve the user's uncommitted `train.csv` change; stage only experiment files.
- Nine configurations defined before execution: depth [5, 8, None], minimum leaf size [5, 3, 1]. Keep 200 trees, seed 42, and all other parameters/features/preprocessing fixed.
- Include the current depth-5/leaf-1 control and assert its CV score matches the existing baseline.
- Use the same five development folds; select the highest mean CV accuracy. Exact ties use the first configuration: shallower depth, then larger minimum leaf size.
- Inspect the reused holdout only after selecting; do not expand the search after seeing results.
- Preserve historical submission scores; submission 4 stays unsubmitted with unknown Kaggle score.
- Validate exact schema, 418 predictions, test ID order, binary values, CSV round trip, and recorded hash.

## Task: evaluate and prepare submission 4

- [x] Add the hypothesis, fixed search, all configuration results, selection rationale, and export cells to `titanic.ipynb`.
- [x] Execute in a fresh kernel; record measured results in `docs/EXPERIMENTS.md` and link the new artifact in README.
- [x] Verify export integrity, notebook execution, existing tests, dependency audit, and review the changes.
- [x] Commit and push verified work; guide Gabe to upload `submission_04_random_forest_tuned.csv` and return its score.

## Decisions

This is a controlled tuning experiment, not a new feature-engineering experiment. A CV improvement is exploratory and does not guarantee a Kaggle improvement. If the control wins, report that rather than forcing a different configuration.

## Results and verification

The existing depth-5/leaf-1 control won: mean CV 0.8328868315, fold standard deviation 0.0231187689, reused holdout 0.8324022346. All five selected fold scores equal the control. The unlimited-depth/leaf-1 model fit training better but scored worse on CV. No search expansion was made.

Decision: export the selected cleaned forest without forcing new settings. The 418 predictions equal `submission_forest_clean.csv`; the historical uploaded forest CSV is unavailable, and its preprocessing differed, so its leaderboard score cannot be assigned to this output. The experiment is documented as no tuning improvement and not yet uploaded. Cost: the uploaded score may not improve, and the search does not establish a novel prediction set relative to the unavailable historical file.

Verification: all 13 code cells executed in a fresh kernel. Four unit tests passed; dependency audit found no known vulnerabilities. An independent fit from the recorded parameters reproduced all 418 predictions. Checked file hash, schema, ID order, binary/nonmissing values, nine-search completeness, winning mean CV, control fold equality, preserved submission 3 score, notebook schema, and documentation links. Rendered the report after adding the measured conclusion.

Independent read-only review: approved with no Critical, Important, or Minor findings. Submission 4 remains prepared for the user to upload; no Kaggle action was taken.
