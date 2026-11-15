# Submission 6: controlled feature comparison

Goal: compare predefined passenger feature sets with the existing forest, explain feature reliance, and prepare a sixth non-dummy upload. Five uploads are confirmed; preparation does not count as a sixth.

## Protocol fixed before execution

- Same seed-42 stratified 712/179 split and five development folds. Preprocessing fits inside each training fold.
- Forest unchanged: 200 trees, depth 5, minimum leaf size 1, seed 42, other defaults unchanged. No further hyperparameter search.
- Four feature sets: basic (Pclass, Age, SibSp, Parch, Fare, Sex, Embarked); compact (Pclass, Age, Fare, FamilySize, IsAlone, Sex, Embarked, Title); baseline (the original ten fields); extended (baseline plus HasCabin, Deck, FarePerPerson).
- Deck is the first cabin letter, or Unknown. HasCabin indicates a nonempty cabin string. FarePerPerson = Fare / FamilySize, a proxy rather than a known individual fare because tickets and families need not coincide. All new features are row-local and label-free.
- Select highest mean development accuracy; exact ties favor fewer input features, then listed order. Report fold scores, training accuracy, precision, recall, F1, ROC-AUC, average precision, and timing. Assert baseline reproduces the earlier forest folds.
- Explain the selected model using grouped permutation accuracy drops on validation folds only, 10 repeats per group per fold, seed 42. Shuffle Sex/Name together, SibSp/Parch together, and Age, Pclass, Fare, Embarked, Cabin individually, then recompute derived features. Exclude Cabin for sets that do not use it. Report spread across permutations/folds, not confidence intervals. Correlation across groups and selection bias limit interpretation; no causal claims or subsequent selection based on importance.
- Inspect reused holdout only after selection, then refit all 891 labeled rows. Export submission_06_feature_comparison.csv and outputs/experiment_06.json, verify 418 ordered binary predictions and hash. Preserve all previous hashes/scores and train.csv's unrelated local edit.

## Delivery checks

- Test row-local cabin/fare features, missing/empty/unseen cabins, original input preservation, feature-set isolation, and grouped permutation preserving within-group relationships.
- Fresh-kernel full notebook run; independent reproduction of selected predictions; schema/hash checks; dependency audit and diff review.
- Document observed results and limits, commit/push, and give Gabe upload instructions. Submission 6's score stays unknown until evidence arrives.

## Observed result and explicit delivery adjustment

The overall rule selected baseline (0.832887), reproducing submission 4 exactly. Compact (0.824436) is the strongest alternative, ahead of extended (0.821698) and basic (0.817394). To provide a distinct sixth experiment, prepare compact as an ablation upload and retain baseline as the overall winner. This delivery decision follows the observed comparison and is disclosed in the report; it does not redefine the original overall-winner rule. No configurations are added and no holdout or leaderboard result chooses the alternative. Permutation importance explains the baseline winner; final holdout diagnostics and CSV use the compact candidate. Metadata distinguishes these roles.

## Verification record

- New feature/inspection tests first failed because the new helpers did not exist; implemented helpers and all seven tests passed.
- Fresh-kernel notebook run passed all 24 code cells, with HTML generated. Final measured conclusions were added as prose only afterward; code and outputs were preserved and HTML refreshed.
- Independent refit reproduced all 418 compact predictions and CSV hash, selected-candidate CV folds, and holdout confusion counts. Independently reconstructed one grouped permutation and checked every importance summary against its 50 measurements.
- Earlier submission 3, 4, and 5 score associations and exact CSV hashes remained unchanged. The original forest control reproduced all earlier fold accuracies.
- New plots visually checked. Dependency audit found no known vulnerabilities. Code, report text, and staged diff reviewed; unrelated train.csv edit excluded.
- Prepared CSV: 162 predicted survivors, two labels changed from submission 4, SHA-256 5a9d368108b4deb45559e59ead25c4d88d2bd681764516efd9e29958459cbcbb. Upload/scoring remains pending.
