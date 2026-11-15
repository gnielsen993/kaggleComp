import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from titanic_features import engineer_features
from titanic_models import make_pipeline, make_submission


def passengers():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4],
        "Pclass": [3, 1, 2, 3],
        "Name": ["A, Mr. One", "B, Mrs. Two", "C, Dr. Three", None],
        "Sex": ["male", "female", "male", "female"],
        "Age": [20.0, 40.0, np.nan, 30.0],
        "SibSp": [0, 1, 0, 0],
        "Parch": [0, 2, 1, 0],
        "Fare": [7.0, 80.0, 15.0, 9.0],
        "Embarked": ["S", "C", "S", np.nan],
    })


class FeatureTests(unittest.TestCase):
    def test_features_are_rowwise_and_do_not_mutate_input(self):
        raw = passengers()
        original = raw.copy(deep=True)
        actual = engineer_features(raw)
        assert_frame_equal(raw, original)
        self.assertEqual(actual["FamilySize"].tolist(), [1, 4, 2, 1])
        self.assertEqual(actual["IsAlone"].tolist(), [1, 0, 0, 1])
        self.assertEqual(actual["Title"].tolist(), ["Mr", "Mrs", "Other", "Other"])
        assert_frame_equal(actual.iloc[[1]], engineer_features(raw.iloc[[1]]))

    def test_imputation_uses_only_fitted_rows_and_handles_unknown_categories(self):
        raw = passengers()
        model = make_pipeline(LogisticRegression(max_iter=1000))
        model.fit(raw, [0, 1, 0, 1])
        imputer = model.named_steps["preprocess"].named_transformers_["numeric"].named_steps["impute"]
        names = model.named_steps["preprocess"].transformers_[0][2]
        self.assertEqual(imputer.statistics_[names.index("Age")], 30.0)
        before = imputer.statistics_.copy()
        validation = raw.iloc[[0, 1]].copy()
        validation["Age"] = [10000.0, np.nan]
        validation["Fare"] = np.nan
        validation["Embarked"] = ["UNSEEN", np.nan]
        predicted = model.predict(validation)
        self.assertEqual(len(predicted), 2)
        np.testing.assert_array_equal(imputer.statistics_, before)
        feature_names = model.named_steps["preprocess"].get_feature_names_out()
        self.assertFalse(any("PassengerId" in name or "Survived" in name for name in feature_names))


class SubmissionTests(unittest.TestCase):
    def test_valid_submission_preserves_passenger_order(self):
        result = make_submission(pd.Series([894, 892, 893]), np.array([1, 0, 1]))
        self.assertEqual(result.columns.tolist(), ["PassengerId", "Survived"])
        self.assertEqual(result["PassengerId"].tolist(), [894, 892, 893])
        self.assertEqual(result["Survived"].tolist(), [1, 0, 1])

    def test_rejects_malformed_submissions(self):
        cases = [([1, 2], [1]), ([1, 1], [0, 1]), ([1, None], [0, 1]),
                 ([1, 2], [0, np.nan]), ([1, 2], [0, 0.7]),
                 ([1, 2], [0, 2]), ([], [])]
        for ids, predictions in cases:
            with self.subTest(ids=ids, predictions=predictions):
                with self.assertRaises(ValueError):
                    make_submission(ids, predictions)


if __name__ == "__main__":
    unittest.main()
