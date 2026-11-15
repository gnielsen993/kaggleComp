import unittest

import numpy as np
from pandas.testing import assert_frame_equal
from sklearn.linear_model import LogisticRegression

from test_titanic import passengers
from titanic_features import engineer_extended_features
from titanic_inspection import permute_columns_together
from titanic_models import FEATURE_SETS, make_pipeline


class FeatureComparisonTests(unittest.TestCase):
    def test_cabin_and_fare_features_are_row_local(self):
        raw = passengers()
        raw['Cabin'] = [' C23 C25 C27 ', None, '  ', 't1']
        raw.loc[2, 'Fare'] = np.nan
        original = raw.copy(deep=True)
        result = engineer_extended_features(raw)
        assert_frame_equal(raw, original)
        self.assertEqual(result.Deck.tolist(), ['C', 'Unknown', 'Unknown', 'T'])
        self.assertEqual(result.HasCabin.tolist(), [1, 0, 0, 1])
        np.testing.assert_allclose(result.FarePerPerson, [7, 20, np.nan, 9])
        assert_frame_equal(result.iloc[[1]], engineer_extended_features(raw.iloc[[1]]))

    def test_each_set_uses_only_declared_features_and_training_statistics(self):
        raw = passengers()
        raw['Cabin'] = ['C1', None, '', 'D4']
        for name, (numeric, categorical) in FEATURE_SETS.items():
            with self.subTest(name=name):
                model = make_pipeline(LogisticRegression(max_iter=1000), feature_set=name)
                model.fit(raw, [0, 1, 0, 1])
                pre = model.named_steps['preprocess']
                self.assertEqual(pre.transformers_[0][2], list(numeric))
                self.assertEqual(pre.transformers_[1][2], list(categorical))
                imp = pre.named_transformers_['numeric'].named_steps['impute']
                before = imp.statistics_.copy()
                validation = raw.iloc[[0, 1]].copy()
                validation['Cabin'] = ['Z99', None]
                validation['Age'] = [9999, np.nan]
                self.assertEqual(len(model.predict(validation)), 2)
                np.testing.assert_array_equal(before, imp.statistics_)
                self.assertFalse({'Survived', 'PassengerId', 'Name', 'Ticket'} & set(numeric + categorical))
        with self.assertRaises(ValueError):
            make_pipeline(LogisticRegression(), feature_set='typo')

    def test_group_permutation_preserves_pairs_index_and_other_columns(self):
        raw = passengers().set_index('PassengerId')
        original = raw.copy(deep=True)
        result = permute_columns_together(raw, ['SibSp', 'Parch'], np.array([2, 0, 3, 1]))
        assert_frame_equal(raw, original)
        self.assertEqual(result.index.tolist(), raw.index.tolist())
        np.testing.assert_array_equal(result[['SibSp', 'Parch']], raw.iloc[[2, 0, 3, 1]][['SibSp', 'Parch']])
        assert_frame_equal(result.drop(columns=['SibSp', 'Parch']), raw.drop(columns=['SibSp', 'Parch']))


if __name__ == '__main__':
    unittest.main()
