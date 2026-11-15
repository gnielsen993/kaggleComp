"""Training-only preprocessing and validated Titanic submission construction."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from titanic_features import engineer_features


def make_pipeline(estimator):
    """Wrap an unfitted estimator; fitting learns statistics only from supplied rows."""
    numeric = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"]
    categorical = ["Sex", "Embarked", "Title"]
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([
        ("features", FunctionTransformer(engineer_features, validate=False)),
        ("preprocess", preprocessing),
        ("model", clone(estimator)),
    ])


def make_submission(passenger_ids, predictions):
    """Validate predictions before export, retaining the caller's passenger order."""
    ids = np.asarray(passenger_ids)
    values = np.asarray(predictions)
    if ids.ndim != 1 or values.ndim != 1 or not len(ids) or len(ids) != len(values):
        raise ValueError("IDs and predictions must be nonempty, equal-length vectors.")
    if pd.isna(ids).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("Passenger IDs must be present and unique.")
    if pd.isna(values).any() or not np.isin(values, [0, 1]).all():
        raise ValueError("Predictions must contain only 0 and 1, with no missing values.")
    return pd.DataFrame({"PassengerId": ids, "Survived": values.astype(int)})
