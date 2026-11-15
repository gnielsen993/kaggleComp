"""Training-only preprocessing and validated Titanic submission construction."""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from titanic_features import engineer_extended_features, engineer_features


# Predeclared input columns before one-hot encoding; order also breaks final ties.
FEATURE_SETS = {
    "basic": (("Pclass", "Age", "SibSp", "Parch", "Fare"), ("Sex", "Embarked")),
    "compact": (("Pclass", "Age", "Fare", "FamilySize", "IsAlone"), ("Sex", "Embarked", "Title")),
    "baseline": (("Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone"), ("Sex", "Embarked", "Title")),
    "extended": (("Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "HasCabin", "FarePerPerson"),
                 ("Sex", "Embarked", "Title", "Deck")),
}


def make_pipeline(estimator, *, feature_set="baseline"):
    """Wrap an unfitted estimator; fitting learns statistics only from supplied rows."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set!r}")
    numeric, categorical = (list(columns) for columns in FEATURE_SETS[feature_set])
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
        ("features", FunctionTransformer(
            engineer_extended_features if feature_set == "extended" else engineer_features,
            validate=False,
        )),
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
