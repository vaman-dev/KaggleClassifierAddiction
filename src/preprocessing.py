from __future__ import annotations

from typing import Literal

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.config import (
    ID_COLUMN,
    TARGET_COLUMN,
)


XGBoostMissingStrategy = Literal[
    "median_indicator",
    "median",
    "native",
]


# =========================================================
# Train data separation
# =========================================================

def separate_train_data(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise ValueError(
            "Training dataframe is missing "
            f"required columns: {missing_columns}"
        )

    ids = frame[
        ID_COLUMN
    ].copy()

    y = frame[
        TARGET_COLUMN
    ].copy()

    X = frame.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    ).copy()

    return (
        X,
        y,
        ids,
    )


# =========================================================
# Competition test separation
# =========================================================

def separate_test_data(
    frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.Series,
]:

    if ID_COLUMN not in frame.columns:
        raise ValueError(
            f"Test dataframe must contain "
            f"'{ID_COLUMN}'."
        )

    ids = frame[
        ID_COLUMN
    ].copy()

    X = frame.drop(
        columns=[
            ID_COLUMN,
        ]
    ).copy()

    return (
        X,
        ids,
    )


# =========================================================
# Feature type detection
# =========================================================

def detect_feature_types(
    X: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:

    numerical_columns = (
        X
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        X
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    return (
        numerical_columns,
        categorical_columns,
    )


# =========================================================
# Numerical preprocessing
# =========================================================

def create_median_indicator_numeric_pipeline() -> Pipeline:
    """
    Primary strategy.

    NaN
      ↓
    median replacement
      +
    missing indicator
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )


def create_median_numeric_pipeline() -> Pipeline:
    """
    Comparison strategy.

    NaN
      ↓
    median replacement

    No missing indicator is generated.
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=False,
                ),
            ),
        ]
    )


# =========================================================
# Categorical preprocessing
# =========================================================

def create_categorical_pipeline() -> Pipeline:
    """
    Preserve categorical missingness explicitly.

    NaN
      ↓
    __MISSING__
      ↓
    One-hot encoding
    """

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="__MISSING__",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )


# =========================================================
# Generic XGBoost preprocessor
# =========================================================

def create_xgboost_preprocessor(
    X: pd.DataFrame,
    strategy: XGBoostMissingStrategy = "median_indicator",
) -> ColumnTransformer:
    """
    Create preprocessing for an XGBoost experiment.

    Supported strategies:

    median_indicator
        Numerical NaN -> median + missing indicator

    median
        Numerical NaN -> median only

    native
        Numerical NaN stays as NaN and is handled
        directly by XGBoost.

    Categorical missing values are always converted
    to an explicit __MISSING__ category and one-hot encoded.
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_types(
        X
    )

    transformers = []

    if numerical_columns:

        if strategy == "median_indicator":

            numerical_transformer = (
                create_median_indicator_numeric_pipeline()
            )

        elif strategy == "median":

            numerical_transformer = (
                create_median_numeric_pipeline()
            )

        elif strategy == "native":

            numerical_transformer = "passthrough"

        else:

            raise ValueError(
                "Unknown XGBoost missing-value strategy: "
                f"{strategy!r}"
            )

        transformers.append(
            (
                "numerical",
                numerical_transformer,
                numerical_columns,
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                create_categorical_pipeline(),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable model features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


# =========================================================
# Named XGBoost experiment preprocessors
# =========================================================

def create_xgboost_median_indicator_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    return create_xgboost_preprocessor(
        X,
        strategy="median_indicator",
    )


def create_xgboost_median_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    return create_xgboost_preprocessor(
        X,
        strategy="median",
    )


def create_xgboost_native_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:

    return create_xgboost_preprocessor(
        X,
        strategy="native",
    )


# =========================================================
# Random Forest benchmark preprocessing
# =========================================================

def create_random_forest_preprocessor(
    X: pd.DataFrame,
) -> ColumnTransformer:
    """
    Random Forest benchmark uses the same missingness
    strategy selected from EDA:

    numerical:
        median + missing indicator

    categorical:
        explicit missing category + one-hot encoding
    """

    (
        numerical_columns,
        categorical_columns,
    ) = detect_feature_types(
        X
    )

    transformers = []

    if numerical_columns:

        transformers.append(
            (
                "numerical",
                create_median_indicator_numeric_pipeline(),
                numerical_columns,
            )
        )

    if categorical_columns:

        transformers.append(
            (
                "categorical",
                create_categorical_pipeline(),
                categorical_columns,
            )
        )

    if not transformers:
        raise ValueError(
            "No usable model features were found."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )