from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    ID_COLUMN,
    RANDOM_STATE,
    SAMPLE_SUBMISSION_PATH,
    TARGET_COLUMN,
    TEST_PATH,
    TRAIN_PATH,
    VALIDATION_SIZE,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {path}"
        )

    return pd.read_csv(path)


def validate_competition_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    sample_submission_df: pd.DataFrame,
) -> None:

    if ID_COLUMN not in train_df.columns:
        raise ValueError(
            f"'{ID_COLUMN}' is missing from train.csv"
        )

    if ID_COLUMN not in test_df.columns:
        raise ValueError(
            f"'{ID_COLUMN}' is missing from test.csv"
        )

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' is missing from train.csv"
        )

    if TARGET_COLUMN in test_df.columns:
        raise ValueError(
            f"'{TARGET_COLUMN}' must not exist in test.csv"
        )

    if train_df[TARGET_COLUMN].isna().any():
        raise ValueError(
            f"'{TARGET_COLUMN}' contains missing values."
        )

    unique_targets = set(
        train_df[TARGET_COLUMN]
        .dropna()
        .unique()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "Target must contain only binary values "
            f"{{0, 1}}. Found: {sorted(unique_targets)}"
        )

    train_features = set(
        train_df.columns
    ) - {TARGET_COLUMN}

    test_features = set(
        test_df.columns
    )

    if train_features != test_features:
        raise ValueError(
            "train.csv and test.csv feature columns do not match.\n"
            f"Missing from test: {train_features - test_features}\n"
            f"Extra in test: {test_features - train_features}"
        )

    expected_submission_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    if not expected_submission_columns.issubset(
        sample_submission_df.columns
    ):
        raise ValueError(
            "sample_submission.csv must contain "
            f"'{ID_COLUMN}' and '{TARGET_COLUMN}'."
        )

    if len(test_df) != len(sample_submission_df):
        raise ValueError(
            "test.csv and sample_submission.csv "
            "must contain the same number of rows."
        )

    if not test_df[ID_COLUMN].reset_index(
        drop=True
    ).equals(
        sample_submission_df[
            ID_COLUMN
        ].reset_index(drop=True)
    ):
        raise ValueError(
            "IDs in test.csv do not match "
            "sample_submission.csv."
        )

    if train_df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate IDs found in train.csv."
        )

    if test_df[ID_COLUMN].duplicated().any():
        raise ValueError(
            "Duplicate IDs found in test.csv."
        )


def load_competition_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    train_df = read_csv(
        TRAIN_PATH
    )

    test_df = read_csv(
        TEST_PATH
    )

    sample_submission_df = read_csv(
        SAMPLE_SUBMISSION_PATH
    )

    validate_competition_data(
        train_df=train_df,
        test_df=test_df,
        sample_submission_df=sample_submission_df,
    )

    return (
        train_df,
        test_df,
        sample_submission_df,
    )


def create_train_validation_split(
    train_df: pd.DataFrame,
    validation_size: float = VALIDATION_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    train_split, validation_split = (
        train_test_split(
            train_df,
            test_size=validation_size,
            random_state=random_state,
            shuffle=True,
            stratify=train_df[
                TARGET_COLUMN
            ],
        )
    )

    return (
        train_split.reset_index(
            drop=True
        ),
        validation_split.reset_index(
            drop=True
        ),
    )