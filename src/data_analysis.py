from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.config import (
    ID_COLUMN,
    TARGET_COLUMN,
)


def print_section(
    title: str,
    width: int = 70,
) -> None:

    print()
    print("=" * width)
    print(title)
    print("=" * width)


def get_feature_columns(
    frame: pd.DataFrame,
) -> list[str]:

    return [
        column
        for column in frame.columns
        if column not in {
            ID_COLUMN,
            TARGET_COLUMN,
        }
    ]


# =========================================================
# General dataset analysis
# =========================================================

def analyse_shapes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    print_section(
        "DATASET SHAPES"
    )

    print(
        "Train:",
        train_df.shape,
    )

    print(
        "Test:",
        test_df.shape,
    )


def analyse_columns(
    train_df: pd.DataFrame,
) -> None:

    print_section(
        "COLUMNS"
    )

    for index, column in enumerate(
        train_df.columns,
        start=1,
    ):
        print(
            f"{index}. {column}"
        )


def analyse_dtypes(
    train_df: pd.DataFrame,
) -> None:

    print_section(
        "DATA TYPES"
    )

    print(
        train_df.dtypes
    )


def analyse_missing_values(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    missing_report = pd.DataFrame(
        {
            "missing_count":
                train_df.isna().sum(),

            "missing_percentage":
                train_df.isna().mean()
                * 100,
        }
    )

    missing_report = (
        missing_report[
            missing_report[
                "missing_count"
            ] > 0
        ]
        .sort_values(
            by="missing_percentage",
            ascending=False,
        )
    )

    print_section(
        "MISSING VALUES"
    )

    if missing_report.empty:
        print(
            "No missing values found."
        )
    else:
        print(
            missing_report
        )

    return missing_report


def analyse_duplicates(
    train_df: pd.DataFrame,
) -> None:

    print_section(
        "DUPLICATES"
    )

    duplicate_count = (
        train_df
        .drop(
            columns=[ID_COLUMN],
            errors="ignore",
        )
        .duplicated()
        .sum()
    )

    print(
        "Duplicate feature rows:",
        duplicate_count,
    )


def analyse_target(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    target_counts = (
        train_df[
            TARGET_COLUMN
        ]
        .value_counts()
        .sort_index()
    )

    target_percentage = (
        train_df[
            TARGET_COLUMN
        ]
        .value_counts(
            normalize=True
        )
        .sort_index()
        * 100
    )

    report = pd.DataFrame(
        {
            "count":
                target_counts,

            "percentage":
                target_percentage,
        }
    )

    print_section(
        "TARGET DISTRIBUTION"
    )

    print(
        report
    )

    return report


def analyse_feature_types(
    train_df: pd.DataFrame,
) -> tuple[
    list[str],
    list[str],
]:

    feature_df = (
        train_df[
            get_feature_columns(
                train_df
            )
        ]
    )

    numerical_columns = (
        feature_df
        .select_dtypes(
            include="number"
        )
        .columns
        .tolist()
    )

    categorical_columns = (
        feature_df
        .select_dtypes(
            exclude="number"
        )
        .columns
        .tolist()
    )

    print_section(
        "FEATURE TYPES"
    )

    print(
        f"Numerical ({len(numerical_columns)}):"
    )

    for column in numerical_columns:
        print(
            f" - {column}"
        )

    print(
        f"\nCategorical ({len(categorical_columns)}):"
    )

    for column in categorical_columns:
        print(
            f" - {column}"
        )

    return (
        numerical_columns,
        categorical_columns,
    )


def analyse_numerical_features(
    train_df: pd.DataFrame,
    numerical_columns: list[str],
) -> None:

    print_section(
        "NUMERICAL FEATURE SUMMARY"
    )

    if not numerical_columns:
        print(
            "No numerical features."
        )
        return

    print(
        train_df[
            numerical_columns
        ]
        .describe()
        .T
    )


def analyse_categorical_features(
    train_df: pd.DataFrame,
    categorical_columns: list[str],
) -> None:

    print_section(
        "CATEGORICAL FEATURES"
    )

    if not categorical_columns:
        print(
            "No categorical features."
        )
        return

    for column in categorical_columns:

        print(
            f"\n{column}"
        )

        print(
            train_df[
                column
            ]
            .value_counts(
                dropna=False
            )
        )


def detect_numeric_like_object_columns(
    train_df: pd.DataFrame,
) -> None:

    print_section(
        "POSSIBLE NUMERIC-LIKE OBJECT COLUMNS"
    )

    object_columns = (
        train_df
        .select_dtypes(
            include=[
                "object",
                "string",
            ]
        )
        .columns
    )

    suspicious = []

    for column in object_columns:

        converted = (
            pd.to_numeric(
                train_df[
                    column
                ],
                errors="coerce",
            )
        )

        ratio = (
            converted
            .notna()
            .mean()
        )

        if ratio >= 0.50:
            suspicious.append(
                (
                    column,
                    ratio,
                )
            )

    if not suspicious:
        print(
            "No suspicious numeric-like "
            "object columns detected."
        )
        return

    for column, ratio in suspicious:
        print(
            f"{column}: {ratio:.2%} "
            "numeric-like"
        )


# =========================================================
# Missingness analysis
# =========================================================

def analyse_missingness_vs_target(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    for column in get_feature_columns(
        train_df
    ):

        missing_mask = (
            train_df[
                column
            ].isna()
        )

        missing_count = int(
            missing_mask.sum()
        )

        if missing_count == 0:
            continue

        present_mask = (
            ~missing_mask
        )

        missing_rate = (
            train_df.loc[
                missing_mask,
                TARGET_COLUMN,
            ]
            .mean()
        )

        present_rate = (
            train_df.loc[
                present_mask,
                TARGET_COLUMN,
            ]
            .mean()
        )

        difference = (
            missing_rate
            - present_rate
        )

        rows.append(
            {
                "feature":
                    column,

                "missing_count":
                    missing_count,

                "missing_percentage":
                    missing_mask.mean()
                    * 100,

                "present_count":
                    int(
                        present_mask.sum()
                    ),

                "addiction_rate_present":
                    present_rate * 100,

                "addiction_rate_missing":
                    missing_rate * 100,

                "rate_difference_pp":
                    difference * 100,

                "absolute_difference_pp":
                    abs(difference) * 100,

                "missing_lift":
                    (
                        missing_rate
                        / present_rate
                        if present_rate > 0
                        else float("nan")
                    ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    result = (
        result
        .sort_values(
            by="absolute_difference_pp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    percentage_columns = [
        "missing_percentage",
        "addiction_rate_present",
        "addiction_rate_missing",
        "rate_difference_pp",
        "absolute_difference_pp",
        "missing_lift",
    ]

    result[
        percentage_columns
    ] = (
        result[
            percentage_columns
        ]
        .round(4)
    )

    return result


def analyse_missing_count_vs_target(
    train_df: pd.DataFrame,
) -> pd.DataFrame:

    feature_columns = (
        get_feature_columns(
            train_df
        )
    )

    missing_count = (
        train_df[
            feature_columns
        ]
        .isna()
        .sum(axis=1)
    )

    analysis_df = pd.DataFrame(
        {
            "missing_feature_count":
                missing_count,

            TARGET_COLUMN:
                train_df[
                    TARGET_COLUMN
                ],
        }
    )

    report = (
        analysis_df
        .groupby(
            "missing_feature_count"
        )[TARGET_COLUMN]
        .agg(
            row_count="count",
            addiction_rate="mean",
        )
        .reset_index()
    )

    report[
        "addiction_rate"
    ] = (
        report[
            "addiction_rate"
        ]
        * 100
    ).round(4)

    return report


def analyse_pairwise_missingness(
    train_df: pd.DataFrame,
    min_support: int = 500,
) -> pd.DataFrame:

    feature_columns = [
        column
        for column in get_feature_columns(
            train_df
        )
        if train_df[
            column
        ].isna().any()
    ]

    overall_rate = (
        train_df[
            TARGET_COLUMN
        ].mean()
    )

    target = (
        train_df[
            TARGET_COLUMN
        ]
    )

    results = []

    for feature_a, feature_b in combinations(
        feature_columns,
        2,
    ):

        missing_a = (
            train_df[
                feature_a
            ].isna()
        )

        missing_b = (
            train_df[
                feature_b
            ].isna()
        )

        masks = {
            "both_present":
                ~missing_a & ~missing_b,

            "a_missing":
                missing_a & ~missing_b,

            "b_missing":
                ~missing_a & missing_b,

            "both_missing":
                missing_a & missing_b,
        }

        stats = {}

        valid_pair = True

        for name, mask in masks.items():

            count = int(
                mask.sum()
            )

            if count < min_support:
                valid_pair = False
                break

            stats[
                name
            ] = {
                "count":
                    count,

                "rate":
                    target[
                        mask
                    ].mean(),
            }

        if not valid_pair:
            continue

        present_rate = (
            stats[
                "both_present"
            ]["rate"]
        )

        a_rate = (
            stats[
                "a_missing"
            ]["rate"]
        )

        b_rate = (
            stats[
                "b_missing"
            ]["rate"]
        )

        both_rate = (
            stats[
                "both_missing"
            ]["rate"]
        )

        interaction_effect = (
            both_rate
            - a_rate
            - b_rate
            + present_rate
        )

        results.append(
            {
                "feature_a":
                    feature_a,

                "feature_b":
                    feature_b,

                "both_present_count":
                    stats[
                        "both_present"
                    ]["count"],

                "a_missing_only_count":
                    stats[
                        "a_missing"
                    ]["count"],

                "b_missing_only_count":
                    stats[
                        "b_missing"
                    ]["count"],

                "both_missing_count":
                    stats[
                        "both_missing"
                    ]["count"],

                "both_present_rate":
                    present_rate * 100,

                "a_missing_only_rate":
                    a_rate * 100,

                "b_missing_only_rate":
                    b_rate * 100,

                "both_missing_rate":
                    both_rate * 100,

                "difference_vs_overall_pp":
                    (
                        both_rate
                        - overall_rate
                    ) * 100,

                "difference_vs_present_pp":
                    (
                        both_rate
                        - present_rate
                    ) * 100,

                "interaction_effect_pp":
                    interaction_effect * 100,
            }
        )

    result = pd.DataFrame(
        results
    )

    if result.empty:
        return result

    result[
        "absolute_interaction_pp"
    ] = (
        result[
            "interaction_effect_pp"
        ].abs()
    )

    result = (
        result
        .sort_values(
            by="absolute_interaction_pp",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )

    rate_columns = [
        "both_present_rate",
        "a_missing_only_rate",
        "b_missing_only_rate",
        "both_missing_rate",
        "difference_vs_overall_pp",
        "difference_vs_present_pp",
        "interaction_effect_pp",
        "absolute_interaction_pp",
    ]

    result[
        rate_columns
    ] = (
        result[
            rate_columns
        ].round(4)
    )

    return result


# =========================================================
# High-level analysis runners
# =========================================================

def run_data_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    analyse_shapes(
        train_df,
        test_df,
    )

    analyse_columns(
        train_df
    )

    analyse_dtypes(
        train_df
    )

    analyse_missing_values(
        train_df
    )

    analyse_duplicates(
        train_df
    )

    analyse_target(
        train_df
    )

    (
        numerical_columns,
        categorical_columns,
    ) = analyse_feature_types(
        train_df
    )

    analyse_numerical_features(
        train_df,
        numerical_columns,
    )

    analyse_categorical_features(
        train_df,
        categorical_columns,
    )

    detect_numeric_like_object_columns(
        train_df
    )


def run_missingness_analysis(
    train_df: pd.DataFrame,
    *,
    min_pairwise_support: int,
    top_pairs: int = 30,
) -> None:

    print_section(
        "INDIVIDUAL MISSINGNESS VS ADDICTION",
        width=100,
    )

    individual_report = (
        analyse_missingness_vs_target(
            train_df
        )
    )

    print(
        individual_report.to_string(
            index=False
        )
        if not individual_report.empty
        else "No missing values found."
    )

    print_section(
        "NUMBER OF MISSING FEATURES VS ADDICTION",
        width=100,
    )

    count_report = (
        analyse_missing_count_vs_target(
            train_df
        )
    )

    print(
        count_report.to_string(
            index=False
        )
    )

    print_section(
        "PAIRWISE MISSINGNESS INTERACTIONS",
        width=120,
    )

    pairwise_report = (
        analyse_pairwise_missingness(
            train_df,
            min_support=min_pairwise_support,
        )
    )

    if pairwise_report.empty:
        print(
            "No pairwise combinations "
            "satisfied minimum support."
        )
        return

    print(
        pairwise_report
        .head(
            top_pairs
        )
        .to_string(
            index=False
        )
    )