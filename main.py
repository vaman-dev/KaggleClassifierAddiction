from __future__ import annotations

import numpy as np
import sklearn

from src.config import (
    PAIRWISE_MIN_SUPPORT,
    RANDOM_STATE,
)

from src.data_analysis import (
    run_data_analysis,
    run_missingness_analysis,
)

from src.data_loader import (
    create_train_validation_split,
    load_competition_data,
)

from src.preprocessing import (
    separate_test_data,
    separate_train_data,
)

from src.xgboost_model import (
    create_xgboost_model,
)

from src.model_evaluation import (
    evaluate_models,
)

from src.cross_validation import (
    run_xgboost_cross_validation,
    print_cross_validation_summary,
)

from src.hyperparameter_tuning import run_hyperparameter_tuning_pipeline


RUN_GENERAL_EDA = False
RUN_MISSINGNESS_EDA = False
RUN_XGBOOST_STRATEGY_EXPERIMENTS = False
RUN_CURRENT_CHAMPION = True

RUN_CROSS_VALIDATION = False

RUN_HYPERPARAMETER_TUNING = False


XGBOOST_STRATEGIES = {
    "XGBoost - Median + Indicator": "median_indicator",
    "XGBoost - Median Only": "median",
    "XGBoost - Native NaN": "native",
}


CURRENT_CHAMPION_NAME = "XGBoost + Native NaN"
CURRENT_CHAMPION_STRATEGY = "native"
CURRENT_CHAMPION_AUC = 0.964272


def print_section(
    title: str,
    width: int = 70,
) -> None:
    print()
    print("=" * width)
    print(title)
    print("=" * width)


def print_project_status() -> None:
    print_section(
        "CURRENT PROJECT STATUS"
    )

    print(
        f"Champion model: "
        f"{CURRENT_CHAMPION_NAME}"
    )

    print(
        f"Champion validation ROC-AUC: "
        f"{CURRENT_CHAMPION_AUC:.6f}"
    )

    print(
        "Current missing-value strategy: "
        "Native numerical NaN handling"
    )

    print()
    print("Next phase:")
    print(
        "Stratified K-Fold Cross-Validation"
    )


def load_and_split_data():
    print_section(
        "LOADING COMPETITION DATA"
    )

    (
        competition_train_df,
        competition_test_df,
        sample_submission_df,
    ) = load_competition_data()

    print(
        "Original train:",
        competition_train_df.shape,
    )

    print(
        "Competition test:",
        competition_test_df.shape,
    )

    print(
        "Sample submission:",
        sample_submission_df.shape,
    )

    print_section(
        "TRAIN / VALIDATION SPLIT"
    )

    (
        train_df,
        validation_df,
    ) = create_train_validation_split(
        competition_train_df
    )

    print(
        "Training split:",
        train_df.shape,
    )

    print(
        "Validation split:",
        validation_df.shape,
    )

    return (
        competition_train_df,
        competition_test_df,
        sample_submission_df,
        train_df,
        validation_df,
    )


def prepare_model_data(
    train_df,
    validation_df,
    competition_test_df,
):
    print_section(
        "PREPARING MODEL DATA"
    )

    (
        X_train,
        y_train,
        train_ids,
    ) = separate_train_data(
        train_df
    )

    (
        X_validation,
        y_validation,
        validation_ids,
    ) = separate_train_data(
        validation_df
    )

    (
        X_test,
        test_ids,
    ) = separate_test_data(
        competition_test_df
    )

    print(
        "X_train:",
        X_train.shape,
    )

    print(
        "y_train:",
        y_train.shape,
    )

    print(
        "X_validation:",
        X_validation.shape,
    )

    print(
        "y_validation:",
        y_validation.shape,
    )

    print(
        "X_test:",
        X_test.shape,
    )

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        train_ids,
        validation_ids,
        test_ids,
    )


def run_xgboost_strategy_experiments(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> None:
    print_section(
        "XGBOOST MISSING-VALUE EXPERIMENTS"
    )

    models = {}

    for (
        model_name,
        missing_strategy,
    ) in XGBOOST_STRATEGIES.items():

        models[
            model_name
        ] = create_xgboost_model(
            X_train,
            missing_strategy=missing_strategy,
        )

    results = evaluate_models(
        models=models,
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    print_section(
        "XGBOOST STRATEGY COMPARISON"
    )

    print(
        results.to_string(
            index=False
        )
    )


def run_current_champion(
    X_train,
    y_train,
    X_validation,
    y_validation,
) -> None:
    print_section(
        "CURRENT CHAMPION EVALUATION"
    )

    model = create_xgboost_model(
        X_train,
        missing_strategy=(
            CURRENT_CHAMPION_STRATEGY
        ),
    )

    results = evaluate_models(
        models={
            CURRENT_CHAMPION_NAME:
                model
        },
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
    )

    print_section(
        "CURRENT CHAMPION RESULT"
    )

    print(
        results.to_string(
            index=False
        )
    )


def main() -> None:
    print_section(
        "SMARTPHONE ADDICTION CLASSIFIER"
    )

    print(
        f"NumPy: {np.__version__}"
    )

    print(
        f"Scikit-learn: "
        f"{sklearn.__version__}"
    )

    print_project_status()

    (
        competition_train_df,
        competition_test_df,
        sample_submission_df,
    ) = load_competition_data()

    print_section(
        "LOADING COMPETITION DATA"
    )

    print(
        "Original train:",
        competition_train_df.shape,
    )

    print(
        "Competition test:",
        competition_test_df.shape,
    )

    print(
        "Sample submission:",
        sample_submission_df.shape,
    )


    # =====================================================
    # Phase 5 - Cross-validation
    # =====================================================

    if RUN_CROSS_VALIDATION:

        (
            X_full,
            y_full,
            _,
        ) = separate_train_data(
            competition_train_df
        )

        cv_results = (
            run_xgboost_cross_validation(
                X=X_full,
                y=y_full,
                n_splits=5,
            )
        )

        print_cross_validation_summary(
            cv_results
        )

        return


    # =====================================================
    # Previous holdout workflow
    # =====================================================

    (
        train_df,
        validation_df,
    ) = create_train_validation_split(
        competition_train_df
    )

    if RUN_GENERAL_EDA:

        run_data_analysis(
            train_df,
            competition_test_df,
        )

    if RUN_MISSINGNESS_EDA:

        run_missingness_analysis(
            train_df,
            min_pairwise_support=(
                PAIRWISE_MIN_SUPPORT
            ),
            top_pairs=30,
        )

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        train_ids,
        validation_ids,
        test_ids,
    ) = prepare_model_data(
        train_df,
        validation_df,
        competition_test_df,
    )

    if RUN_XGBOOST_STRATEGY_EXPERIMENTS:

        run_xgboost_strategy_experiments(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

    if RUN_CURRENT_CHAMPION:

        run_current_champion(
            X_train,
            y_train,
            X_validation,
            y_validation,
        )

    if RUN_HYPERPARAMETER_TUNING:

        run_hyperparameter_tuning_pipeline()


if __name__ == "__main__":
    main()
