from __future__ import annotations

import time

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE
from src.xgboost_model import create_xgboost_model


def run_xgboost_cross_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:

    cross_validator = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    print()
    print("=" * 80)
    print("XGBOOST STRATIFIED CROSS-VALIDATION")
    print("=" * 80)

    for fold_number, (
        train_index,
        validation_index,
    ) in enumerate(
        cross_validator.split(
            X,
            y,
        ),
        start=1,
    ):

        print()
        print("-" * 80)
        print(
            f"FOLD {fold_number}/{n_splits}"
        )
        print("-" * 80)

        X_fold_train = (
            X.iloc[
                train_index
            ]
            .reset_index(drop=True)
        )

        y_fold_train = (
            y.iloc[
                train_index
            ]
            .reset_index(drop=True)
        )

        X_fold_validation = (
            X.iloc[
                validation_index
            ]
            .reset_index(drop=True)
        )

        y_fold_validation = (
            y.iloc[
                validation_index
            ]
            .reset_index(drop=True)
        )

        print(
            "Training rows:",
            X_fold_train.shape[0],
        )

        print(
            "Validation rows:",
            X_fold_validation.shape[0],
        )

        model = create_xgboost_model(
            X_fold_train,
            missing_strategy="native",
        )

        start_time = (
            time.perf_counter()
        )

        model.fit(
            X_fold_train,
            y_fold_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        probabilities = (
            model.predict_proba(
                X_fold_validation
            )[:, 1]
        )

        fold_auc = (
            roc_auc_score(
                y_fold_validation,
                probabilities,
            )
        )

        print(
            f"ROC-AUC: "
            f"{fold_auc:.6f}"
        )

        print(
            f"Training time: "
            f"{training_seconds:.2f}s"
        )

        fold_results.append(
            {
                "fold": fold_number,
                "train_rows": (
                    len(train_index)
                ),
                "validation_rows": (
                    len(validation_index)
                ),
                "roc_auc": fold_auc,
                "training_seconds": (
                    training_seconds
                ),
            }
        )

    results = pd.DataFrame(
        fold_results
    )

    return results


def summarize_cross_validation(
    results: pd.DataFrame,
) -> dict[str, float]:

    auc_values = (
        results[
            "roc_auc"
        ]
        .to_numpy()
    )

    training_times = (
        results[
            "training_seconds"
        ]
        .to_numpy()
    )

    summary = {
        "mean_auc": float(
            np.mean(
                auc_values
            )
        ),

        "std_auc": float(
            np.std(
                auc_values,
                ddof=1,
            )
        ),

        "min_auc": float(
            np.min(
                auc_values
            )
        ),

        "max_auc": float(
            np.max(
                auc_values
            )
        ),

        "mean_training_seconds": float(
            np.mean(
                training_times
            )
        ),

        "total_training_seconds": float(
            np.sum(
                training_times
            )
        ),
    }

    return summary


def print_cross_validation_summary(
    results: pd.DataFrame,
) -> None:

    summary = (
        summarize_cross_validation(
            results
        )
    )

    print()
    print("=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)

    print(
        results.to_string(
            index=False
        )
    )

    print()
    print("=" * 80)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 80)

    print(
        f"Mean ROC-AUC: "
        f"{summary['mean_auc']:.6f}"
    )

    print(
        f"Std ROC-AUC:  "
        f"{summary['std_auc']:.6f}"
    )

    print(
        f"Min ROC-AUC:  "
        f"{summary['min_auc']:.6f}"
    )

    print(
        f"Max ROC-AUC:  "
        f"{summary['max_auc']:.6f}"
    )

    print(
        f"Mean training time: "
        f"{summary['mean_training_seconds']:.2f}s"
    )

    print(
        f"Total training time: "
        f"{summary['total_training_seconds']:.2f}s"
    )