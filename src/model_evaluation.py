from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import pandas as pd

from sklearn.metrics import (
    roc_auc_score,
)


def get_model_scores(
    model: Any,
    X: pd.DataFrame,
):
    if hasattr(
        model,
        "predict_proba",
    ):
        probabilities = (
            model.predict_proba(X)
        )

        return probabilities[:, 1]

    if hasattr(
        model,
        "decision_function",
    ):
        return (
            model.decision_function(
                X
            )
        )

    raise TypeError(
        f"{type(model).__name__} "
        "provides neither predict_proba() "
        "nor decision_function()."
    )


def evaluate_models(
    models: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> pd.DataFrame:

    results = []

    for name, model in models.items():

        print()
        print(
            "=" * 70
        )
        print(
            f"TRAINING: {name}"
        )
        print(
            "=" * 70
        )

        start_time = (
            time.perf_counter()
        )

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter()
            - start_time
        )

        validation_scores = (
            get_model_scores(
                model,
                X_validation,
            )
        )

        validation_auc = (
            roc_auc_score(
                y_validation,
                validation_scores,
            )
        )

        print(
            f"ROC-AUC: "
            f"{validation_auc:.6f}"
        )

        print(
            f"Training time: "
            f"{training_seconds:.2f}s"
        )

        results.append(
            {
                "model":
                    name,

                "roc_auc":
                    validation_auc,

                "training_seconds":
                    training_seconds,
            }
        )

    return (
        pd.DataFrame(
            results
        )
        .sort_values(
            by="roc_auc",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )