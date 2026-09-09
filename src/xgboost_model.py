from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.preprocessing import (
    XGBoostMissingStrategy,
    create_xgboost_preprocessor,
)


# ---------------------------------------------------------
# Default / baseline XGBoost parameters
# ---------------------------------------------------------

BASE_XGBOOST_PARAMS: dict[str, Any] = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "n_estimators": 3000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
    "gamma": 0.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}


def create_xgboost_classifier(
    model_params: dict[str, Any] | None = None,
) -> XGBClassifier:
    """
    Create an XGBoost classifier.

    Any supplied model_params override the baseline parameters.
    """

    params = BASE_XGBOOST_PARAMS.copy()

    if model_params is not None:
        params.update(model_params)

    return XGBClassifier(**params)


def create_xgboost_model(
    X_train: pd.DataFrame,
    *,
    missing_strategy: XGBoostMissingStrategy = "native",
    model_params: dict[str, Any] | None = None,
) -> Pipeline:
    """
    Create the complete preprocessing + XGBoost pipeline.
    """

    preprocessor = create_xgboost_preprocessor(
        X_train,
        strategy=missing_strategy,
    )

    classifier = create_xgboost_classifier(
        model_params=model_params,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )
