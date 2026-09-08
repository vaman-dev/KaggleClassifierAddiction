# src/xgboost_model.py

from __future__ import annotations

import pandas as pd

from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    RANDOM_STATE,
)

from src.preprocessing import (
    XGBoostMissingStrategy,
    create_xgboost_preprocessor,
)


def create_xgboost_classifier() -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=3000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def create_xgboost_model(
    X_train: pd.DataFrame,
    *,
    missing_strategy: XGBoostMissingStrategy = "native",
) -> Pipeline:

    preprocessor = create_xgboost_preprocessor(
        X_train,
        strategy=missing_strategy,
    )

    classifier = create_xgboost_classifier()

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                classifier,
            ),
        ]
    )
