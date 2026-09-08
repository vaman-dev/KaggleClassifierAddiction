from __future__ import annotations

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import create_tree_preprocessor


def create_random_forest_benchmark(
    X_train: pd.DataFrame,
) -> Pipeline:

    return Pipeline(
        steps=[
            (
                "preprocessor",
                create_tree_preprocessor(
                    X_train
                ),
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )