from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_STATE
from src.cross_validation import run_xgboost_cross_validation, summarize_cross_validation
from src.xgboost_model import create_xgboost_model


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    study: optuna.Study


# ---------------------------------------------------------
# Objective
# ---------------------------------------------------------

def objective(
    trial: optuna.Trial,
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 3,
) -> float:
    """
    Evaluate one Optuna trial using stratified cross-validation.

    IMPORTANT:
    - Every trial creates a fresh model.
    - Suggested parameters are actually applied.
    - Programming/configuration errors are NOT silently replaced
      with the baseline model.
    """

    params = {
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8,
        ),

        "min_child_weight": trial.suggest_int(
            "min_child_weight",
            1,
            10,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.02,
            0.10,
            log=True,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.70,
            1.00,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.70,
            1.00,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            0.0,
            5.0,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            0.5,
            10.0,
            log=True,
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            3.0,
        ),
    }

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_scores: list[float] = []

    for fold_number, (train_idx, val_idx) in enumerate(
        cv.split(X, y),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Fresh pipeline for every fold
        model = create_xgboost_model(
            X_train,
            missing_strategy="native",
            model_params=params,
        )

        model.fit(
            X_train,
            y_train,
        )

        val_probabilities = model.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(
            y_val,
            val_probabilities,
        )

        fold_scores.append(auc)

        # Report intermediate result to Optuna
        running_mean = float(np.mean(fold_scores))

        trial.report(
            running_mean,
            step=fold_number,
        )

        # Allows Optuna to stop unpromising trials
        if fold_number < n_splits and trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(fold_scores))


# ---------------------------------------------------------
# Run tuning
# ---------------------------------------------------------

def run_optuna_tuning(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 3,
    n_splits: int = 2,
) -> TuningResult:
    """
    Run a small Optuna study.

    Start small while validating the tuning infrastructure.
    """

    sampler = optuna.samplers.TPESampler(
        seed=RANDOM_STATE,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=1,
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    start_time = time.perf_counter()

    study.optimize(
        lambda trial: objective(
            trial,
            X,
            y,
            n_splits=n_splits,
        ),
        n_trials=n_trials,
    )

    elapsed = time.perf_counter() - start_time

    print()
    print("=" * 70)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 70)

    for state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
                  optuna.trial.TrialState.FAIL):
        print(f"{state.name} trials: {sum(t.state == state for t in study.trials)}")
    print(f"Best CV ROC-AUC  : {study.best_value:.6f}")
    print(f"Elapsed time     : {elapsed:.2f}s")

    print()
    print("Best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    return TuningResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        study=study,
    )


def run_final_cross_validation(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    best_params: dict[str, Any],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Evaluate selected parameters on each fold; do not fit on all data yet."""
    results = run_xgboost_cross_validation(
        X_full, y_full, n_splits=n_splits, model_params=best_params,
    )
    return {"cv_results": results, **summarize_cross_validation(results)}
