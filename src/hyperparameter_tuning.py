from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from src.config import PROJECT_ROOT, RANDOM_STATE
from src.cross_validation import run_xgboost_cross_validation, summarize_cross_validation
from src.feature_engineering import engineer_frozen_features
from src.xgboost_model import create_xgboost_model


FROZEN_FEATURE_BENCHMARK_MEAN_AUC = 0.964673
FROZEN_FEATURE_BENCHMARK_STD_AUC = 0.000499
DEFAULT_TUNING_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "hyperparameter_tuning"


# ---------------------------------------------------------
# Result structure
# ---------------------------------------------------------

@dataclass
class TuningResult:
    best_params: dict[str, Any]
    best_score: float
    study: optuna.Study


@dataclass
class FinalTuningResult:
    tuning: TuningResult
    final_cv: dict[str, Any]
    beats_frozen_feature_benchmark: bool


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
    n_trials: int = 25,
    n_splits: int = 3,
    output_dir: Path = DEFAULT_TUNING_OUTPUT_DIR,
) -> TuningResult:
    """
    Tune the frozen feature-engineered model with TPE and pruning.

    The selected candidate must still pass a separate five-fold confirmation.
    """

    X_engineered = engineer_frozen_features(X)

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
            X_engineered,
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
    print(f"Best trial number: {study.best_trial.number}")
    print(f"Best CV ROC-AUC  : {study.best_value:.6f}")
    print(f"Elapsed time     : {elapsed:.2f}s")

    print()
    print("Best parameters:")

    for name, value in study.best_params.items():
        print(f"  {name}: {value}")

    result = TuningResult(
        best_params=study.best_params,
        best_score=float(study.best_value),
        study=study,
    )
    save_optuna_results(result, output_dir=output_dir)
    return result


def run_final_cross_validation(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    best_params: dict[str, Any],
    n_splits: int = 5,
) -> dict[str, Any]:
    """Confirm selected parameters on frozen features with independent CV."""
    X_engineered = engineer_frozen_features(X_full)
    results = run_xgboost_cross_validation(
        X_engineered, y_full, n_splits=n_splits, model_params=best_params,
    )
    return {"cv_results": results, **summarize_cross_validation(results)}


def run_tuning_and_final_validation(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_trials: int = 25,
    tuning_splits: int = 3,
    final_splits: int = 5,
    output_dir: Path = DEFAULT_TUNING_OUTPUT_DIR,
) -> FinalTuningResult:
    """Run the real tuning stage followed by a clean five-fold confirmation."""
    tuning_result = run_optuna_tuning(
        X,
        y,
        n_trials=n_trials,
        n_splits=tuning_splits,
        output_dir=output_dir,
    )
    final_cv = run_final_cross_validation(
        X,
        y,
        best_params=tuning_result.best_params,
        n_splits=final_splits,
    )
    beats_benchmark = (
        final_cv["mean_auc"] > FROZEN_FEATURE_BENCHMARK_MEAN_AUC
    )

    print()
    print("=" * 70)
    print("FINAL 5-FOLD CONFIRMATION")
    print("=" * 70)
    print(
        "Frozen feature benchmark: "
        f"{FROZEN_FEATURE_BENCHMARK_MEAN_AUC:.6f} "
        f"± {FROZEN_FEATURE_BENCHMARK_STD_AUC:.6f}"
    )
    print(
        "Tuned final CV: "
        f"{final_cv['mean_auc']:.6f} ± {final_cv['std_auc']:.6f}"
    )
    print(
        "Mean AUC difference: "
        f"{final_cv['mean_auc'] - FROZEN_FEATURE_BENCHMARK_MEAN_AUC:+.6f}"
    )
    print(f"Candidate beats benchmark: {beats_benchmark}")

    result = FinalTuningResult(
        tuning=tuning_result,
        final_cv=final_cv,
        beats_frozen_feature_benchmark=beats_benchmark,
    )
    save_final_validation_results(result, output_dir=output_dir)
    return result


def save_optuna_results(
    result: TuningResult,
    *,
    output_dir: Path,
) -> None:
    """Save trial history and the selected three-fold candidate for review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.study.trials_dataframe().to_csv(
        output_dir / "optuna_trials.csv",
        index=False,
    )
    (output_dir / "best_params.json").write_text(
        json.dumps(
            {
                "best_trial_number": result.study.best_trial.number,
                "best_3fold_mean_auc": result.best_score,
                "best_params": result.best_params,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def save_final_validation_results(
    result: FinalTuningResult,
    *,
    output_dir: Path,
) -> None:
    """Save the independent five-fold confirmation and benchmark comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.final_cv["cv_results"].to_csv(
        output_dir / "final_5fold_results.csv",
        index=False,
    )
    (output_dir / "final_5fold_summary.json").write_text(
        json.dumps(
            {
                "frozen_feature_benchmark_mean_auc": (
                    FROZEN_FEATURE_BENCHMARK_MEAN_AUC
                ),
                "frozen_feature_benchmark_std_auc": (
                    FROZEN_FEATURE_BENCHMARK_STD_AUC
                ),
                "final_5fold_mean_auc": result.final_cv["mean_auc"],
                "final_5fold_std_auc": result.final_cv["std_auc"],
                "best_params": result.tuning.best_params,
                "beats_frozen_feature_benchmark": (
                    result.beats_frozen_feature_benchmark
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
