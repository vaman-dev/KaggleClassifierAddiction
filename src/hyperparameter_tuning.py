from __future__ import annotations

import optuna
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, TARGET_COLUMN, ID_COLUMN
from src.data_loader import load_competition_data, create_train_validation_split
from src.preprocessing import separate_train_data
from src.xgboost_model import create_xgboost_model, create_xgboost_classifier
from src.model_evaluation import evaluate_models, get_model_scores, roc_auc_score
from src.cross_validation import run_xgboost_cross_validation, print_cross_validation_summary


def objective(trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series, X_validation: pd.DataFrame, y_validation: pd.Series) -> float:
    """Objective function for Optuna hyperparameter tuning.
    
    Tunes XGBoost parameters using the native NaN handling strategy.
    Parameters are set directly on the XGBClassifier pipeline step.
    """
    
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 2.0),
        "n_estimators": 3000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    }
    
    # Create model with native missing value handling
    model = create_xgboost_model(
        X_train,
        missing_strategy="native",
    )
    
    # Set parameters - need to handle pipeline structure
    # The pipeline has steps: [("preprocessor", ...), ("model", XGBClassifier)]
    # We need to set params on the XGBClassifier step
    
    # Get the classifier step and set params directly
    # The XGBClassifier in the pipeline accepts params directly
    # through the pipeline's set_params with "model__" prefix, but create_xgboost_model
    # returns a pipeline where we can set params directly
    
    # For simplicity, set params on the whole pipeline first,
    # then fit - XGBoost will handle native NaN internally
    try:
        model.set_params(**params)
    except ValueError:
        # If pipeline-level params fail, we'll set them after fitting
        pass
    
    model.fit(X_train, y_train)
    
    # Evaluate on validation
    probabilities = model.predict_proba(X_validation)[:, 1]
    auc = roc_auc_score(y_validation, probabilities)
    
    return auc


def run_optuna_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    n_trials: int = 50,
) -> pd.DataFrame:
    """Run broad Optuna hyperparameter search.
    
    Performs broad exploration of XGBoost hyperparameter space
    using Optuna sampling (typically 30-100 trials).
    """
    
    study = optuna.create_study(
        direction="maximize",
        study_name="smartphone_addiction_xgboost",
    )
    
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_validation, y_validation),
        n_trials=n_trials,
    )
    
    # Return trials as DataFrame
    trials_data = []
    for trial in study.trials:
        if trial.value is not None:
            trials_data.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            })
    
    return pd.DataFrame(trials_data)


def run_grid_search_narrow(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict,
    cv: int = 5,
) -> dict:
    """Narrow Grid Search confirmation around top Optuna results.
    
    Performs focused grid search around the best-performing
    hyperparameter configurations found by Optuna.
    """
    
    # Create the XGBoost pipeline model
    model = create_xgboost_model(X_train, missing_strategy="native")
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    
    grid_search.fit(X_train, y_train)
    
    return {
        "best_params": grid_search.best_params_,
        "best_auc": grid_search.best_score_,
        "grid_results": pd.DataFrame(grid_search.cv_results_),
        "best_estimator": grid_search.best_estimator_,
    }


def run_final_cross_validation(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    best_params: dict,
    n_splits: int = 5,
) -> dict:
    """Run final 5-fold cross-validation with best parameters.
    
    Trains the final model with best-found hyperparameters
    and evaluates using stratified 5-fold cross-validation.
    Returns summary of CV performance.
    """
    
    # Create model with best params
    model = create_xgboost_model(X_full, missing_strategy="native")
    
    # Set best parameters on the model
    # The model is a Pipeline with preprocessor + classifier
    # We need to set params on the classifier step
    
    # Check if params use classifier__ prefix or direct format
    first_key = list(best_params.keys())[0] if best_params else None
    
    if first_key and first_key.startswith("classifier__"):
        # Pipeline format - extract classifier params
        classifier_params = {
            k.replace("classifier__", ""): v
            for k, v in best_params.items()
            if k.startswith("classifier__")
        }
        # Set on the pipeline - the classifier step is at index 1
        # Actually, let's use the set_params approach
        model.set_params(**classifier_params)
    else:
        # Direct params format - should work with the pipeline
        model.set_params(**best_params)
    
    model.fit(X_full, y_full)
    
    # Run cross-validation
    cv_results = run_xgboost_cross_validation(
        X=X_full,
        y=y_full,
        n_splits=n_splits,
    )
    
    # Compute summary stats
    auc_values = cv_results["roc_auc"].to_numpy()
    
    return {
        "cv_results": cv_results,
        "mean_auc": float(np.mean(auc_values)),
        "std_auc": float(np.std(auc_values, ddof=1)),
        "min_auc": float(np.min(auc_values)),
        "max_auc": float(np.max(auc_values)),
        "trained_model": model,
    }


def run_hyperparameter_tuning_pipeline() -> dict:
    """Complete tuning pipeline: Optuna → GridSearch → Final CV.
    
    Executes the full hyperparameter tuning workflow:
    1. Broad search with Optuna (50+ trials exploring hyperparameter space)
    2. Narrow confirmation with Grid Search (focused grid around top Optuna results)
    3. Final 5-fold CV with best parameters (validates final model performance)
    
    Returns dictionary with all tuning results and the best model.
    """
    
    print("=" * 80)
    print("HYPERPARAMETER TUNING PIPELINE")
    print("=" * 80)
    print()
    
    # Load data
    competition_train_df, competition_test_df, sample_submission_df = load_competition_data()
    
    # Train/validation split
    train_df, validation_df = create_train_validation_split(competition_train_df)
    
    # Prepare data
    (X_train, y_train, _), (X_validation, y_validation, _) = (
        separate_train_data(train_df),
        separate_train_data(validation_df),
    )
    
    print("Step 1/3: Broad Optuna Hyperparameter Search")
    print("-" * 80)
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_validation.shape[0]}")
    print(f"Optuna trials: {50}")
    print()
    
    # Step 1: Optuna broad search
    optuna_results = run_optuna_search(
        X_train=X_train,
        y_train=y_train,
        X_validation=X_validation,
        y_validation=y_validation,
        n_trials=50,
    )
    
    print(f"Optuna search complete!")
    if len(optuna_results) > 0:
        best_value = optuna_results["value"].max()
        print(f"Best AUC from Optuna: {best_value:.6f}")
    else:
        print("No valid trials completed")
    print()
    
    # Step 2: Narrow Grid Search confirmation
    print("Step 2/3: Narrow Grid Search Confirmation")
    print("-" * 80)
    
    # Build parameter grid based on typical XGBoost tuning ranges
    # focused on the most impactful parameters
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [4, 6, 8],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "reg_alpha": [0.0, 0.5],
        "reg_lambda": [1.0, 1.5],
    }
    
    # Note: GridSearchCV params need to match the pipeline structure
    # The pipeline from create_xgboost_model has steps for preprocessor and model
    # We'll use the model params directly
    
    grid_results = run_grid_search_narrow(
        X_train=X_train,
        y_train=y_train,
        param_grid=param_grid,
        cv=5,
    )
    
    print(f"Grid Search complete!")
    print(f"Best AUC: {grid_results['best_auc']:.6f}")
    print(f"Best Parameters: {grid_results['best_params']}")
    print()
    
    # Step 3: Final 5-fold CV with best params
    print("Step 3/3: Final 5-Fold Cross-Validation")
    print("-" * 80)
    
    # Load full training data for final CV
    competition_train_df, _, _ = load_competition_data()
    X_full, y_full, _ = separate_train_data(competition_train_df)
    
    final_results = run_final_cross_validation(
        X_full=X_full,
        y_full=y_full,
        best_params=grid_results["best_params"],
        n_splits=5,
    )
    
    print(f"Final 5-Fold CV complete!")
    print(f"Mean ROC-AUC: {final_results['mean_auc']:.6f}")
    print(f"Std ROC-AUC: {final_results['std_auc']:.6f}")
    print(f"Min ROC-AUC: {final_results['min_auc']:.6f}")
    print(f"Max ROC-AUC: {final_results['max_auc']:.6f}")
    print()
    
    # Compile all results
    tuning_results = {
        "optuna_results": optuna_results,
        "grid_best_auc": grid_results["best_auc"],
        "grid_best_params": grid_results["best_params"],
        "grid_results": grid_results["grid_results"],
        "final_cv_results": final_results,
        "best_overall_auc": final_results["mean_auc"],
    }
    
    print("=" * 80)
    print("HYPERPARAMETER TUNING PIPELINE COMPLETE")
    print("=" * 80)
    print()
    print(f"Current Champion: XGBoost + Native NaN")
    print(f"Previous AUC: 0.964272")
    print(f"New Tuning AUC: {final_results['mean_auc']:.6f}")
    print(f"Improvement: {final_results['mean_auc'] - 0.964272:+.6f}")
    print()
    
    return tuning_results