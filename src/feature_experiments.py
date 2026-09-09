from __future__ import annotations

import pandas as pd

from src.cross_validation import (
    create_stratified_folds,
    run_xgboost_cross_validation,
    summarize_cross_validation,
)
from src.feature_engineering import engineer_features


FEATURE_EXPERIMENTS = (
    ("Baseline", {}),
    ("Baseline + usage totals", {"usage_totals": True}),
    ("Baseline + usage shares", {"usage_shares": True}),
    ("Baseline + interaction intensity", {"interaction_intensity": True}),
    ("Baseline + sleep features", {"sleep_features": True}),
    ("Baseline + weekend features", {"weekend_features": True}),
    ("Baseline + missingness features", {"missingness_features": True}),
)


def run_feature_group_experiments(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Compare one feature group at a time on one frozen set of folds.

    This function reports evidence only. Feature retention is decided after
    reviewing the cross-validation results.
    """
    fold_indices = create_stratified_folds(X, y, n_splits=n_splits)
    rows: list[dict[str, float | str]] = []

    for name, options in FEATURE_EXPERIMENTS:
        print(f"\n{'=' * 80}\nFEATURE EXPERIMENT: {name}\n{'=' * 80}")
        results = run_xgboost_cross_validation(
            engineer_features(X, **options),
            y,
            n_splits=n_splits,
            fold_indices=fold_indices,
        )
        summary = summarize_cross_validation(results)
        row: dict[str, float | str] = {
            "experiment": name,
            "mean_auc": summary["mean_auc"],
            "std_auc": summary["std_auc"],
        }
        for fold, fold_auc in zip(results["fold"], results["roc_auc"], strict=True):
            row[f"fold_{fold}_auc"] = fold_auc
        rows.append(row)

    comparison = pd.DataFrame(rows)
    comparison["auc_delta_vs_baseline"] = (
        comparison["mean_auc"] - comparison.loc[0, "mean_auc"]
    )
    return comparison
