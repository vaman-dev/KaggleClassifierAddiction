from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


FeatureTransformer = Callable[[pd.DataFrame], pd.DataFrame]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while treating a zero denominator as missing data."""
    return numerator / denominator.replace(0, np.nan)


def add_usage_total_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["leisure_hours"] = X["social_media_hours"] + X["gaming_hours"]
    return X


def add_usage_share_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    screen_time = X["daily_screen_time_hours"]
    X["social_media_share"] = safe_divide(X["social_media_hours"], screen_time)
    X["gaming_share"] = safe_divide(X["gaming_hours"], screen_time)
    X["work_study_share"] = safe_divide(X["work_study_hours"], screen_time)
    return X


def add_interaction_intensity_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    screen_time = X["daily_screen_time_hours"]
    X["notifications_per_screen_hour"] = safe_divide(
        X["notifications_per_day"], screen_time
    )
    X["app_opens_per_screen_hour"] = safe_divide(
        X["app_opens_per_day"], screen_time
    )
    return X


def add_sleep_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["screen_to_sleep_ratio"] = safe_divide(
        X["daily_screen_time_hours"], X["sleep_hours"]
    )
    X["screen_minus_sleep"] = X["daily_screen_time_hours"] - X["sleep_hours"]
    return X


def add_weekend_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    screen_time = X["daily_screen_time_hours"]
    X["weekend_screen_delta"] = X["weekend_screen_time"] - screen_time
    X["weekend_screen_ratio"] = safe_divide(X["weekend_screen_time"], screen_time)
    return X


def add_missingness_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    X["missing_feature_count"] = X.isna().sum(axis=1)
    return X


FEATURE_GROUPS: dict[str, FeatureTransformer] = {
    "usage_totals": add_usage_total_features,
    "usage_shares": add_usage_share_features,
    "interaction_intensity": add_interaction_intensity_features,
    "sleep_features": add_sleep_features,
    "weekend_features": add_weekend_features,
    "missingness_features": add_missingness_features,
}

# Frozen after the five-fold feature comparison on 2026-09-09.
FROZEN_FEATURE_OPTIONS = {
    "usage_shares": True,
}


def engineer_features(
    X: pd.DataFrame,
    *,
    usage_totals: bool = False,
    usage_shares: bool = False,
    interaction_intensity: bool = False,
    sleep_features: bool = False,
    weekend_features: bool = False,
    missingness_features: bool = False,
) -> pd.DataFrame:
    """Return a copy of X with only the requested feature groups added."""
    original_missing_count = X.isna().sum(axis=1)
    enabled_groups = {
        "usage_totals": usage_totals,
        "usage_shares": usage_shares,
        "interaction_intensity": interaction_intensity,
        "sleep_features": sleep_features,
        "weekend_features": weekend_features,
        "missingness_features": missingness_features,
    }
    engineered = X.copy()
    for name, enabled in enabled_groups.items():
        if enabled:
            if name == "missingness_features":
                # Count source-column missingness only. Derived ratios can be
                # missing because their denominator is zero, which is separate
                # from a missing source value.
                engineered["missing_feature_count"] = original_missing_count
            else:
                engineered = FEATURE_GROUPS[name](engineered)
    return engineered


def engineer_frozen_features(X: pd.DataFrame) -> pd.DataFrame:
    """Apply the feature groups selected by the frozen CV experiment."""
    return engineer_features(X, **FROZEN_FEATURE_OPTIONS)
