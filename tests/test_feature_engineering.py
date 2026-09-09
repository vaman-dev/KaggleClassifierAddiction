import numpy as np
import pandas as pd

from src.feature_engineering import (
    FROZEN_FEATURE_OPTIONS,
    engineer_features,
    engineer_frozen_features,
    safe_divide,
)


def sample_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "daily_screen_time_hours": [10.0, 0.0, np.nan],
            "social_media_hours": [2.0, 1.0, 3.0],
            "gaming_hours": [3.0, 2.0, 1.0],
            "work_study_hours": [4.0, 3.0, 2.0],
            "sleep_hours": [8.0, 0.0, 7.0],
            "notifications_per_day": [100.0, 50.0, 20.0],
            "app_opens_per_day": [40.0, 25.0, 10.0],
            "weekend_screen_time": [12.0, 1.0, 5.0],
        }
    )


def test_safe_divide_turns_zero_denominators_into_missing_values():
    result = safe_divide(pd.Series([4.0, 2.0]), pd.Series([2.0, 0.0]))
    assert result.tolist()[0] == 2.0
    assert pd.isna(result.iloc[1])


def test_all_requested_feature_groups_are_added_without_mutating_input():
    X = sample_features()
    original_columns = X.columns.tolist()
    result = engineer_features(
        X,
        usage_totals=True,
        usage_shares=True,
        interaction_intensity=True,
        sleep_features=True,
        weekend_features=True,
        missingness_features=True,
    )
    assert X.columns.tolist() == original_columns
    assert result.loc[0, "leisure_hours"] == 5.0
    assert result.loc[0, "social_media_share"] == 0.2
    assert result.loc[0, "screen_minus_sleep"] == 2.0
    assert result.loc[0, "missing_feature_count"] == 0
    assert result.loc[1, "missing_feature_count"] == 0
    assert pd.isna(result.loc[1, "gaming_share"])


def test_frozen_feature_set_contains_only_usage_shares():
    result = engineer_frozen_features(sample_features())
    assert FROZEN_FEATURE_OPTIONS == {"usage_shares": True}
    assert {"social_media_share", "gaming_share", "work_study_share"}.issubset(result)
    assert "leisure_hours" not in result
