from unittest.mock import Mock

import numpy as np
import optuna
import pandas as pd
import pytest

from src import cross_validation as cv
from src import hyperparameter_tuning as tuning
from src.xgboost_model import BASE_XGBOOST_PARAMS, create_xgboost_model


@pytest.fixture
def data():
    return pd.DataFrame({"value": range(20)}), pd.Series([0, 1] * 10)


def test_xgboost_parameter_overrides_are_applied(data):
    custom = {"max_depth": 3, "learning_rate": 0.025, "min_child_weight": 7}
    baseline = BASE_XGBOOST_PARAMS.copy()
    model = create_xgboost_model(data[0], model_params=custom).named_steps["model"]
    for key, value in custom.items():
        assert model.get_params()[key] == value
    assert BASE_XGBOOST_PARAMS == baseline


def fake_model():
    model = Mock()
    model.predict_proba.side_effect = lambda X: np.tile([0.5, 0.5], (len(X), 1))
    return model


def test_final_cv_applies_best_params_to_every_fresh_fold(monkeypatch, data):
    factory = Mock(side_effect=lambda *a, **k: fake_model())
    monkeypatch.setattr(cv, "create_xgboost_model", factory)
    params = {"max_depth": 3, "gamma": 0.17}
    result = tuning.run_final_cross_validation(*data, best_params=params)
    assert factory.call_count == 5
    assert result["mean_auc"] == 0.5
    for call in factory.call_args_list:
        assert call.kwargs["model_params"] == params
        assert len(call.args[0]) == 16


def test_objective_reports_scores_and_applies_suggestions(monkeypatch, data):
    factory = Mock(side_effect=lambda *a, **k: fake_model())
    monkeypatch.setattr(tuning, "create_xgboost_model", factory)
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    study.optimize(lambda t: tuning.objective(t, *data, n_splits=2), n_trials=1)
    trial = study.trials[0]
    assert trial.value == 0.5
    assert trial.intermediate_values == {1: 0.5, 2: 0.5}
    assert factory.call_count == 2
    for call in factory.call_args_list:
        assert call.kwargs["model_params"] == trial.params


def test_pruning_stops_remaining_folds(monkeypatch, data):
    factory = Mock(side_effect=lambda *a, **k: fake_model())
    monkeypatch.setattr(tuning, "create_xgboost_model", factory)
    study = optuna.create_study(pruner=optuna.pruners.ThresholdPruner(lower=0.6))
    study.optimize(lambda t: tuning.objective(t, *data, n_splits=2), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.PRUNED
    assert factory.call_count == 1


def test_configuration_errors_fail_visibly(monkeypatch, data):
    monkeypatch.setattr(tuning, "create_xgboost_model", Mock(side_effect=ValueError("bad config")))
    study = optuna.create_study()
    with pytest.raises(ValueError, match="bad config"):
        study.optimize(lambda t: tuning.objective(t, *data, n_splits=2), n_trials=1)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_main_imports_with_tuning_disabled():
    import main
    assert main.RUN_HYPERPARAMETER_TUNING is False
