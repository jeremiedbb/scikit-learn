# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from importlib.util import find_spec
from itertools import product

import numpy as np
import pytest

from sklearn.callback import ScoringMonitor
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    MetaEstimator,
    WhileEstimator,
)
from sklearn.metrics import check_scoring


def make_expected_ouptput_MaxIterEstimator(
    max_iter, scoring, on, X_train, y_train, X_val, y_val
):
    """Generate the expected dict output of a ScoringMonitor on a MaxIterEstimator."""
    scorer = check_scoring(None, scoring)
    if isinstance(scoring, str):
        scoring_names = [scoring]
    elif callable(scoring):
        scoring_names = ["score"]
    else:  # multi-scorer
        scoring_names = list(scorer._scorers.keys())
    est_name = MaxIterEstimator.__name__

    expected_log = []
    for i in range(max_iter):
        fitted_est = MaxIterEstimator(max_iter=i + 1).fit()
        if on in ("train_set", "both"):
            log_item = {
                f"0_{est_name}_fit": 0,
                f"1_{est_name}_iteration": i,
                "on": "train_set",
            }
            score = scorer(fitted_est, X_train, y_train)
            if isinstance(score, dict):
                log_item.update(score)
            else:
                log_item[scoring_names[0]] = score
            expected_log.append(log_item)

        if on in ("validation_set", "both"):
            log_item = {
                f"0_{est_name}_fit": 0,
                f"1_{est_name}_iteration": i,
                "on": "validation_set",
            }
            score = scorer(fitted_est, X_val, y_val)
            if isinstance(score, dict):
                log_item.update(score)
            else:
                log_item[scoring_names[0]] = score
            expected_log.append(log_item)

    return expected_log, scoring_names


def make_expected_ouptput_MetaEstimator(
    n_outer, n_inner, max_iter, scoring, on, X_train, y_train, X_val, y_val
):
    """Generate the expected dict output of a ScoringMonitor on a MetaEstimator.

    The sub-estimators are expected to be MaxIterEstimator.
    """
    scorer = check_scoring(None, scoring)
    if isinstance(scoring, str):
        scoring_names = [scoring]
    elif callable(scoring):
        scoring_names = ["score"]
    else:  # multi-scorer
        scoring_names = list(scorer._scorers.keys())
    meta_est_name = MetaEstimator.__name__
    sub_est_name = MaxIterEstimator.__name__

    expected_log = []
    for i_outer, i_inner in product(range(n_outer), range(n_inner)):
        for i_estimator_iteration in range(max_iter):
            fitted_est = MaxIterEstimator(max_iter=i_estimator_iteration + 1).fit()
            if on in ("train_set", "both"):
                log_item = {
                    "on": "train_set",
                    f"0_{meta_est_name}_fit": 0,
                    f"1_{meta_est_name}_outer": i_outer,
                    f"2_{meta_est_name}_inner|{sub_est_name}_fit": i_inner,
                    f"3_{sub_est_name}_iteration": i_estimator_iteration,
                }
                score = scorer(fitted_est, X_train, y_train)
                if isinstance(score, dict):
                    log_item.update(score)
                else:
                    log_item[scoring_names[0]] = score
                expected_log.append(log_item)

            if on in ("validation_set", "both"):
                log_item = {
                    "on": "validation_set",
                    f"0_{meta_est_name}_fit": 0,
                    f"1_{meta_est_name}_outer": i_outer,
                    f"2_{meta_est_name}_inner|{sub_est_name}_fit": i_inner,
                    f"3_{sub_est_name}_iteration": i_estimator_iteration,
                }
                score = scorer(fitted_est, X_val, y_val)
                if isinstance(score, dict):
                    log_item.update(score)
                else:
                    log_item[scoring_names[0]] = score
                expected_log.append(log_item)

    return expected_log, scoring_names


def custom_score(estimator, X, y):
    """Custom score to test the ScoringMonitor with a callable."""
    return 0


@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_scoring_monitor_logged_values(scoring, on):
    """Test that the correct values are logged with a simple estimator.

    This test only looks at the dict outputs from `get_logs`.
    """
    max_iter = 3
    n_dim = 5
    n_samples = 3
    estimator = MaxIterEstimator(max_iter=max_iter)
    callback = ScoringMonitor(on=on, scoring=scoring)
    estimator.set_callbacks(callback)
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)

    estimator.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    log = callback.get_logs(as_frame=False, select="most_recent")

    expected_log, scoring_names = make_expected_ouptput_MaxIterEstimator(
        max_iter, scoring, on, X_train, y_train, X_val, y_val
    )
    assert len(log) == len(expected_log)
    for i in range(len(log)):
        assert set(log[i].keys()) == set(expected_log[i].keys())
        for key, val in log[i].items():
            assert val == expected_log[i][key]


@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_scoring_monitor_logged_values_dataframe(scoring, on):
    """Test that the correct values are logged with a simple estimator.

    This test only looks at the pandas dataframe outputs from `get_logs`.
    """
    pd = pytest.importorskip("pandas")

    max_iter = 3
    n_dim = 5
    n_samples = 3
    estimator = MaxIterEstimator(max_iter=max_iter)
    callback = ScoringMonitor(on=on, scoring=scoring)
    estimator.set_callbacks(callback)
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)

    estimator.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    log_df = callback.get_logs(as_frame=True, select="most_recent")

    expected_log, scoring_names = make_expected_ouptput_MaxIterEstimator(
        max_iter, scoring, on, X_train, y_train, X_val, y_val
    )
    expected_log_df = pd.DataFrame(expected_log)
    expected_log_df = expected_log_df.set_index(
        [
            col
            for col in expected_log_df.columns
            if col not in scoring_names and col != "on"
        ]
    )
    assert np.array_equal(log_df.index.names, expected_log_df.index.names)
    assert log_df.equals(expected_log_df)


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_scoring_monitor_logged_values_meta_estimator(prefer, scoring, on):
    """Test that the correct values are logged with a meta-estimator.

    This test only looks at the dict outputs from `get_logs`.
    """
    n_outer = 3
    n_inner = 2
    max_iter = 4
    n_dim = 5
    n_samples = 3
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    callback = ScoringMonitor(on=on, scoring=scoring)
    est = MaxIterEstimator(max_iter=max_iter)
    est.set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )

    meta_est.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    log = callback.get_logs(as_frame=False, select="most_recent")

    expected_log, scoring_names = make_expected_ouptput_MetaEstimator(
        n_outer, n_inner, max_iter, scoring, on, X_train, y_train, X_val, y_val
    )

    assert len(log) == len(expected_log)
    for i in range(len(log)):
        assert set(log[i].keys()) == set(expected_log[i].keys())
        for key, val in log[i].items():
            # The log items might not be in the same order as the expected log because
            # of the parallelization.
            assert val in [expected_log[k][key] for k in range(len(log))]


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_scoring_monitor_logged_values_dataframe_meta_estimator(prefer, scoring, on):
    """Test that the correct values are logged with a meta-estimator.

    This test only looks at the pandas dataframe outputs from `get_logs`.
    """
    pd = pytest.importorskip("pandas")
    n_outer = 3
    n_inner = 2
    max_iter = 4
    n_dim = 5
    n_samples = 3
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    callback = ScoringMonitor(on=on, scoring=scoring)
    est = MaxIterEstimator(max_iter=max_iter)
    est.set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )

    meta_est.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    logs = callback.get_logs(as_frame=False, select="most_recent")

    expected_log_dict, scoring_names = make_expected_ouptput_MetaEstimator(
        n_outer, n_inner, max_iter, scoring, on, X_train, y_train, X_val, y_val
    )
    log_df = callback.get_logs(as_frame=True, select="most_recent")

    expected_log_df = pd.DataFrame(expected_log_dict)
    expected_log_df = expected_log_df.set_index(
        [
            col
            for col in expected_log_df.columns
            if col not in scoring_names and col != "on"
        ]
    )

    assert np.array_equal(log_df.index.names, expected_log_df.index.names)
    assert log_df.equals(expected_log_df)


@pytest.mark.parametrize(
    "as_frame",
    [False, "auto"],
)
def test_get_logs_output_type_no_pandas(as_frame):
    """Test the type of the get_logs when not explicitly asking for dataframes."""
    estimator = MaxIterEstimator()
    callback = ScoringMonitor(scoring="neg_mean_squared_error")
    estimator.set_callbacks(callback)

    empty_logs_all = callback.get_logs(select="all", as_frame=as_frame)
    assert isinstance(empty_logs_all, dict)
    assert not empty_logs_all

    empty_logs_most_recent = callback.get_logs(select="most_recent", as_frame=as_frame)

    estimator.fit()
    estimator.fit()

    logs_all = callback.get_logs(select="all", as_frame=as_frame)
    logs_most_recent = callback.get_logs(select="most_recent", as_frame=as_frame)

    assert isinstance(logs_all, dict)
    assert len(logs_all) == 2

    if find_spec("pandas") and as_frame == "auto":
        import pandas as pd

        assert isinstance(next(iter(logs_all.values())), pd.DataFrame)
        assert isinstance(empty_logs_most_recent, pd.DataFrame)
        assert empty_logs_most_recent.empty
        assert isinstance(logs_most_recent, pd.DataFrame)

    else:
        assert isinstance(empty_logs_most_recent, list)
        assert not empty_logs_most_recent
        assert isinstance(logs_most_recent, list)


def test_get_logs_output_type_pandas():
    """Test the type of the get_logs when explicitly asking for dataframes."""
    pd = pytest.importorskip("pandas")
    estimator = MaxIterEstimator()
    callback = ScoringMonitor(scoring="neg_mean_squared_error")
    estimator.set_callbacks(callback)

    empty_logs_all = callback.get_logs(select="all", as_frame=True)
    assert isinstance(empty_logs_all, dict)
    assert not empty_logs_all

    empty_logs_most_recent = callback.get_logs(select="most_recent", as_frame=True)
    assert isinstance(empty_logs_most_recent, pd.DataFrame)
    assert empty_logs_most_recent.empty

    estimator.fit()
    estimator.fit()

    logs_all = callback.get_logs(select="all", as_frame=True)
    assert isinstance(logs_all, dict)
    assert len(logs_all) == 2
    assert isinstance(next(iter(logs_all.values())), pd.DataFrame)

    logs_most_recent = callback.get_logs(select="most_recent", as_frame=True)
    assert isinstance(logs_most_recent, pd.DataFrame)


def test_estimator_without_optional_kwargs():
    """Smoke test when used on an estimator which does not provide optional kwargs.

    The callback should not crash when used on an estimator where `data` and
    `reconstruction_attributes` are not provided to `eval_on_fit_task_end`.
    """
    estimator = WhileEstimator()
    estimator.set_callbacks(ScoringMonitor(on="both", scoring="r2"))
    estimator.fit()
