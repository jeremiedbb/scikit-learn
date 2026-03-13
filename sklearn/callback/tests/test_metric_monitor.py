# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import Counter
from importlib.util import find_spec
from itertools import product

import numpy as np
import pytest

from sklearn.callback import MetricMonitor
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    MetaEstimator,
    WhileEstimator,
)
from sklearn.metrics import check_scoring
from sklearn.utils._param_validation import InvalidParameterError


def make_expected_ouptput_MaxIterEstimator(
    max_iter, metric, on, X_train, y_train, X_val, y_val
):
    """Generate the expected dict output of a MetricMonitor on a MaxIterEstimator."""
    scorer = check_scoring(None, metric)
    if isinstance(metric, str):
        metric_names = [metric]
        expected_log_dict = {metric: []}
    elif callable(metric):
        metric_names = ["metric"]
        expected_log_dict = {"metric": []}
    else:
        metric_names = list(scorer._scorers.keys())
        expected_log_dict = {name: [] for name in metric_names}

    expected_log_dict.update(
        {
            "on": [],
            f"0_{MaxIterEstimator.__name__}_fit": [],
            f"1_{MaxIterEstimator.__name__}_iteration": [],
        }
    )

    for i in range(max_iter):
        fitted_est = MaxIterEstimator(max_iter=i + 1).fit()
        if on in ("train_set", "both"):
            expected_log_dict[f"0_{MaxIterEstimator.__name__}_fit"].append(0)
            expected_log_dict[f"1_{MaxIterEstimator.__name__}_iteration"].append(i)
            expected_log_dict["on"].append("train_set")
            score = scorer(fitted_est, X_train, y_train)
            if isinstance(score, dict):
                for key, val in score.items():
                    expected_log_dict[key].append(val)
            else:
                expected_log_dict[metric_names[0]].append(score)

        if on in ("validation_set", "both"):
            expected_log_dict[f"0_{MaxIterEstimator.__name__}_fit"].append(0)
            expected_log_dict[f"1_{MaxIterEstimator.__name__}_iteration"].append(i)
            expected_log_dict["on"].append("validation_set")
            score = scorer(fitted_est, X_val, y_val)
            if isinstance(score, dict):
                for key, val in score.items():
                    expected_log_dict[key].append(val)
            else:
                expected_log_dict[metric_names[0]].append(score)

    return expected_log_dict, metric_names


def make_expected_ouptput_MetaEstimator(
    n_outer, n_inner, max_iter, metric, on, X_train, y_train, X_val, y_val
):
    """Generate the expected dict output of a MetricMonitor on a MetaEstimator.

    The sub-estimators are expected to be MaxIterEstimator.
    """
    scorer = check_scoring(None, metric)
    if isinstance(metric, str):
        metric_names = [metric]
        expected_log_dict = {metric: []}
    elif callable(metric):
        metric_names = ["metric"]
        expected_log_dict = {"metric": []}
    else:
        metric_names = list(scorer._scorers.keys())
        expected_log_dict = {name: [] for name in metric_names}

    expected_log_dict.update(
        {
            "on": [],
            f"0_{MetaEstimator.__name__}_fit": [],
            f"1_{MetaEstimator.__name__}_outer": [],
            f"2_{MetaEstimator.__name__}_inner|{MaxIterEstimator.__name__}_fit": [],
            f"3_{MaxIterEstimator.__name__}_iteration": [],
        }
    )

    for i_outer, i_inner in product(range(n_outer), range(n_inner)):
        for i_estimator_iteration in range(max_iter):
            fitted_est = MaxIterEstimator(max_iter=i_estimator_iteration + 1).fit()
            if on in ("train_set", "both"):
                expected_log_dict["on"].append("train_set")
                expected_log_dict[f"0_{MetaEstimator.__name__}_fit"].append(0)
                expected_log_dict[f"1_{MetaEstimator.__name__}_outer"].append(i_outer)
                expected_log_dict[
                    f"2_{MetaEstimator.__name__}_inner|{MaxIterEstimator.__name__}_fit"
                ].append(i_inner)
                expected_log_dict[f"3_{MaxIterEstimator.__name__}_iteration"].append(
                    i_estimator_iteration
                )
                score = scorer(fitted_est, X_train, y_train)
                if isinstance(score, dict):
                    for key, val in score.items():
                        expected_log_dict[key].append(val)
                else:
                    expected_log_dict[metric_names[0]].append(score)

            if on in ("validation_set", "both"):
                expected_log_dict["on"].append("validation_set")
                expected_log_dict[f"0_{MetaEstimator.__name__}_fit"].append(0)
                expected_log_dict[f"1_{MetaEstimator.__name__}_outer"].append(i_outer)
                expected_log_dict[
                    f"2_{MetaEstimator.__name__}_inner|{MaxIterEstimator.__name__}_fit"
                ].append(i_inner)
                expected_log_dict[f"3_{MaxIterEstimator.__name__}_iteration"].append(
                    i_estimator_iteration
                )
                score = scorer(fitted_est, X_val, y_val)
                if isinstance(score, dict):
                    for key, val in score.items():
                        expected_log_dict[key].append(val)
                else:
                    expected_log_dict[metric_names[0]].append(score)

    return expected_log_dict, metric_names


def custom_metric(estimator, X, y):
    """Custom metric to test the MetricMonitor with a callable."""
    return 0


@pytest.mark.parametrize(
    "metric",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_metric],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_metric_monitor_logged_values_dict(metric, on):
    """Test that the correct values are logged with a simple estimator.

    This test only looks at the dict outputs from `get_logs`.
    """
    max_iter = 3
    n_dim = 5
    n_samples = 3
    estimator = MaxIterEstimator(max_iter=max_iter)
    callback = MetricMonitor(on=on, metric=metric)
    estimator.set_callbacks(callback)
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)

    estimator.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    logs = callback.get_logs(as_frame=False, select="most_recent")

    expected_log_dict, metric_names = make_expected_ouptput_MaxIterEstimator(
        max_iter, metric, on, X_train, y_train, X_val, y_val
    )
    assert set(logs.keys()) == set(expected_log_dict.keys())
    for key, val in logs.items():
        assert val == expected_log_dict[key]


@pytest.mark.parametrize(
    "metric",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_metric],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_metric_monitor_logged_values_dataframe(metric, on):
    """Test that the correct values are logged with a simple estimator.

    This test only looks at the pandas dataframe outputs from `get_logs`.
    """
    pd = pytest.importorskip("pandas")

    max_iter = 3
    n_dim = 5
    n_samples = 3
    estimator = MaxIterEstimator(max_iter=max_iter)
    callback = MetricMonitor(on=on, metric=metric)
    estimator.set_callbacks(callback)
    rng = np.random.RandomState(0)
    X_train, y_train = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)
    X_val, y_val = rng.uniform(size=(n_dim, n_samples)), rng.uniform(size=n_dim)

    estimator.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    log_df = callback.get_logs(as_frame=True, select="most_recent")

    expected_log_dict, metric_names = make_expected_ouptput_MaxIterEstimator(
        max_iter, metric, on, X_train, y_train, X_val, y_val
    )
    expected_log_df = pd.DataFrame(expected_log_dict)
    expected_log_df = expected_log_df.set_index(
        [
            col
            for col in expected_log_df.columns
            if col not in metric_names and col != "on"
        ]
    )
    assert np.array_equal(log_df.index.names, expected_log_df.index.names)
    assert log_df.equals(expected_log_df)


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "metric",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_metric],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_metric_monitor_logged_values_dict_meta_estimator(prefer, metric, on):
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
    callback = MetricMonitor(on=on, metric=metric)
    est = MaxIterEstimator(max_iter=max_iter)
    est.set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )

    meta_est.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    logs = callback.get_logs(as_frame=False, select="most_recent")

    expected_log_dict, metric_names = make_expected_ouptput_MetaEstimator(
        n_outer, n_inner, max_iter, metric, on, X_train, y_train, X_val, y_val
    )
    logs = callback.get_logs(as_frame=False, select="most_recent")
    assert set(logs.keys()) == set(expected_log_dict.keys())
    for key, val in logs.items():
        # Verify list equality up to a permutation because the parallelization
        # of the meta-est can change the logging order.
        assert Counter(val) == Counter(expected_log_dict[key])


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "metric",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_metric],
)
@pytest.mark.parametrize(
    "on",
    ["train_set", "validation_set", "both"],
)
def test_metric_monitor_logged_values_dataframe_meta_estimator(prefer, metric, on):
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
    callback = MetricMonitor(on=on, metric=metric)
    est = MaxIterEstimator(max_iter=max_iter)
    est.set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )

    meta_est.fit(X=X_train, y=y_train, X_val=X_val, y_val=y_val)
    logs = callback.get_logs(as_frame=False, select="most_recent")

    expected_log_dict, metric_names = make_expected_ouptput_MetaEstimator(
        n_outer, n_inner, max_iter, metric, on, X_train, y_train, X_val, y_val
    )
    log_df = callback.get_logs(as_frame=True, select="most_recent")

    expected_log_df = pd.DataFrame(expected_log_dict)
    expected_log_df = expected_log_df.set_index(
        [
            col
            for col in expected_log_df.columns
            if col not in metric_names and col != "on"
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
    callback = MetricMonitor(metric="neg_mean_squared_error")
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
        assert isinstance(empty_logs_most_recent, dict)
        assert not empty_logs_most_recent
        assert isinstance(logs_most_recent, dict)


def test_get_logs_output_type_pandas():
    """Test the type of the get_logs when explicitly asking for dataframes."""
    pd = pytest.importorskip("pandas")
    estimator = MaxIterEstimator()
    callback = MetricMonitor(metric="neg_mean_squared_error")
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


def test_signature_check_error():
    def wrong_signature_metric(y, y_pred):
        return None

    with pytest.raises(
        InvalidParameterError,
        match=f"If the 'metric' parameter of {MetricMonitor.__name__} is a callable,",
    ):
        MetricMonitor(metric=wrong_signature_metric)


def test_estimator_without_optional_kwargs():
    """Smoke test when used on an estimator which does not provide optional kwargs.

    The callback should not crash when used on an estimator where `data` and
    `reconstruction_attributes` are not provided to `eval_on_fit_task_end`.
    """
    estimator = WhileEstimator()
    estimator.set_callbacks(MetricMonitor(on="both", metric="r2"))
    estimator.fit()
