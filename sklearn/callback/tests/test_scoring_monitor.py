# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re
from itertools import product

import numpy as np
import pytest

from sklearn import config_context
from sklearn.callback import ScoringMonitor
from sklearn.callback.tests._utils import (
    MaxIterEstimator,
    MetaEstimator,
    WhileEstimator,
)
from sklearn.datasets import make_regression
from sklearn.metrics import check_scoring, make_scorer, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils._metadata_requests import UnsetMetadataPassedError


def _get_scorer_and_names(scoring):
    scorer = check_scoring(None, scoring)
    if isinstance(scoring, str):
        score_names = [scoring]
    elif callable(scoring):
        score_names = ["score"]
    else:  # multi-scorer
        score_names = list(scorer._scorers.keys())
    return scorer, score_names


def _make_dataframe(log, score_names):
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(log)
    df = df.set_index(
        [col for col in df.columns if col not in score_names + ["eval_on"]]
    )
    return df


def _make_expected_output_MaxIterEstimator(
    max_iter, scoring, as_frame, X_train, y_train, X_val, y_val
):
    """Generate the expected output of a ScoringMonitor on a MaxIterEstimator."""
    scorer, score_names = _get_scorer_and_names(scoring)
    est_name = MaxIterEstimator.__name__

    expected_log = []
    for i in range(max_iter):
        fitted_est = MaxIterEstimator(max_iter=i + 1).fit()

        for eval_on in ("train", "val"):
            log_item = {
                f"0_{est_name}_fit": 0,
                f"1_{est_name}_iteration": i,
                "eval_on": eval_on,
            }

            X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
            scores = scorer(fitted_est, X, y)
            if isinstance(scores, dict):
                log_item.update(scores)
            else:
                log_item[score_names[0]] = scores
            expected_log.append(log_item)

    if as_frame:
        expected_log = _make_dataframe(expected_log, score_names)

    return expected_log, score_names


def _make_expected_output_MetaEstimator(
    n_outer, n_inner, max_iter, scoring, as_frame, X_train, y_train, X_val, y_val
):
    """Generate the expected output of a ScoringMonitor on a MetaEstimator.

    The sub-estimators are expected to be MaxIterEstimator.
    """
    scorer, score_names = _get_scorer_and_names(scoring)
    meta_est_name = MetaEstimator.__name__
    sub_est_name = MaxIterEstimator.__name__

    expected_log = []
    for i_outer, i_inner in product(range(n_outer), range(n_inner)):
        for i_estimator_iteration in range(max_iter):
            fitted_est = MaxIterEstimator(max_iter=i_estimator_iteration + 1).fit()

            for eval_on in ("train", "val"):
                log_item = {
                    "eval_on": eval_on,
                    f"0_{meta_est_name}_fit": 0,
                    f"1_{meta_est_name}_outer": i_outer,
                    f"2_{meta_est_name}_inner|{sub_est_name}_fit": i_inner,
                    f"3_{sub_est_name}_iteration": i_estimator_iteration,
                }

                X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
                scores = scorer(fitted_est, X, y)
                if isinstance(scores, dict):
                    log_item.update(scores)
                else:
                    log_item[score_names[0]] = scores
                expected_log.append(log_item)

    if as_frame:
        expected_log = _make_dataframe(expected_log, score_names)

    return expected_log, score_names


def custom_score(estimator, X, y):
    """Custom score to test the ScoringMonitor with a callable."""
    return 0


@pytest.mark.parametrize("eval_on", ["train", "val", "both"])
def test_eval_on(eval_on):
    max_iter = 3
    callback = ScoringMonitor(eval_on=eval_on, scoring="r2")
    estimator = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    estimator.fit(X=X, y=y, X_val=X_val, y_val=y_val)
    log = callback.get_logs(as_frame=False, select="most_recent")

    if eval_on in ("train", "val"):
        assert len(log) == max_iter
        assert all(entry["eval_on"] == eval_on for entry in log)
    else:  # eval_on == "both"
        train_log = [entry for entry in log if entry["eval_on"] == "train"]
        val_log = [entry for entry in log if entry["eval_on"] == "val"]
        assert len(train_log) == len(val_log) == max_iter


@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_logged_values(scoring, as_frame):
    """Test that the correct values are logged with a simple estimator."""
    if as_frame:
        pytest.importorskip("pandas")

    max_iter = 3
    callback = ScoringMonitor(eval_on="both", scoring=scoring)
    estimator = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    estimator.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    log = callback.get_logs(as_frame=as_frame, select="most_recent")
    expected_log, _ = _make_expected_output_MaxIterEstimator(
        max_iter, scoring, as_frame, X, y, X_val, y_val
    )

    if as_frame:
        assert np.array_equal(log.index.names, expected_log.index.names)
        assert log.equals(expected_log)
    else:
        assert all(
            entry == expected_entry for entry, expected_entry in zip(log, expected_log)
        )


@pytest.mark.parametrize("prefer", ["processes", "threads"])
@pytest.mark.parametrize(
    "scoring",
    ["neg_mean_squared_error", ("neg_mean_squared_error", "r2"), custom_score],
)
@pytest.mark.parametrize("as_frame", [True, False])
def test_logged_values_meta_estimator(prefer, scoring, as_frame):
    """Test that the correct values are logged with a meta-estimator."""
    if as_frame:
        pytest.importorskip("pandas")

    n_outer, n_inner, max_iter = 3, 2, 5
    callback = ScoringMonitor(eval_on="both", scoring=scoring)
    est = MaxIterEstimator(max_iter=max_iter).set_callbacks(callback)
    meta_est = MetaEstimator(
        est, n_outer=n_outer, n_inner=n_inner, n_jobs=2, prefer=prefer
    )
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    with config_context(enable_metadata_routing=True):
        est.set_fit_request(X_val=True, y_val=True)
        meta_est.fit(X=X, y=y, X_val=X_val, y_val=y_val)

    log = callback.get_logs(as_frame=as_frame, select="most_recent")
    expected_log, score_names = _make_expected_output_MetaEstimator(
        n_outer, n_inner, max_iter, scoring, as_frame, X, y, X_val, y_val
    )

    if as_frame:
        assert np.array_equal(log.index.names, expected_log.index.names)
        assert log.equals(expected_log)
    else:
        # The log items might not be in the same order as the expected log due to the
        # parallelization of the meta-estimator. Hence we sort the log by the values of
        # all keys but "eval_on" and the score names before comparing.
        keys = tuple(log[0].keys() - {"eval_on"} - set(score_names))
        sorted_log = sorted(log, key=lambda x: tuple(str(x[k]) for k in keys))
        sorted_expected_log = sorted(
            expected_log, key=lambda x: tuple(str(x[k]) for k in keys)
        )
        assert all(
            entry == expected_entry
            for entry, expected_entry in zip(sorted_log, sorted_expected_log)
        )


@pytest.mark.parametrize("as_frame", [True, False])
def test_get_logs_output_type_no_fit(as_frame):
    """Check that get_logs return empty containers of the right type before fit."""
    if as_frame:
        pd = pytest.importorskip("pandas")

    callback = ScoringMonitor(scoring="neg_mean_squared_error")

    # "all" logs is always a dict indexed by run ids.
    logs_all = callback.get_logs(select="all", as_frame=as_frame)
    assert isinstance(logs_all, dict)
    assert len(logs_all) == 0

    log_most_recent = callback.get_logs(select="most_recent", as_frame=as_frame)
    expected_type = list if not as_frame else pd.DataFrame
    assert isinstance(log_most_recent, expected_type)
    assert len(log_most_recent) == 0


@pytest.mark.parametrize("as_frame", [True, False])
def test_get_logs_output_type(as_frame):
    """Test the type of the get_logs output."""
    if as_frame:
        pd = pytest.importorskip("pandas")

    callback = ScoringMonitor(scoring="neg_mean_squared_error")
    estimator = MaxIterEstimator().set_callbacks(callback)
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)

    estimator.fit(X, y)
    estimator.fit(X, y)

    logs_all = callback.get_logs(select="all", as_frame=as_frame)
    assert isinstance(logs_all, dict)
    assert len(logs_all) == 2
    expected_type = list if not as_frame else pd.DataFrame
    assert all(isinstance(log, expected_type) for log in logs_all.values())

    log_most_recent = callback.get_logs(select="most_recent", as_frame=as_frame)
    assert isinstance(log_most_recent, expected_type)


def test_estimator_without_reconstruction_attributes():
    """Smoke test on an estimator which does not provide reconstruction_attributes."""
    callback = ScoringMonitor(eval_on="both", scoring="r2")
    WhileEstimator().set_callbacks(callback).fit()
    assert len(callback.get_logs()) == 0


def test_sample_weights_and_metadata_routing():
    """Check that the ScoringMonitor works with sample weights and metadata-routing.

    - passing sample weights results in a different score than not passing them.
    - passing sample weights without metadata-routing enabled gives the same scores as
      passing them with metadata-routing enabled.
    - Not requesting sample weights gives an error if metadata-routing is enabled.
    """
    n_samples = 100
    X, y = make_regression(n_samples=n_samples, n_features=2, random_state=0)
    sample_weight = np.random.randint(0, 5, size=n_samples)

    # no sample weights
    callback = ScoringMonitor(eval_on="train", scoring="r2")
    MaxIterEstimator().set_callbacks(callback).fit(X=X, y=y)
    log_no_sw = callback.get_logs(as_frame=False, select="most_recent")

    # sample weights, no metadata-routing
    callback = ScoringMonitor(eval_on="train", scoring="r2")
    MaxIterEstimator().set_callbacks(callback).fit(
        X=X, y=y, sample_weight=sample_weight
    )
    log_sw_no_mr = callback.get_logs(as_frame=False, select="most_recent")

    # sample weights, metadata-routing
    with config_context(enable_metadata_routing=True):
        scorer = make_scorer(r2_score)
        scorer.set_score_request(sample_weight=True)
        callback = ScoringMonitor(eval_on="train", scoring={"r2": scorer})
        MaxIterEstimator().set_callbacks(callback).fit(
            X=X, y=y, sample_weight=sample_weight
        )
        log_sw_mr = callback.get_logs(as_frame=False, select="most_recent")

        # error if sample_weight not requested
        scorer = make_scorer(r2_score)
        callback = ScoringMonitor(eval_on="train", scoring={"r2": scorer})
        est = MaxIterEstimator().set_callbacks(callback)
        with pytest.raises(
            TypeError,
            match=re.escape("score got unexpected argument(s) {'sample_weight'}"),
        ):
            est.fit(X=X, y=y, sample_weight=sample_weight)

    assert any(
        np.logical_not(np.isclose(l1["r2"], l2["r2"]))
        for l1, l2 in zip(log_no_sw, log_sw_no_mr)
    )
    assert all(
        np.isclose(l1["r2"], l2["r2"]) for l1, l2 in zip(log_sw_no_mr, log_sw_mr)
    )


def test_validation_set_metadata_routing():
    """Integration test for metadata-routing on the validation set.

    X_val and y_val must be requested for the MaxIterEstimator to be able to use them.
    """
    X, y = make_regression(n_samples=100, n_features=2, random_state=0)
    X, X_val, y, y_val = train_test_split(X, y, test_size=0.2, random_state=0)

    callback = ScoringMonitor(eval_on="both", scoring="r2")
    est = MaxIterEstimator(max_iter=10).set_callbacks(callback)

    # Without metadata-routing enabled, passing X_val and y_val gives an error
    msg = re.escape(
        "[X_val, y_val] are passed but are not explicitly set as requested or not "
        "requested for MaxIterEstimator.fit"
    )
    with pytest.raises(UnsetMetadataPassedError, match=msg):
        MetaEstimator(est).fit(X=X, y=y, X_val=X_val, y_val=y_val)

    with config_context(enable_metadata_routing=True):
        # passing X_val and y_val without requesting them gives the same error
        with pytest.raises(UnsetMetadataPassedError, match=msg):
            MetaEstimator(est).fit(X=X, y=y, X_val=X_val, y_val=y_val)

        # with metadata-routing enabled and requested
        est.set_fit_request(X_val=True, y_val=True)
        MetaEstimator(est, n_outer=2, n_inner=3).fit(X=X, y=y, X_val=X_val, y_val=y_val)
        log = callback.get_logs(as_frame=False, select="most_recent")

        log_train = [entry for entry in log if entry["eval_on"] == "train"]
        log_val = [entry for entry in log if entry["eval_on"] == "val"]

        # 2 * 3 MetaEstimator iterations, 10 MaxIterEstimator iterations
        assert len(log_train) == len(log_val) == 2 * 3 * 10

        # The scores on the train and validation sets should be different
        assert any(
            train_entry["r2"] != val_entry["r2"]
            for train_entry, val_entry in zip(log_train, log_val)
        )
