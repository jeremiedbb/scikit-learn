# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import re

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


def _make_dataframe(log):
    pd = pytest.importorskip("pandas")
    return pd.DataFrame(log)


def _make_expected_output_MaxIterEstimator(
    max_iter, scoring, as_frame, X_train, y_train, X_val, y_val
):
    """Generate the expected output of a ScoringMonitor on a MaxIterEstimator."""
    scorer, score_names = _get_scorer_and_names(scoring)
    est_name = MaxIterEstimator.__name__

    expected_log = {
        (0,): {"values": [], "task_path": (f"{est_name} fit",)},
        tuple(): {"values": [], "task_path": tuple()},
    }
    # fit loop iterations
    for i in range(max_iter):
        fitted_est = MaxIterEstimator(max_iter=i + 1).fit()

        for eval_on in ("train", "val"):
            log_item = {
                "task_name": "iteration",
                "task_id": i,
                "estimator_name": est_name,
                "eval_on": eval_on,
            }

            X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
            scores = scorer(fitted_est, X, y)
            if isinstance(scores, dict):
                log_item.update(scores)
            else:
                log_item[score_names[0]] = scores
            expected_log[(0,)]["values"].append(log_item)

    # fit root task
    for eval_on in ("train", "val"):
        log_item = {
            "task_name": "fit",
            "task_id": 0,
            "estimator_name": est_name,
            "eval_on": eval_on,
        }
        X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
        scores = scorer(fitted_est, X, y)
        if isinstance(scores, dict):
            log_item.update(scores)
        else:
            log_item[score_names[0]] = scores
        expected_log[tuple()]["values"].append(log_item)

    if as_frame:
        for key in expected_log:
            expected_log[key]["values"] = _make_dataframe(expected_log[key]["values"])

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

    expected_log = {}
    for i_outer in range(n_outer):
        expected_log[(0, i_outer)] = {
            "values": [],
            "task_path": (f"{meta_est_name} fit", f"{meta_est_name} outer"),
        }
        for i_inner in range(n_inner):
            expected_log[(0, i_outer, i_inner)] = {
                "values": [],
                "task_path": (
                    f"{meta_est_name} fit",
                    f"{meta_est_name} outer",
                    f"{meta_est_name} inner | {sub_est_name} fit",
                ),
            }
            for i_estimator_iteration in range(max_iter):
                fitted_est = MaxIterEstimator(max_iter=i_estimator_iteration + 1).fit()

                for eval_on in ("train", "val"):
                    log_item = {
                        "task_name": "iteration",
                        "task_id": i_estimator_iteration,
                        "estimator_name": sub_est_name,
                        "eval_on": eval_on,
                    }

                    X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
                    scores = scorer(fitted_est, X, y)
                    if isinstance(scores, dict):
                        log_item.update(scores)
                    else:
                        log_item[score_names[0]] = scores
                    expected_log[(0, i_outer, i_inner)]["values"].append(log_item)

            for eval_on in ("train", "val"):
                log_item = {
                    "task_name": "fit",
                    "task_id": i_inner,
                    "estimator_name": sub_est_name,
                    "eval_on": eval_on,
                }

                X, y = (X_train, y_train) if eval_on == "train" else (X_val, y_val)
                scores = scorer(fitted_est, X, y)
                if isinstance(scores, dict):
                    log_item.update(scores)
                else:
                    log_item[score_names[0]] = scores
                expected_log[(0, i_outer)]["values"].append(log_item)

    if as_frame:
        for key in expected_log:
            expected_log[key]["values"] = _make_dataframe(expected_log[key]["values"])

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
    log = callback.get_logs(as_frame=False, select="most_recent")["logs"][(0,)][
        "values"
    ]

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

    for task_id, task_log in log["logs"].items():
        assert task_log["task_path"] == expected_log[task_id]["task_path"]
        log_values = task_log["values"]
        expected_values = expected_log[task_id]["values"]
        if as_frame:
            assert log_values.equals(expected_values)
        else:
            assert all(
                entry == expected_entry
                for entry, expected_entry in zip(log_values, expected_values)
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

    for task_id, task_log in log["logs"].items():
        assert task_log["task_path"] == expected_log[task_id]["task_path"]
        log_values = task_log["values"]
        expected_values = expected_log[task_id]["values"]
        # The log items might not be in the same order as the expected log due to the
        # parallelization of the meta-estimator. Hence we sort / set_index the log by
        # task_id.
        if as_frame:
            assert log_values.set_index("task_id").equals(
                expected_values.set_index("task_id")
            )
        else:
            sorted_log_values = sorted(log_values, key=lambda x: x["task_id"])
            sorted_expected_values = sorted(expected_values, key=lambda x: x["task_id"])
            assert all(
                entry == expected_entry
                for entry, expected_entry in zip(
                    sorted_log_values, sorted_expected_values
                )
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
    assert isinstance(log_most_recent, dict)
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
    assert all(isinstance(log, dict) for log in logs_all.values())

    log_most_recent = callback.get_logs(select="most_recent", as_frame=as_frame)
    assert isinstance(log_most_recent, dict)


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

    for task_id in log_no_sw["logs"]:
        assert any(
            np.logical_not(np.isclose(l1["r2"], l2["r2"]))
            for l1, l2 in zip(
                log_no_sw["logs"][task_id]["values"],
                log_sw_no_mr["logs"][task_id]["values"],
            )
        )
        assert all(
            np.isclose(l1["r2"], l2["r2"])
            for l1, l2 in zip(
                log_sw_no_mr["logs"][task_id]["values"],
                log_sw_mr["logs"][task_id]["values"],
            )
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


# TODO: add test for task_tree and info in logs
