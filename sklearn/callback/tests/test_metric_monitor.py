# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest

from sklearn.callback import MetricMonitor
from sklearn.callback.tests._utils import Estimator, EstimatorWithoutPredict
from sklearn.metrics import mean_pinball_loss


def test_metric_monitor():
    max_iter = 10
    n_dim = 5
    n_samples = 3
    alpha = 0.6
    estimator = Estimator(max_iter=max_iter)
    callback = MetricMonitor(mean_pinball_loss, metric_kwargs={"alpha": alpha})
    estimator.set_callbacks(callback)
    X, y = np.ones((n_dim, n_samples)), np.ones(n_dim)

    estimator.fit(X, y)

    expected_log = [
        (i, mean_pinball_loss(y, X.mean(axis=1) * (i + 1), alpha=alpha))
        for i in range(max_iter)
    ]

    assert np.array_equal(callback.log, expected_log)


def test_no_predict_error():
    estimator = EstimatorWithoutPredict()
    callback = MetricMonitor(mean_pinball_loss, metric_kwargs={"alpha": 0.6})
    estimator.set_callbacks(callback)

    with pytest.raises(ValueError, match="does not have a predict method"):
        estimator.fit()
