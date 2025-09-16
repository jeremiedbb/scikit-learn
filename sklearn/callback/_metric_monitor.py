# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from functools import partial


class MetricMonitor:
    """Callback that monitors a metric for each iterative steps of an estimator.

    The specified metric function is called on the target values `y` and the predicted
    values on the samples `y_pred = estimator.predict(X)` at each iterative step of the
    estimator.

    Parameters
    ----------
    metric : function
        The metric to compute.
    metric_kwargs : dict
        Keyword argumments for the metric.
    """

    def __init__(self, metric, metric_kwargs):
        valid_params = inspect.signature(metric).parameters
        invalid_params = [arg for arg in metric_kwargs if arg not in valid_params]
        if invalid_params:
            raise ValueError(
                f"The parameters '{invalid_params}' cannot be used with the function"
                f"{metric.__module__}.{metric.__name__}."
            )
        self.metric_func = partial(metric, **metric_kwargs)
        self.log = []

    def _on_fit_begin(self, estimator, *, data):
        if not hasattr(estimator, "predict"):
            raise ValueError(
                f"Estimator {estimator.__class__} does not have a predict method, which"
                "is necessary to use a MetricMonitor callback."
            )

    def _on_fit_iter_end(
        self, estimator, task_info, data, from_reconstruction_attributes, **kwargs
    ):
        y_pred = from_reconstruction_attributes().predict(data["X_train"])
        metric_value = self.metric_func(data["y_train"], y_pred)
        self.log.append((task_info["task_id"], metric_value))

    def _on_fit_end(self, estimator, task_info):
        pass
