# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from multiprocessing import Manager

import pandas as pd

from sklearn.callback._callback_context import _get_task_info_path


class MetricMonitor:
    """Callback that monitors a metric for each iterative steps of an estimator.

    The specified metric function is called on the target values `y` and the predicted
    values on the samples `y_pred = estimator.predict(X)` at each iterative step of the
    estimator.

    Parameters
    ----------
    metric : function
        The metric to compute.
    metric_kwargs : dict or None, default=None
        Keyword argumments for the metric, None means no argument is used.
    on_validation : bool, default=True
        Whether to compute the metric on validation data (if True) or training data
        (if False).
    """

    def __init__(self, metric, metric_params=None, on_validation=True):
        self.on_validation = on_validation
        self.metric_params = metric_params or dict()
        if metric_params is not None:
            valid_params = inspect.signature(metric).parameters
            invalid_params = [arg for arg in metric_params if arg not in valid_params]
            if invalid_params:
                raise ValueError(
                    f"The parameters '{invalid_params}' cannot be used with the"
                    f" function {metric.__module__}.{metric.__name__}."
                )
        self.metric_func = metric
        self._shared_log = Manager().list()

    def _on_fit_begin(self, estimator):
        if not hasattr(estimator, "predict"):
            raise ValueError(
                f"Estimator {estimator.__class__} does not have a predict method, which"
                " is necessary to use a MetricMonitor callback."
            )
        self._fit_log = Manager().Queue()

    def _on_fit_task_end(
        self, estimator, task_info, data, from_reconstruction_attributes, **kwargs
    ):
        # TODO: add check to verify we're on the innermost level of the fit loop
        # e.g. for the KMeans
        X, y = (
            (data["X_val"], data["y_val"])
            if self.on_validation
            else (data["X_train"], data["y_train"])
        )
        y_pred = from_reconstruction_attributes().predict(X)
        metric_value = self.metric_func(y, y_pred, **self.metric_params)
        log_item = {self.metric_func.__name__: metric_value}
        for node_info in _get_task_info_path(task_info):
            prev_task_str = (
                f"{node_info['prev_estimator_name']}_{node_info['prev_task_name']}|"
                if node_info["prev_estimator_name"] is not None
                else ""
            )
            log_item[
                f"{node_info['depth']}_{prev_task_str}{node_info['estimator_name']}_"
                f"{node_info['task_name']}"
            ] = node_info["task_id"]
        self._fit_log.put(log_item)

    def _on_fit_end(self, estimator, task_info):
        while not self._fit_log.empty():
            self._shared_log.append(self._fit_log.get())

    def get_logs(self):
        self.log = pd.DataFrame(list(self._shared_log))
        if not self.log.empty:
            self.log = self.log.set_index(
                [col for col in self.log.columns if col != self.metric_func.__name__]
            ).sort_index()
        return self.log.copy()
