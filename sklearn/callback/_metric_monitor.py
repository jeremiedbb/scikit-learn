# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import inspect
from collections import defaultdict

from sklearn.callback._callback_context import get_context_path
from sklearn.callback._callback_support import get_callback_manager
from sklearn.metrics import check_scoring
from sklearn.utils._optional_dependencies import check_pandas_support
from sklearn.utils._param_validation import (
    InvalidParameterError,
    StrOptions,
    validate_params,
)


class MetricMonitor:
    """Callback that monitors a metric for each iterative steps of an estimator.

    The specified metric function is called on the target values `y` and the predicted
    values on the samples `y_pred = estimator.predict(X)` at each iterative step of the
    estimator.

    Parameters
    ----------
    on : {"train_set", "validation_set", "both"}, default="train_set"
        Which data to compue the metric on. Possible values are "train_set",
        "validation_set" and "both". "train_set" corresponds to using the X and y
        arguments of the fit function, "validation_set" corresponds to using the X_val
        and y_val arguments, "both" corresponds to using both.

    metric : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the metric on the model.

        If `metric` is a callale, it must have the signature : `metric(estimator, X, y)`
        and return a single value. Scikit-learn's metric functions (such as
        sklearn.metrics.mean_squared_error) have a signature of the form
        `metric(y_true, y_pred, **kwargs)` and cannot be used directly. A callable with
        the right signature can be generated from such a metric using the
        `sklearn.metrics.make_scorer` function.

        If `metric` is a string, the scorer with the corresponding name is used, see
        :ref:`scoring_string_names`.

        If `metric` is a list or tuple of strings, or a dictionary with metric names as
        keys and callables as values; then a multimetric scorer is used.

        If `metric` is `None`, the `estimator`'s :ref:`default evaluation criterion
          <scoring_api_overview>` is used.
    """

    requested_fit_info = ["fitted_estimator"]

    @validate_params(
        {"on": [StrOptions({"train_set", "validation_set", "both"})]},
        prefer_skip_nested_validation=True,
    )
    def __init__(self, *, on="train_set", metric):
        self.on = on
        if callable(metric):
            signature = tuple(
                p.name for p in inspect.signature(metric).parameters.values()
            )
            required_signature = ("estimator", "X", "y")
            if signature != required_signature:
                raise InvalidParameterError(
                    f"If the 'metric' parameter of {self.__class__.__name__} is a "
                    f"callable, its signature must be {required_signature}. Got "
                    f"{signature} instead."
                )
        if metric is not None:
            self.scorer = check_scoring(None, metric)
        self.metric = metric
        self._shared_log = get_callback_manager().list()

    def on_fit_begin(self, estimator):
        if self.metric is None:
            self.scorer = check_scoring(estimator, self.metric)

    def on_fit_task_end(
        self, estimator, context, *, data=None, fitted_estimator=None, **kwargs
    ):
        # TODO: add a task_info dict in the logs
        if fitted_estimator is None or data is None:
            return
        context_path = get_context_path(context)
        if self.on in ("train_set", "both"):
            X, y = None, None
            if "X_train" in data and "y_train" in data:
                X, y = data["X_train"], data["y_train"]
            self._add_log_entry(X, y, "train_set", fitted_estimator, context_path)
        if self.on == "validation_set" or self.on == "both":
            X, y = None, None
            if "X_val" in data and "y_val" in data:
                X, y = data["X_val"], data["y_val"]
            self._add_log_entry(X, y, "validation_set", fitted_estimator, context_path)

    def _add_log_entry(self, X, y, on, fitted_estimator, context_path):
        if X is not None and y is not None:
            metric_value = self.scorer(fitted_estimator, X, y)
        else:
            metric_value = None

        if isinstance(metric_value, dict):
            log_item = metric_value
        else:
            metric_name = self.metric if isinstance(self.metric, str) else "metric"
            log_item = {metric_name: metric_value}
        log_item["on"] = on
        for depth, ctx in enumerate(context_path):
            if depth == 0:
                timestamp = ctx.init_time.strftime("UTC%Y-%m-%d-%H:%M:%S.%f")
                run_id = f"{ctx.estimator_name}_{timestamp}_{ctx.root_uuid}"
            prev_task_str = (
                f"{ctx.source_estimator_name}_{ctx.source_task_name}|"
                if ctx.source_estimator_name is not None
                else ""
            )
            # The prefix __index__ is used to identify columns that will be used as
            # index in the mulit_index dataframe returned by get_logs.
            log_item[
                f"__index__{depth}_{prev_task_str}{ctx.estimator_name}_{ctx.task_name}"
            ] = ctx.task_id

        self._shared_log.append((run_id, log_item))

    def on_fit_end(self, estimator, context):
        pass

    @validate_params(
        {
            "select": [StrOptions({"all", "most_recent"})],
            "as_frame": [StrOptions({"auto"}), "boolean"],
        },
        prefer_skip_nested_validation=True,
    )
    def get_logs(self, select="all", as_frame="auto"):
        """Get the logged values.

        Returns the logs. If select is "all", a dictionary is returned with run ids as
        keys and logs as values. The logs take the form of pandas DataFrames or
        dictionaries depending on the `as_frame` parameter.

        The run ids are strings of the form : "<estimator name>_<timestamp>_<id>".

        Parameters
        ----------
        select : {"all", "most_recent"}, default="all"
            Which log run to return.

            If `select` is "all", all runs are returned in a dictionary, and the
            dictionary is empty if there are no logs.

            If `select` is "most_recent", only the log from the last run is directly
            returned, and if there are no logs, an empty dictionary or DataFrame is
            returned.

        as_frame : "auto" or bool, default="auto"
            Whether to have the logs (the items of the dict if `select` is "all",
            otherwise the output) formatted as multi-index pandas DataFrames. If set to
            False the logs are formatted as dictionaries instead. If set to "auto", the
            avialbility of pandas is evaluated and the format is chosen accordingly.

        Returns
        -------
        logs : dict or pandas DataFrame
            The logged values, formatted as :

            - a dict of pandas dataframes if `select` is "all" and `as_frame` is True or
              "auto" with pandas available.

            - a pandas dataframe is `select` is "most_recent" and `as_frame` is True or
              "auto" with pandas available.

            - a dict of dict if `select` is "all" and `as_frame` is False or "auto" with
              pandas unavailable.

            - a dict is `select` is "most_recent" and `as_frame` is False or "auto" with
              pandas unavailable.
        """
        log_item_list = list(self._shared_log)

        index_prefix = "__index__"
        logs_dict = defaultdict(lambda: defaultdict(list))
        index_names = set()
        for run_id, log_item in log_item_list:
            for key, val in log_item.items():
                if key.startswith(index_prefix):
                    key = key[len(index_prefix) :]
                    index_names.add(key)
                logs_dict[run_id][key].append(val)

        if select == "most_recent" and logs_dict:
            run_ids = list(logs_dict.keys())
            run_timetsamps = [r.split("_")[-2] for r in run_ids]
            most_recent_id = run_ids[run_timetsamps.index(max(run_timetsamps))]
            logs_dict = {most_recent_id: logs_dict[most_recent_id]}
        else:
            logs_dict = dict(sorted(logs_dict.items()))  # sort by run_id

        default_if_no_logs = {}

        if as_frame:
            try:
                pd = check_pandas_support(f"`{self.__class__.__name__}.get_logs`")

                for run_id in logs_dict:
                    df = pd.DataFrame(logs_dict[run_id])
                    if not df.empty:
                        df = df.set_index(
                            [col for col in df.columns if col in index_names]
                        ).sort_index()
                    logs_dict[run_id] = df

                default_if_no_logs = pd.DataFrame({})

            except ImportError as exc:
                if as_frame != "auto":
                    raise ImportError(
                        "Returning pandas objects requires pandas to be installed. "
                        "Alternatively, set 'as_frame' to False or 'auto'."
                    ) from exc

        if select == "most_recent":
            # We return the only value in logs_dict.
            return next(iter(logs_dict.values()), default_if_no_logs)

        return logs_dict
