# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause


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
    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the metric on the model.

        If `scoring` represents a single score, one can use:

        - a single string (see :ref:`scoring_string_names`);
        - a callable (see :ref:`scoring_callable`) that returns a single value;
        - `None`, the `estimator`'s
          :ref:`default evaluation criterion <scoring_api_overview>` is used.

        If `scoring` represents multiple scores, one can use:

        - a list or tuple of unique strings;
        - a callable returning a dictionary where the keys are the metric
          names and the values are the metric scores;
        - a dictionary with metric names as keys and callables as values.

    on : str, default="train_set"
        Which data to compue the metric on. Possible values are "train_set",
        "validation_set" and "both". "train_set" corresponds to using the X and y
        arguments of the fit function, "validation_set" corresponds to using the X_val
        and y_val arguments, "both" corresponds to using both.
    """

    requires_fit_info = ["reconstruction_attributes"]

    def __init__(self, scoring, on="train_set"):
        possible_on_values = ("train_set", "validation_set", "both")
        if on not in possible_on_values:
            raise InvalidParameterError(
                f"The 'on' parameter of {self.__class__.__name__} must be a str among "
                f"{possible_on_values}. Got {on} instead."
            )
        self.on = on
        check_scoring(None, scoring, allow_none=True)  # validate the scoring param
        self.scoring = scoring
        self._shared_log = get_callback_manager().list()

    def on_fit_begin(self, estimator):
        self.scorer = check_scoring(estimator, self.scoring)

    def on_fit_task_end(
        self, estimator, context, data, fitted_estimator=None, **kwargs
    ):
        # TODO: add a task_info dict in the logs
        if fitted_estimator is None:
            return
        context_path = get_context_path(context)
        if self.on == "train_set" or self.on == "both":
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
            metric_name = self.scorer._score_func.__name__
            if self.scorer._sign == -1:
                metric_name = "neg_" + metric_name
            log_item = {metric_name: metric_value}
        log_item["on"] = on
        for depth, ctx in enumerate(context_path):
            if depth == 0:
                timestamp = ctx.init_time.strftime("UTC%Y-%m-%d_%H:%M:%S.%f")
                run_id = f"{ctx.estimator_name}_{timestamp}_{ctx.uuid}"
            prev_task_str = (
                f"{ctx.source_estimator_name}_{ctx.source_task_name}|"
                if ctx.source_estimator_name is not None
                else ""
            )
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
        select : "all" or "most_recent", default="all"
            Which log run to return.

            If set to "all", all runs are returned in a dictionary, and the dictionary
            is empty if there are no logs.

            If set to "most_recent", only the log from the last run is directly
            returned, and if there are no logs, an empty dictionary or DataFrame is
            returned.

        as_frame : "auto" or bool, default="auto"
            Whether to have the logs formatted as multi-index pandas DataFrames. If set
            to False the logs are formatted as dictionaries instead. If set to "auto",
            the avialbility of pandas is evaluated and the format is chosen accordingly.

        Returns
        -------
        logs : dict or pandas DataFrame
            The logged values, either from all the runs or only the most recent one,
            depending on the `select` parameter. Each run's log is formatted as a pandas
            DataFrame or a dictionary depending on the `as_frame` parameter.
        """
        log_item_list = list(self._shared_log)

        index_prefix = "__index__"
        logs_dict = {}
        index_names = set()
        for run_id, log_item in log_item_list:
            if run_id not in logs_dict:
                logs_dict[run_id] = {}
                for key, val in log_item.items():
                    if key.startswith(index_prefix):
                        key = key[len(index_prefix) :]
                        index_names.add(key)
                    logs_dict[run_id][key] = [val]

            else:
                for key, val in log_item.items():
                    if key.startswith(index_prefix):
                        key = key[len(index_prefix) :]
                    logs_dict[run_id][key].append(val)

        if select == "most_recent":
            if not logs_dict:
                # In case there is no log, here an empty dict is added so that when
                # select is "most_recent" we can always assume hat logs_dict contains
                # just one item and return it.
                logs_dict = {"": {}}
            else:
                run_ids = list(logs_dict.keys())
                sorted(run_ids)
                for run_id in run_ids[:-1]:
                    logs_dict.pop(run_id)

        if as_frame:
            try:
                pd = check_pandas_support(f"`{self.__class__.__name__}.get_logs`")

                for run_id in logs_dict:
                    df = pd.DataFrame(logs_dict[run_id])
                    logs_dict[run_id] = df.set_index(
                        [col for col in df.columns if col in index_names]
                    ).sort_index()

            except ImportError as exc:
                if as_frame != "auto":
                    raise ImportError(
                        "Returning pandas objects requires pandas to be installed. "
                        "Alternatively, set 'as_frame' to False or 'auto'."
                    ) from exc

        if select == "most_recent":
            # We return the only value in logs_dict.
            return next(iter(logs_dict.values()))

        return logs_dict
