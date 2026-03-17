# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict

from sklearn.callback._callback_context import get_context_path
from sklearn.callback._callback_support import get_callback_manager
from sklearn.metrics import check_scoring
from sklearn.utils._optional_dependencies import check_pandas_support
from sklearn.utils._param_validation import StrOptions, validate_params


class ScoringMonitor:
    """Callback that monitors a score for each iterative steps of an estimator.

    The specified scorer is called on the training or validation data at each iterative
    step of the estimator, and logged by the callbacks. The logs can be retrieved
    through the `get_logs` method.

    Parameters
    ----------
    on : {"train_set", "validation_set", "both"}, default="train_set"
        Which data to compue the score on. Possible values are "train_set",
        "validation_set" and "both". "train_set" corresponds to using the X and y
        arguments of the fit function, "validation_set" corresponds to using the X_val
        and y_val arguments. "both" corresponds to using both.

    scoring : str, callable, list, tuple, dict or None
        The scoring method to use to monitor the model.

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
    """

    @validate_params(
        {
            "on": [StrOptions({"train_set", "validation_set", "both"})],
            "scoring": [str, callable, list, tuple, dict, None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, *, on="train_set", scoring):
        self.on = on
        self.scoring = scoring
        self._shared_log = get_callback_manager().list()
        self._run_scorers = {}

    def setup(self, context):
        # A scorer per run is needed to avoid race conditions when the callback is set
        # on different estimators and the scorer is the estimator's default scorer.
        self._run_scorers[context.root_uuid] = check_scoring(
            context.estimator, self.scoring
        )

    def teardown(self, context):
        pass

    def on_fit_task_begin(self, context):
        pass

    def on_fit_task_end(
        self,
        context,
        *,
        X=None,
        y=None,
        metadata=None,
        fitted_estimator=None,
    ):
        if fitted_estimator is None:
            return

        context_path = get_context_path(context)
        if self.on in ("train_set", "both"):
            sample_weight = metadata.get("sample_weight", None)
            self._add_log_entry(
                X, y, "train_set", fitted_estimator, sample_weight, context_path
            )
        if self.on in ("validation_set", "both"):
            X, y = metadata.get("X_val", None), metadata.get("y_val", None)
            self._add_log_entry(
                X, y, "validation_set", fitted_estimator, metadata, context_path
            )

    def _add_log_entry(self, X, y, on, fitted_estimator, sample_weight, context_path):
        if X is None or y is None:
            return

        # run_id
        root_ctx = context_path[0]
        timestamp = root_ctx.init_time.strftime("UTC%Y-%m-%d-%H:%M:%S.%f")
        run_id = f"{root_ctx.estimator_name}_{timestamp}_{root_ctx.root_uuid}"

        # log_data
        log_data = {"on": on}
        score_params = {}
        scorer = self._run_scorers[root_ctx.root_uuid]
        if sample_weight is not None and scorer._accept_sample_weight():
            score_params["sample_weight"] = sample_weight
        score_value = scorer(fitted_estimator, X, y, **score_params)
        if isinstance(score_value, dict):
            log_data.update(score_value)
        else:
            score_name = self.scoring if isinstance(self.scoring, str) else "score"
            log_data[score_name] = score_value

        # log_index
        log_index = {}
        for depth, ctx in enumerate(context_path):
            source_task_str = (
                f"{ctx.source_estimator_name}_{ctx.source_task_name}|"
                if ctx.source_estimator_name is not None
                else ""
            )
            # TODO(callbacks): add a task_info attribute in CallbackContext containing
            # detailed info of task, and enhance the logs with it.
            log_index[
                f"{depth}_{source_task_str}{ctx.estimator_name}_{ctx.task_name}"
            ] = ctx.task_id

        self._shared_log.append((run_id, log_index, log_data))

    @validate_params(
        {
            "select": [StrOptions({"all", "most_recent"})],
            "as_frame": ["boolean"],
        },
        prefer_skip_nested_validation=True,
    )
    def get_logs(self, select="all", as_frame=True):
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

        as_frame : bool, default=True
            Whether to have the individual run logs formatted as multi-index Pandas
            DataFrames. If set to False the individual run logs are formatted as lists
            of dictionaries instead.

        Returns
        -------
        logs : dict or pandas DataFrame
            The logged values, formatted as :

            - a dict of pandas dataframes if `select` is "all" and `as_frame` is True.

            - a pandas dataframe is `select` is "most_recent" and `as_frame` is True.

            - a dict of lists of dicts if `select` is "all" and `as_frame` is False.

            - a list of dicts is `select` is "most_recent" and `as_frame` is False.
        """
        log_item_list = list(self._shared_log)

        logs_dict = defaultdict(list)
        index_names = set()
        for run_id, log_index, log_data in log_item_list:
            index_names = index_names.union(list(log_index.keys()))
            log_data.update(log_index)
            logs_dict[run_id].append(log_data)

        # Sort runs chronologically using the timestamp in the run_id keys
        logs_dict = dict(sorted(logs_dict.items(), key=lambda x: x[0].split("_")[-2]))

        default_if_no_logs = []

        if as_frame:
            pd = check_pandas_support(f"`{self.__class__.__name__}.get_logs`")

            for run_id in logs_dict:
                df = pd.DataFrame(logs_dict[run_id])
                if not df.empty:
                    df = df.set_index(
                        [col for col in df.columns if col in index_names]
                    ).sort_index()
                logs_dict[run_id] = df

            default_if_no_logs = pd.DataFrame({})

        if select == "most_recent":
            return list(logs_dict.values())[-1] if logs_dict else default_if_no_logs

        return logs_dict
