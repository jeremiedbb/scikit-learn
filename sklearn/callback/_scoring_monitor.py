# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from collections import defaultdict

from sklearn.callback._callback_context import get_context_path
from sklearn.callback._callback_support import get_callback_manager
from sklearn.metrics import check_scoring
from sklearn.utils._optional_dependencies import check_pandas_support
from sklearn.utils._param_validation import StrOptions, validate_params


class ScoringMonitor:
    """Callback that monitors a score for each iterative step of an estimator.

    The specified scorer is called on the training or validation data at each iterative
    step of the estimator, and logged by the callbacks. The logs can be retrieved
    through the `get_logs` method.

    Parameters
    ----------
    eval_on : {"train", "val", "both"}, default="train"
        Which data to compute the score on:

        - `"train"`: only the scores on the training data (the `X` and `y` arguments of
          the fit function) are logged;
        - `"val"`: only the scores on the validation data (the `X_val` and `y_val`
          arguments of the fit function) are logged;
        - `"both"`: the scores of both the training and validation data are logged.

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
            "eval_on": [StrOptions({"train", "val", "both"})],
            "scoring": [str, callable, list, tuple, dict, None],
        },
        prefer_skip_nested_validation=True,
    )
    def __init__(self, *, eval_on="train", scoring):
        self.eval_on = eval_on
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
        if self.eval_on in ("train", "both"):
            sample_weight = metadata.get("sample_weight", None)
            self._add_log_entry(
                X, y, "train", fitted_estimator, sample_weight, context_path
            )
        if self.eval_on in ("val", "both"):
            X, y = metadata.get("X_val", None), metadata.get("y_val", None)
            sample_weight = metadata.get("sample_weight_val", None)
            self._add_log_entry(
                X, y, "val", fitted_estimator, sample_weight, context_path
            )

    def _add_log_entry(
        self, X, y, eval_on, fitted_estimator, sample_weight, context_path
    ):
        if X is None or y is None:
            return

        # run_info
        root_ctx = context_path[0]
        run_id = root_ctx.root_uuid
        run_info = {
            "timestamp": root_ctx.init_time.strftime("UTC%Y-%m-%d-%H:%M:%S.%f"),
            "root_estimator_name": root_ctx.estimator_name,
        }

        # task_id and parent_tasks_info
        task_id = tuple(ctx.task_id for ctx in context_path[:-1])
        parent_tasks_info = tuple(
            {
                "task_name": ctx.task_name,
                "estimator_name": ctx.estimator_name,
                "source_task_name": ctx.source_task_name,
                "source_estimator_name": ctx.source_estimator_name,
            }
            for ctx in context_path[:-1]
        )

        # log_data
        current_ctx = context_path[-1]
        log_data = {
            "task_name": current_ctx.task_name,
            "task_id": current_ctx.task_id,
            "estimator_name": current_ctx.estimator_name,
            "eval_on": eval_on,
        }
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

        self._shared_log.append(
            (run_id, run_info, task_id, log_data, parent_tasks_info)
        )

    @validate_params(
        {
            "select": [StrOptions({"all", "most_recent"})],
            "as_frame": ["boolean"],
        },
        prefer_skip_nested_validation=True,
    )
    def get_logs(self, select="most_recent", as_frame=False):
        """Get the logged values.

        If `select == "all"`, a dictionary is returned with run ids as keys and logs as
        values. A run corresponds to a fit execution of the outermost meta-estimator
        that is a parent of the estimator the callback is registered on. If the
        estimator is not wrapped in a meta-estimator, a run corresponds to a single
        fit execution of the estimator.

        For each run key in the dictionary, the value is a dictionary containing:
            - "info": a dictionary containing the timestamp for the start of fit and the
              estimator name for the outermost parent meta-estimator.

            - "task_tree": nested dictionaries describing the tree structure of the
              tasks.

            - "logs": a dictionary with tuples of task id as key and for values
              dictionaries containing:
                  - "values": pandas Dataframe or list of dict containing the score
                    values, the type being controlled by the `as_frame` argument.

                  - "task_path": a tuple of strings with the estimator names and task
                    names corresponding to the task ids in the key of this dict.

        Parameters
        ----------
        select : {"all", "most_recent"}, default="most_recent"
            Which log run to return:

            - `"all"`: returns the whole log as a dictionary indexed by run ids;
            - `"most_recent"`: only returns the log of the most recent run based on
              the timestamp in the run id.

        as_frame : bool, default=False
            Whether to have the individual task logs formatted as Pandas DataFrames. If
            set to False the individual run logs are formatted as lists of dictionaries
            instead.

        Returns
        -------
        logs : dict
            The logged values.
        """
        log_item_list = list(self._shared_log)

        logs_dict = defaultdict(
            lambda: {
                "logs": defaultdict(lambda: {"values": []}),
                "task_tree": {},
                "info": {},
            }
        )

        for run_id, run_info, task_id, log_data, parent_tasks_info in log_item_list:
            logs_dict[run_id]["logs"][task_id]["values"].append(log_data)
            logs_dict[run_id]["logs"][task_id]["task_path"] = tuple(
                f"{info['source_estimator_name']} {info['source_task_name']} | "
                f"{info['estimator_name']} {info['task_name']}"
                if info["source_task_name"] is not None
                else f"{info['estimator_name']} {info['task_name']}"
                for info in parent_tasks_info
            )
            logs_dict[run_id]["info"].update(run_info)
            task_dict = logs_dict[run_id]["task_tree"]
            for i, id in enumerate(task_id):
                if id not in task_dict:
                    task_dict[id] = {
                        "subtasks": {},
                        "task_info": {**parent_tasks_info[i]},
                    }
                task_dict = task_dict[id]["subtasks"]

        # Sort runs chronologically.
        logs_dict = dict(
            sorted(
                logs_dict.items(),
                key=lambda x: x[1]["info"]["timestamp"],
            )
        )
        # Convert the defaultdicts to dicts.
        for run_id in logs_dict:
            logs_dict[run_id]["logs"] = dict(logs_dict[run_id]["logs"])

        if as_frame:
            pd = check_pandas_support(f"`{self.__class__.__name__}.get_logs`")

            for run_id in logs_dict:
                for task_id in logs_dict[run_id]["logs"]:
                    logs_dict[run_id]["logs"][task_id]["values"] = pd.DataFrame(
                        logs_dict[run_id]["logs"][task_id]["values"]
                    )

        if select == "most_recent":
            return list(logs_dict.values())[-1] if logs_dict else {}

        return logs_dict
