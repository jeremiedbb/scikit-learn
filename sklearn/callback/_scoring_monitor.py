# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import colorsys
from collections import defaultdict

from sklearn.callback._callback_context import get_context_path
from sklearn.callback._callback_support import get_callback_manager
from sklearn.metrics import check_scoring
from sklearn.utils._optional_dependencies import (
    check_matplotlib_support,
    check_pandas_support,
)
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
        # A scorer per estimator is needed to avoid race conditions when the callback is
        # set on different estimators and the scorer is the estimator's default
        # scorer.
        if context.estimator_name not in self._run_scorers:
            self._run_scorers[context.estimator_name] = check_scoring(
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

        if self.eval_on in ("train", "both"):
            sample_weight = metadata.get("sample_weight", None)
            self._add_log_entry(X, y, "train", fitted_estimator, sample_weight, context)
        if self.eval_on in ("val", "both"):
            X, y = metadata.get("X_val", None), metadata.get("y_val", None)
            sample_weight = metadata.get("sample_weight_val", None)
            self._add_log_entry(X, y, "val", fitted_estimator, sample_weight, context)

    def _add_log_entry(self, X, y, eval_on, fitted_estimator, sample_weight, context):
        if X is None or y is None:
            return

        context_path = get_context_path(context)

        # run_info
        root_ctx = context_path[0]
        run_id = root_ctx.root_uuid.hex
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
        log_data = {
            "task_name": context.task_name,
            "task_id": context.task_id,
            "estimator_name": context.estimator_name,
            "eval_on": eval_on,
            "subtasks_ordered": context.subtasks_ordered,
        }
        score_params = {}
        scorer = self._run_scorers[context.estimator_name]
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

    def plot(self, run_id=None):
        """Create a matplotlib figure for a given run.

        The figure contains one subplot per depth level in the logs (excluding
        depth 0) and per score:

        - depth levels whose parent subtasks are not ordered are displayed as grouped
          box plots;
        - the leaf depth level is displayed as line plots, with one curve for each
          group of values that shares the same parent key.

        Parameters
        ----------
        run_id : str or None, default=None
            Identifier of the run to plot. It must be one of the keys returned by
            `get_logs(select="all")`. If None, the most recent run is plotted.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created matplotlib figure.
        """
        check_matplotlib_support(f"{self.__class__.__name__}.plot")
        import matplotlib.pyplot as plt
        from matplotlib import colors as mcolors

        logs_all = self.get_logs(select="all", as_frame=False)
        if run_id is None:
            if not logs_all:
                raise ValueError("No run available to plot.")
            run_id = next(reversed(logs_all))

        if run_id not in logs_all:
            raise ValueError(f"Unknown run_id: {run_id!r}.")

        run_logs = logs_all[run_id]["logs"]
        run_task_tree = logs_all[run_id]["task_tree"]
        keys_by_depth = defaultdict(list)
        for key in run_logs:
            depth = len(key)
            if depth > 0:
                keys_by_depth[depth].append(key)

        if not keys_by_depth:
            raise ValueError("No logs available with depth > 0 for this run.")

        depths = sorted(keys_by_depth)
        metric_metadata = {
            "task_name",
            "task_id",
            "subtasks_ordered",
            "estimator_name",
            "eval_on",
        }

        score_names = []
        for task_log in run_logs.values():
            values = task_log["values"]
            if values:
                score_names = [k for k in values[0] if k not in metric_metadata]
                if score_names:
                    break

        if not score_names:
            raise ValueError("No score values available in logs for this run.")

        fig, axes = plt.subplots(
            nrows=len(depths),
            ncols=len(score_names),
            squeeze=False,
            constrained_layout=True,
        )

        for depth_idx, depth in enumerate(depths):
            depth_keys = sorted(keys_by_depth[depth])
            depth_task_names = []
            parent_titles = []
            ordered_flags = []
            for key in depth_keys:
                task_path = run_logs[key].get("task_path", tuple())
                if len(task_path) >= depth:
                    depth_task_names.append(task_path[depth - 1])
                # Plot type for a depth depends on whether parent subtasks are ordered.
                parent_values = run_logs.get(key[:-1], {}).get("values", [])
                ordered_flags.extend(
                    value.get("subtasks_ordered", False) for value in parent_values
                )
                parent_titles.extend(
                    f"{value['estimator_name']} {value['task_name']}"
                    for value in parent_values
                    if value.get("estimator_name") and value.get("task_name")
                )
            depth_task_names = sorted(set(depth_task_names))
            parent_titles = sorted(set(parent_titles))
            depth_task_ordered = bool(ordered_flags) and all(ordered_flags)
            if len(depth_task_names) == 1:
                subplot_title = depth_task_names[0]
            elif len(depth_task_names) > 1:
                subplot_title = " | ".join(depth_task_names)
            else:
                subplot_title = f"Depth {depth}"
            if len(parent_titles) == 1:
                line_subplot_title = parent_titles[0]
            elif len(parent_titles) > 1:
                line_subplot_title = " | ".join(parent_titles)
            else:
                line_subplot_title = subplot_title
            for score_idx, score_name in enumerate(score_names):
                ax = axes[depth_idx, score_idx]

                if depth_task_ordered:
                    leaf_xticks = set()
                    eval_color = {"train": "#00bcd4", "val": "tab:orange"}
                    unknown_eval_colors = {}
                    seen_eval_labels = set()
                    parent_groups = sorted({key[:-1] for key in depth_keys})
                    group_index = {group: i for i, group in enumerate(parent_groups)}
                    n_groups = max(len(parent_groups), 1)

                    def get_group_color(base_color, group):
                        rgb = mcolors.to_rgb(base_color)
                        if n_groups == 1:
                            return rgb
                        rank = group_index[group] / (n_groups - 1)
                        lightness_shift = (rank - 0.5) * 0.24
                        hue, lightness, saturation = colorsys.rgb_to_hls(*rgb)
                        lightness = min(0.92, max(0.18, lightness + lightness_shift))
                        return colorsys.hls_to_rgb(hue, lightness, saturation)

                    for key in depth_keys:
                        values = run_logs[key]["values"]
                        if not values:
                            continue

                        grouped_by_eval = defaultdict(list)
                        for value in values:
                            grouped_by_eval[value["eval_on"]].append(value)

                        for eval_on, eval_values in grouped_by_eval.items():
                            if score_name not in eval_values[0]:
                                continue

                            eval_values = sorted(
                                eval_values, key=lambda value: value["task_id"]
                            )
                            x_values = [value["task_id"] + 1 for value in eval_values]
                            y_values = [value[score_name] for value in eval_values]
                            leaf_xticks.update(x_values)
                            if (
                                eval_on not in eval_color
                                and eval_on not in unknown_eval_colors
                            ):
                                unknown_eval_colors[eval_on] = (
                                    f"C{2 + len(unknown_eval_colors)}"
                                )
                            color = (
                                get_group_color(eval_color[eval_on], key[:-1])
                                if eval_on in eval_color
                                else unknown_eval_colors[eval_on]
                            )
                            label = (
                                eval_on
                                if eval_on not in seen_eval_labels
                                else "_nolegend_"
                            )
                            seen_eval_labels.add(eval_on)
                            ax.plot(
                                x_values,
                                y_values,
                                marker="o",
                                color=color,
                                alpha=0.8,
                                label=label,
                            )

                    ax.set_title(line_subplot_title)
                    if leaf_xticks:
                        ax.set_xticks(sorted(leaf_xticks))
                    ax.set_xlabel("iteration")
                    ax.set_ylabel(score_name)
                    if ax.lines:
                        ax.legend(loc="best")
                else:
                    eval_ons = set()
                    task_values = defaultdict(dict)
                    group_labels = {}
                    for key in depth_keys:
                        values = run_logs[key]["values"]
                        if not values:
                            continue

                        by_eval = defaultdict(list)
                        for value in values:
                            by_eval[value["eval_on"]].append(value)
                            eval_ons.add(value["eval_on"])
                        group_key = key[:-1]
                        task_path = run_logs[key].get("task_path", tuple())
                        parent_path = task_path[:-1]
                        group_labels[group_key] = (
                            " > ".join(parent_path) if parent_path else "<root>"
                        )
                        for eval_on, eval_values in by_eval.items():
                            if score_name not in eval_values[0]:
                                continue
                            eval_values = sorted(
                                eval_values, key=lambda value: value["task_id"]
                            )
                            # One box per task key and eval_on.
                            task_values[key][eval_on] = [
                                value[score_name] for value in eval_values
                            ]

                    eval_order = [name for name in ("train", "val") if name in eval_ons]
                    eval_order.extend(
                        sorted(
                            name for name in eval_ons if name not in {"train", "val"}
                        )
                    )

                    ordered_keys = sorted(task_values, key=lambda key: (key[:-1], key))
                    base_x = list(range(len(ordered_keys)))
                    n_eval = max(len(eval_order), 1)
                    box_width = 0.8 / n_eval
                    eval_color = {"train": "#00bcd4", "val": "tab:orange"}
                    unknown_eval_colors = {}
                    legend_labels = []
                    legend_handles = []

                    def common_prefix(keys):
                        if not keys:
                            return tuple()
                        prefix = list(keys[0])
                        for key in keys[1:]:
                            i = 0
                            while (
                                i < min(len(prefix), len(key)) and prefix[i] == key[i]
                            ):
                                i += 1
                            prefix = prefix[:i]
                            if not prefix:
                                break
                        return tuple(prefix)

                    def title_from_ancestor(prefix):
                        if not prefix:
                            root_values = run_logs.get(tuple(), {}).get("values", [])
                            if root_values:
                                root_val = root_values[0]
                                return f"{root_val['estimator_name']} {root_val['task_name']}"
                            return line_subplot_title

                        node = run_task_tree
                        task_info = None
                        for task_id in prefix:
                            if task_id not in node:
                                task_info = None
                                break
                            task_info = node[task_id]["task_info"]
                            node = node[task_id]["subtasks"]
                        if task_info is None:
                            return line_subplot_title
                        return f"{task_info['estimator_name']} {task_info['task_name']}"

                    box_subplot_title = title_from_ancestor(common_prefix(ordered_keys))

                    for eval_idx, eval_on in enumerate(eval_order):
                        positions = [
                            x + (eval_idx - (n_eval - 1) / 2) * box_width
                            for x in base_x
                        ]
                        data = [
                            task_values.get(key, {}).get(eval_on, [])
                            for key in ordered_keys
                        ]
                        non_empty = [
                            (pos, values)
                            for pos, values in zip(positions, data)
                            if len(values) > 0
                        ]
                        if not non_empty:
                            continue

                        if (
                            eval_on not in eval_color
                            and eval_on not in unknown_eval_colors
                        ):
                            unknown_eval_colors[eval_on] = (
                                f"C{2 + len(unknown_eval_colors)}"
                            )
                        color = (
                            eval_color[eval_on]
                            if eval_on in eval_color
                            else unknown_eval_colors[eval_on]
                        )

                        box = ax.boxplot(
                            [values for _, values in non_empty],
                            positions=[pos for pos, _ in non_empty],
                            widths=box_width,
                            patch_artist=True,
                            manage_ticks=False,
                        )
                        for patch in box["boxes"]:
                            patch.set_facecolor(color)
                            patch.set_alpha(0.55)
                        for line in box["whiskers"] + box["caps"] + box["medians"]:
                            line.set_color(color)

                        if eval_on not in legend_labels:
                            legend_labels.append(eval_on)
                            legend_handles.append(box["boxes"][0])

                    ax.set_xticks(base_x)
                    ax.set_xticklabels(
                        [
                            run_logs[key].get("task_path", tuple())[-1]
                            if run_logs[key].get("task_path", tuple())
                            else "<root>"
                            for key in ordered_keys
                        ],
                        rotation=45,
                        ha="right",
                    )
                    ax.set_title(box_subplot_title)
                    ax.set_ylabel(score_name)
                    if len(legend_handles) > 1:
                        ax.legend(legend_handles, legend_labels, loc="best")

        return fig
