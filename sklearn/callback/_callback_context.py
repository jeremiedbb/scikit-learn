# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.callback import AutoPropagatedCallback

# TODO(callbacks): move these explanations into a dedicated user guide.
#
# The computations tasks performed by an estimator during fit have an inherent tree
# structure, where each task can be decomposed into subtasks and so on. The root of the
# tree represents the whole fit task.
#
# Each loop in the estimator represents a parent task and each iteration of that loop
# represents a child task. To allow callbacks to be generic and reusable across
# estimators, the smallest tasks, i.e. the leaves of the task tree, correspond to
# operations on the full input data (or batch for incremental estimators).
#
# For instance, KMeans has two nested loops: the outer loop is controlled by `n_init`
# and the inner loop is controlled by `max_iter`. Its task tree looks like this:
#
# KMeans fit
# ├── init 0
# │   ├── iter 0
# │   ├── iter 1
# │   ├── ...
# │   └── iter n
# ├── init 1
# │   ├── iter 0
# │   ├── ...
# │   └── iter n
# └── init 2
#     ├── iter 0
#     ├── ...
#     └── iter n
#
# where each innermost `iter j` task corresponds the computation of the labels and
# centers for the full dataset.
#
# When the estimator is a meta-estimator, a task leaf usually corresponds to fitting
# a sub-estimator. Therefore this leaf and the root task of the sub-estimator actually
# represent the same task. In this case the leaf task of the meta-estimator and the root
# task of the sub-estimator are merged into a single task.
#
# For instance a `Pipeline` would have a task tree that looks like this:
#
# Pipeline fit
# ├── step 0 | preprocessor fit
# │   └── <insert preprocessor task tree here>
# └── step 1 | estimator fit
#     └── <insert estimator task tree here>
#
# Concretely, the tree structure is created dynamically and held by an object named
# `CallbackContext`. There's a context for each task and the context is responsible for
# calling the callback hooks for its task and creating contexts for the child tasks.


class CallbackContext:
    """Task level context for the callbacks.

    This class is responsible for managing the callbacks and holding the tree structure
    of an estimator's tasks. Each instance corresponds to a task of the estimator.

    Instances of this class should be created using the `init_callback_context` method
    of its estimator or the `subcontext` method of this class.

    To keep track of the position of a task in the task tree within the callbacks,
    a dictionary containing all relevant information about the task is provided to the
    callback hooks. It's accessible through the `task_info` property of this class.
    """

    @classmethod
    def _from_estimator(cls, estimator, *, task_name, task_id, max_subtasks=None):
        """Private constructor to create a root context.

        Parameters
        ----------
        estimator : estimator instance
            The estimator this context is responsible for.

        task_name : str
            The name of the task this context is responsible for.

        task_id : int
            The id of the task this context is responsible for.

        max_subtasks : int or None, default=None
            The maximum number of subtasks of this task. 0 means it's a leaf.
            None means the maximum number of subtasks is not known in advance.
        """
        new_ctx = cls.__new__(cls)

        # We don't store the estimator in the context to avoid circular references
        # because the estimator already holds a reference to the context.
        new_ctx._callbacks = getattr(estimator, "_skl_callbacks", [])
        new_ctx._estimator_name = estimator.__class__.__name__
        new_ctx._task_name = task_name
        new_ctx._task_id = task_id
        new_ctx._parent = None
        new_ctx._children_map = {}
        new_ctx._max_subtasks = max_subtasks
        new_ctx._prev_estimator_name = None
        new_ctx._prev_task_name = None

        if hasattr(estimator, "_parent_callback_ctx"):
            # This context's task is the root task of the estimator which itself
            # corresponds to a leaf task of a meta-estimator. Both tasks actually
            # represent the same task so we merge both tasks into a single task,
            # attaching the task tree of the sub-estimator to the task tree of
            # the meta-estimator on the way.
            parent_ctx = estimator._parent_callback_ctx
            new_ctx._merge_with(parent_ctx)
            new_ctx._estimator_depth = parent_ctx._estimator_depth + 1
        else:
            new_ctx._estimator_depth = 0

        return new_ctx

    @classmethod
    def _from_parent(cls, parent_context, *, task_name, task_id, max_subtasks=None):
        """Private constructor to create a sub-context.

        Parameters
        ----------
        parent_context : CallbackContext instance
            The parent context of the new context.

        task_name : str
            The name of the task this context is responsible for.

        task_id : int
            The id of the task this context is responsible for.

        max_subtasks : int or None, default=None
            The maximum number of tasks that can be children of the task this context is
            responsible for. 0 means it's a leaf. None means the maximum number of
            subtasks is not known in advance.
        """
        new_ctx = cls.__new__(cls)

        new_ctx._callbacks = parent_context._callbacks
        new_ctx._estimator_name = parent_context._estimator_name
        new_ctx._estimator_depth = parent_context._estimator_depth
        new_ctx._task_name = task_name
        new_ctx._task_id = task_id
        new_ctx._parent = None
        new_ctx._children_map = {}
        new_ctx._max_subtasks = max_subtasks
        new_ctx._prev_estimator_name = None
        new_ctx._prev_task_name = None

        # This task is a subtask of another task of a same estimator
        parent_context._add_child(new_ctx)

        return new_ctx

    @property
    def task_info(self):
        """Information about the corresponding computation task.

        The key value pairs are:

        - task_name : str
            The name of the task.

        - task_id : int
            The identifier of the task.

        - max_subtasks : int or None
            The maximum number of children tasks for this task. 0 means it's a leaf.
            None means the maximum number of subtasks is not known in advance.

        - estimator_name : str
            The name of the estimator.

        - depth : int
            The depth of the task in the task tree.

        - prev_estimator_name : str or None
            The estimator name of the parent task this task was merged with. None if it
            was not merged with another context.

        - prev_task_name : str or None
            The task name of the parent task this task was merged with. None if it
            was not merged with another context.

        - parent_task_info : dict or None
            The task_info property of this task's parent. None if this task is the root.

        Note that it is a dynamic attribute since the root task of an estimator can
        become an intermediate task of a meta-estimator.
        """
        return {
            "task_name": self._task_name,
            "task_id": self._task_id,
            "max_subtasks": self._max_subtasks,
            "estimator_name": self._estimator_name,
            "depth": self._depth,
            "prev_estimator_name": self._prev_estimator_name,
            "prev_task_name": self._prev_task_name,
            "parent_task_info": None
            if self._parent is None
            else self._parent.task_info,
        }

    @property
    def _depth(self):
        """The depth of this task in the task tree."""
        return 0 if self._parent is None else self._parent._depth + 1

    def __iter__(self):
        """Pre-order depth-first traversal of the task tree."""
        yield self
        for context in self._children_map.values():
            yield from context

    def _add_child(self, child_context):
        """Add `child_context` as a child of this context."""
        if child_context._task_id in self._children_map:
            raise ValueError(
                f"Callback context {self._task_name} of estimator "
                f"{self._estimator_name} already has a child with "
                f"task_id={child_context._task_id}."
            )

        if len(self._children_map) == self._max_subtasks:
            raise ValueError(
                f"Cannot add child to callback context {self._task_name} of estimator "
                f"{self._estimator_name} because it already has its maximum "
                f"number of children ({self._max_subtasks})."
            )

        self._children_map[child_context._task_id] = child_context
        child_context._parent = self

    def _merge_with(self, other_context):
        """Merge this context with `other_context`."""
        # Set the parent of the sub-estimator's root context to the parent of the
        # meta-estimator's leaf context
        self._parent = other_context._parent
        self._task_id = other_context._task_id
        self._max_subtasks = other_context._max_subtasks
        other_context._parent._children_map[self._task_id] = self

        # Keep information about the context it was merged with
        self._prev_task_name = other_context._task_name
        self._prev_estimator_name = other_context._estimator_name

    def subcontext(self, task_name="", task_id=0, max_subtasks=None):
        """Create a context for a subtask of the current task.

        Parameters
        ----------
        task_name : str, default=""
            The name of the subtask.

        task_id : int, default=0
            An identifier of the subtask. Usually a number between 0 and the maximum
            number of sibling contexts, but can be any identifier as long as it's unique
            among the siblings.

        max_subtasks : int or None, default=None
            The maximum number of tasks that can be children of the subtask. 0 means
            it's a leaf. None means the maximum number of subtasks is not known in
            advance.
        """
        return CallbackContext._from_parent(
            parent_context=self,
            task_name=task_name,
            task_id=task_id,
            max_subtasks=max_subtasks,
        )

    def eval_on_fit_begin(self, estimator):
        """Evaluate the `_on_fit_begin` method of the callbacks.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.
        """
        for callback in self._callbacks:
            # Only call the on_fit_begin method of callbacks that are not
            # propagated from a meta-estimator.
            if not (
                isinstance(callback, AutoPropagatedCallback)
                and self._parent is not None
            ):
                callback._on_fit_begin(estimator)

        return self

    def eval_on_fit_task_end(self, estimator, **kwargs):
        """Evaluate the `_on_fit_task_end` method of the callbacks.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.

        **kwargs : dict
            arguments passed to the callback. Possible keys are

            - data: dict
                Dictionary containing the training and validation data. The possible
                keys are "X_train", "y_train", "sample_weight_train", "X_val", "y_val",
                and "sample_weight_val". The values are the corresponding data.

            - stopping_criterion: float
                Usually iterations stop when `stopping_criterion <= tol`.
                This is only provided at the innermost level of iterations, i.e. for
                leaf tasks.

            - tol: float
                Tolerance for the stopping criterion.
                This is only provided at the innermost level of iterations, i.e. for
                leaf tasks.

            - from_reconstruction_attributes: estimator instance
                A ready to predict, transform, etc ... estimator as if the fit stopped
                at the end of this task. Usually it's a copy of the caller estimator
                with the necessary attributes set.

            - fit_state: dict
                Model specific quantities updated during fit. This is not meant to be
                used by generic callbacks but by a callback designed for a specific
                estimator instead.

        Returns
        -------
        stop : bool
            Whether or not to stop the current level of iterations at this end of this
            task.
        """
        return any(
            callback._on_fit_task_end(estimator, self.task_info, **kwargs)
            for callback in self._callbacks
        )

    def eval_on_fit_end(self, estimator):
        """Evaluate the `_on_fit_end` method of the callbacks.

        Parameters
        ----------
        estimator : estimator instance
            The estimator calling this callback hook.
        """
        for callback in self._callbacks:
            # Only call the on_fit_end method of callbacks that are not
            # propagated from a meta-estimator.
            if not (
                isinstance(callback, AutoPropagatedCallback)
                and self._parent is not None
            ):
                callback._on_fit_end(estimator, task_info=self.task_info)

    def propagate_callbacks(self, sub_estimator):
        """Propagate the callbacks to a sub-estimator.

        Parameters
        ----------
        sub_estimator : estimator instance
            The estimator to which the callbacks should be propagated.
        """
        bad_callbacks = [
            callback.__class__.__name__
            for callback in getattr(sub_estimator, "_skl_callbacks", [])
            if isinstance(callback, AutoPropagatedCallback)
        ]

        if bad_callbacks:
            raise TypeError(
                f"The sub-estimator ({sub_estimator.__class__.__name__}) of a"
                f" meta-estimator ({self._estimator_name}) can't have"
                f" auto-propagated callbacks ({bad_callbacks})."
                " Register them directly on the meta-estimator."
            )

        callbacks_to_propagate = [
            callback
            for callback in self._callbacks
            if isinstance(callback, AutoPropagatedCallback)
            and (
                callback.max_estimator_depth is None
                or self._estimator_depth < callback.max_estimator_depth
            )
        ]

        # We store the parent context in the sub-estimator to be able to merge the
        # task trees of the sub-estimator and the meta-estimator.
        sub_estimator._parent_callback_ctx = self

        if callbacks_to_propagate:
            sub_estimator.set_callbacks(
                getattr(sub_estimator, "_skl_callbacks", []) + callbacks_to_propagate
            )

        return self


def _get_task_info_path(task_info):
    """Helper function to get the path of task info from this task to the root task.

    Parameters
    ----------
    task_info : dict
        The dictionary representations of a CallbackContext's task node.

    Returns
    -------
    list of dict
        The list of dictionary representations of the ancestors (itself included) of the
        given task.
    """
    return (
        [task_info]
        if task_info["parent_task_info"] is None
        else _get_task_info_path(task_info["parent_task_info"]) + [task_info]
    )
