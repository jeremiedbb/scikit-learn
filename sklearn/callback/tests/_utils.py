# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

from sklearn._config import config_context, get_config
from sklearn.base import BaseEstimator, clone
from sklearn.callback import CallbackSupportMixin
from sklearn.callback._callback_context import CallbackContext
from sklearn.utils._tags import get_tags
from sklearn.utils.parallel import Parallel, delayed


class TestingCallback:
    """A minimal callback used for smoke testing purposes."""

    def on_fit_begin(self, estimator):
        pass

    def on_fit_end(self):
        pass

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass


class TestingAutoPropagatedCallback(TestingCallback):
    """A minimal auto-propagated callback used for smoke testing purposes."""

    max_estimator_depth = None


class NotValidCallback:
    """Invalid callback since it's missing a method from the protocol.'"""

    def on_fit_begin(self, estimator):
        pass  # pragma: no cover

    def on_fit_task_end(self, estimator, context, **kwargs):
        pass  # pragma: no cover


class BaseEstimatorPrivateFit(BaseEstimator):
    """A class that adds the implementation of a public and private fit method to the
    BaseEstimator class.
    """

    def fit(self, X=None, y=None, X_val=None, y_val=None):
        global_skip_validation = get_config()["skip_parameter_validation"]
        if not global_skip_validation:
            self._validate_params()
        with config_context(
            skip_parameter_validation=global_skip_validation
            or get_tags(self)._prefer_skip_nested_validation
        ):
            if isinstance(self, CallbackSupportMixin):
                callback_ctx = CallbackContext._from_estimator(estimator=self)
                try:
                    return self.__skl_fit__(
                        X=X,
                        y=y,
                        X_val=X_val,
                        y_val=y_val,
                        callback_ctx=callback_ctx,
                    )
                finally:
                    callback_ctx.eval_on_fit_end(estimator=self)
            else:
                return self.__sklearn_fit__(
                    X=X,
                    y=y,
                    X_val=X_val,
                    y_val=y_val,
                )


class Estimator(CallbackSupportMixin, BaseEstimatorPrivateFit):
    """A class that mimics the behavior of an estimator.

    The iterative part uses a loop with a max number of iterations known in advance.
    """

    _parameter_constraints: dict = {}

    def __init__(self, max_iter=20, computation_intensity=0.001):
        self.max_iter = max_iter
        self.computation_intensity = computation_intensity

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags._prefer_skip_nested_validation = False
        return tags

    def __skl_fit__(self, X=None, y=None, X_val=None, y_val=None, callback_ctx=None):
        callback_ctx.set_task_info(
            task_name="fit", task_id=0, max_subtasks=self.max_iter
        )
        callback_ctx.eval_on_fit_begin(estimator=self)
        for i in range(self.max_iter):
            subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={
                    "X_train": X,
                    "y_train": y,
                    "X_val": X_val,
                    "y_val": y_val,
                },
            ):
                break

        self.n_iter_ = i + 1

        return self


class WhileEstimator(CallbackSupportMixin, BaseEstimatorPrivateFit):
    """A class that mimics the behavior of an estimator.

    The iterative part uses a loop with a max number of iterations known in advance.
    """

    _parameter_constraints: dict = {}

    def __init__(self, computation_intensity=0.001):
        self.computation_intensity = computation_intensity

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags._prefer_skip_nested_validation = False
        return tags

    def __skl_fit__(self, X=None, y=None, X_val=None, y_val=None, callback_ctx=None):
        callback_ctx.set_task_info(task_name="fit", task_id=0, max_subtasks=None)
        callback_ctx.eval_on_fit_begin(estimator=self)
        i = 0
        while True:
            subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={
                    "X_train": X,
                    "y_train": y,
                    "X_val": X_val,
                    "y_val": y_val,
                },
            ):
                break

            if i == 20:
                break

            i += 1

        return self


class MetaEstimator(CallbackSupportMixin, BaseEstimatorPrivateFit):
    """A class that mimics the behavior of a meta-estimator.

    It has two levels of iterations. The outer level uses parallelism and the inner
    level is done in a function that is not a method of the class.
    """

    _parameter_constraints: dict = {}

    def __init__(
        self, estimator, n_outer=4, n_inner=3, n_jobs=None, prefer="processes"
    ):
        self.estimator = estimator
        self.n_outer = n_outer
        self.n_inner = n_inner
        self.n_jobs = n_jobs
        self.prefer = prefer

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags._prefer_skip_nested_validation = False
        return tags

    def __skl_fit__(self, X=None, y=None, X_val=None, y_val=None, callback_ctx=None):
        callback_ctx.set_task_info(
            task_name="fit", task_id=0, max_subtasks=self.n_outer
        )
        callback_ctx.eval_on_fit_begin(estimator=self)
        Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(_func)(
                self,
                self.estimator,
                data={
                    "X_train": X,
                    "y_train": y,
                    "X_val": X_val,
                    "y_val": y_val,
                },
                callback_ctx=callback_ctx.subcontext(
                    task_name="outer", task_id=i, max_subtasks=self.n_inner
                ),
            )
            for i in range(self.n_outer)
        )

        return self


def _func(meta_estimator, inner_estimator, data, *, callback_ctx):
    for i in range(meta_estimator.n_inner):
        est = clone(inner_estimator)

        inner_ctx = (
            callback_ctx.subcontext(task_name="inner", task_id=i).propagate_callbacks(
                sub_estimator=est
            )
            if callback_ctx is not None
            else None
        )

        est.fit(
            X=data["X_train"],
            y=data["y_train"],
            X_val=data["X_val"],
            y_val=data["y_val"],
        )

        if callback_ctx is not None:
            inner_ctx.eval_on_fit_task_end(
                estimator=meta_estimator,
                data=data,
            )

    if callback_ctx is not None:
        callback_ctx.eval_on_fit_task_end(
            estimator=meta_estimator,
            data=data,
        )
