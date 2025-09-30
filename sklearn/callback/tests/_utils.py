# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import time

from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.callback import CallbackSupportMixin
from sklearn.utils.parallel import Parallel, delayed


class TestingCallback:
    """A minimal callback used for smoke testing purposes."""

    def _on_fit_begin(self, estimator):
        pass

    def _on_fit_end(self):
        pass

    def _on_fit_task_end(self, estimator, context, **kwargs):
        pass


class TestingAutoPropagatedCallback(TestingCallback):
    """A minimal auto-propagated callback used for smoke testing purposes."""

    max_estimator_depth = None


class NotValidCallback:
    """Invalid callback since it's missing a method from the protocol.'"""

    def _on_fit_begin(self, estimator):
        pass  # pragma: no cover

    def _on_fit_task_end(self, estimator, context, **kwargs):
        pass  # pragma: no cover


class BaseEstimatorPrivateFit(BaseEstimator):
    """A class that adds the implementation of a public and private fit method to the
    BaseEstimator class.
    """

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X=None, y=None, X_val=None, y_val=None):
        if isinstance(self, CallbackSupportMixin):
            callback_ctx = self.init_callback_context()
            callback_ctx.eval_on_fit_begin(estimator=self)
            try:
                return self.__skl_fit__(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    callback_ctx=callback_ctx,
                )
            finally:
                if callback_ctx is not None:
                    callback_ctx.eval_on_fit_end(estimator=self)
        else:
            return self.__skl_fit__(
                X_train=X_train,
                y_train=y_train,
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

    def __skl_fit__(
        self, X_train=None, y_train=None, X_val=None, y_val=None, callback_ctx=None
    ):
        for i in range(self.max_iter):
            if callback_ctx is not None:
                subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if callback_ctx is not None and subcontext.eval_on_fit_task_end(
                estimator=self,
                data={
                    "X_train": X_train,
                    "y_train": y_train,
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

    def __skl_fit__(
        self, X_train=None, y_train=None, X_val=None, y_val=None, callback_ctx=None
    ):
        i = 0
        while True:
            if callback_ctx is not None:
                subcontext = callback_ctx.subcontext(task_id=i)

            time.sleep(self.computation_intensity)  # Computation intensive task

            if callback_ctx is not None and subcontext.eval_on_fit_task_end(
                estimator=self,
                data={
                    "X_train": X_train,
                    "y_train": y_train,
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

    def __skl_fit__(
        self, X_train=None, y_train=None, X_val=None, y_val=None, callback_ctx=None
    ):
        Parallel(n_jobs=self.n_jobs, prefer=self.prefer)(
            delayed(_func)(
                self,
                self.estimator,
                data={
                    "X_train": X_train,
                    "y_train": y_train,
                    "X_val": X_val,
                    "y_val": y_val,
                },
                callback_ctx=callback_ctx.subcontext(
                    task_name="outer", task_id=i, max_subtasks=self.n_inner
                )
                if callback_ctx is not None
                else None,
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

        est.fit(**data)

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
