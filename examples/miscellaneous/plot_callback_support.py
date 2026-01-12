"""
================
Callback support
================

.. currentmodule:: sklearn

This document shows how to make custom :term:`estimators` and :term:`meta-estimators`
compatible with the callback mechanisms supported by scikit-learn.

Generally speaking, a callback is a function that is provided by the
user to be invoked automatically at specific steps of a process, or to be
triggered by specific events. Callbacks provide a clean mechanism for inserting
custom logic (like monitoring progress or metrics, or implementing early stopping)
without modifying the core algorithm of the process.

In scikit-learn, callbacks take the form of classes following a protocol.
This protocol requires the callback classes to implement specific methods which
will be called at specific steps of the fitting of an estimator or a meta-estimator.
These specific methods are :meth:`~callback._base.Callback.on_fit_begin`,
:meth:`~callback._base.Callback.on_fit_task_end` and
:meth:`~callback._base.Callback.on_fit_end`, which are respectively called at
the start of the :term:`fit` method, at the end of each iteration in ``fit``
and at the end of the ``fit`` method.

In order to support the callbacks, estimators need to manipulate
:class:`~callback.CallbackContext` objects. As the name implies, these objects hold
the contextual information necessary to run the calbbacks methods. They are also
responsible for triggering the methods of the callback at the right time.

First a few imports and some random data for the rest of the script.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

import numpy as np

from sklearn.base import BaseEstimator, _fit_context, clone
from sklearn.callback import CallbackSupportMixin, ProgressBar
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import KFold
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)


# %%
# Custom Estimator
# ----------------
# Here we demonstrate how to implement a custom estimator that supports
# callbacks. For the example, a simplified version of KMeans is presented.


# The estimator must inherit from the `CallbackSupportMixin` class.
class SimpleKMeans(CallbackSupportMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, n_clusters=6, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_labels(self, X):
        return np.argmin(euclidean_distances(X, self.centroids_), axis=1)

    # The CallbackContext object is initialized in the `_fit_context` decorator
    # and set as the `_callback_fit_ctx` attribute of the estimator.
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        callback_ctx = self._callback_fit_ctx
        # The maximum number of iterative tasks can be set as an attribute of the
        # callback context.
        callback_ctx.max_subtasks = self.max_iter
        # The `eval_on_fit_begin` method will call the `on_fit_begin` methods of the
        # estimator's callbacks.
        callback_ctx.eval_on_fit_begin(estimator=self)

        random_state = check_random_state(self.random_state)
        self.centroids_ = random_state.rand(self.n_clusters, X.shape[1])

        for i in range(self.max_iter):
            # A subcontext specific to each iteration is spawned.
            subcontext = callback_ctx.subcontext(task_id=i, task_name="fit iteration")

            labels = self._get_labels(X)
            new_centroids = []

            for k in range(self.n_clusters):
                if (labels == k).any():
                    new_centroids.append(X[labels == k].mean(axis=0))
                else:
                    new_centroids.append(self.centroids_[k])

            has_converged = np.array_equal(self.centroids_, new_centroids)
            self.centroids_ = np.stack(new_centroids)

            # The subcontext's `eval_on_fit_task_end` method will call the
            # `on_fit_task_end` methods of the callbacks. Data relative to the
            # current task can be provided through the `data` argument.
            if (
                subcontext.eval_on_fit_task_end(
                    estimator=self,
                    data={"X_train": X, "y_train": None},
                )
                or has_converged
            ):
                # The `eval_on_fit_task_end` method returns a boolean, which will
                # be set to True if any of the callbacks' `on_fit_task_end` method
                # returns True. This allows to implement early stopping with
                # arbitrary criterions.
                break

        # The `on_fit_end` methods of the callbacks are called in the decorator
        # after `fit` is completed.
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self._get_labels(X)

    def transform(self, X):
        check_is_fitted(self)
        return euclidean_distances(X, self.centroids_)


# %%
# To summarize what is required for callback management :
#
# * The estimator class must inherit from :class:`~callback.CallbackSupportMixin`.
# * ``fit`` needs to be decorated with the :func:`~base._fit_context` decorator
#   (or the :func:`~callback.with_callback_context` decorator, see the
#   following note).
# * If known, the maximum number of task iterations the ``fit`` method will perform
#   must be set as the ``max_subtasks`` attribute of the callback context object
#   (accessible as the ``_callback_fit_ctx`` attribute of the estimator).
# * The :meth:`~callback._allback_context.CallbackContext.eval_on_fit_begin` method
#   of the callback context must be called at the beginning of ``fit``,
#   providing the estimator as an argument.
# * For each iteration in ``fit``, a subcontext must be spawned with the callback
#   context's :meth:`~callback._allback_context.CallbackContext.subcontext` method.
# * At the end of each iteration, the subcontext's
#   :meth:`~callback._allback_context.CallbackContext.eval_on_fit_task_end` method
#   must be called, providing relevant contextual information through the ``data``
#   dictionary. This call can be used in an if statement controlling a break, as
#   callbacks can perform early stopping, making
#   :meth:`~callback._allback_context.CallbackContext.eval_on_fit_task_end`
#   return ``True``.
#
# .. note ::
#     Note that the :func:`~base._fit_context` decorator also performs
#     other actions on ``fit``, such as parameter validation. If you want only
#     the callback related actions to be performed, you can use the
#     :func:`sklearn.callback.with_callback_context` decorator instead.

# %%
# Attaching callbacks
# -------------------
# Now the ``SimpleKmeans`` estimator can be used with callbacks, for example with
# the :class:`~callback.ProgressBar` callback to monitor progress.

estimator = SimpleKMeans(random_state=rng)
callback = ProgressBar()
estimator.set_callbacks(callback)
estimator.fit(X)

# %%
# Custom Meta-estimator
# -------------------
# Now we demonstrate how to implement a custom meta-estimator that supports
# callbacks. For the example, we implement a simplified version of a grid search,
# where only a list of parameters is provided and searched through instead of a grid.


# The class must again inherit from `CallbackSupportMixin`.
class SimpleGridSearch(CallbackSupportMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, estimator, param_list, n_splits, score_func):
        self.estimator = estimator
        self.param_list = param_list
        self.n_splits = n_splits
        self.score_func = score_func

    # The `fit` method must also be decorated and the callback context must
    # be provided the max number of iterations and its `eval_on_fit_begin`
    # method must be called.
    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        callback_ctx = self._callback_fit_ctx
        callback_ctx.max_subtasks = len(self.param_list)
        callback_ctx.eval_on_fit_begin(estimator=self)

        kf = KFold(n_splits=self.n_splits)
        self.cv_results_ = []

        for i, params in enumerate(self.param_list):
            # A subcontext for the first level of `fit` iterations must be spawned
            outer_subcontext = callback_ctx.subcontext(
                task_name="param iteration", task_id=i, max_subtasks=self.n_splits
            )
            for j, (train_id, test_id) in enumerate(kf.split(X)):
                # This time a second level of `fit` iterations is also used, a
                # second level of subcontext must then be used.
                inner_subcontext = outer_subcontext.subcontext(
                    task_name=f"split {j}", task_id=j
                )

                estimator = clone(self.estimator).set_params(**params)
                # Since a sub-estimator is used, the callbacks must be propagated
                # to that estimator with the `propagate_callbacks` method.
                inner_subcontext.propagate_callbacks(sub_estimator=estimator)

                train_X, test_X = X[train_id], X[test_id]
                train_y, test_y = (
                    (y[train_id], y[test_id]) if y is not None else (None, None)
                )
                estimator.fit(train_X, train_y)
                self.cv_results_.append(
                    (params, f"split_{j}", self.score_func(estimator, test_X, test_y))
                )

                # The inner subcontext's `eval_on_fit_task_end` must be called
                inner_subcontext.eval_on_fit_task_end(
                    estimator=self, data={"X_train": train_X, "y_train": train_y}
                )

            # The outer subcontext's `eval_on_fit_task_end` must be called as well
            if outer_subcontext.eval_on_fit_task_end(
                estimator=self, data={"X_train": X, "y_train": y}
            ):
                break


# %%
# The main difference with a simple estimator is that the callbacks must be
# propagated to the sub-estimators through the corresponding callback subcontext's
# :meth:`~callback._allback_context.CallbackContext.propagate_callbacks` method.
#
# Also it should be noted that each level of nested iterations requires to spawn
# a sub-context and to call that subcontext's
# :meth:`~callback._allback_context.CallbackContext.eval_on_fit_task_end` method
# at the end of the iteration. These methods can be used at any level to enable
# early stopping through a ``break``. Here it does not make much sense to allow
# stopping inside the inner iterations, so it is only implemented on the outer loop,
# but this is only a design choice.

# %%
# Attaching callbacks to the meta-estimator
# -----------------------------------------
# Callbacks can be attached to the meta-estimator similarly to regular
# estimators. The callbacks which respect the
# :class:`~callback.AutoPropagatedCallback` protocol (such as
# :class:`~callback.ProgressBar`) will be propagated to the sub-estimators.

param_list = [{"n_clusters": 5, "max_iter": 20}, {"n_clusters": 4, "max_iter": 50}]


def score_func(estimator, X, y=None):
    return np.sum(estimator.transform(X).min(axis=1))


sub_estimator = SimpleKMeans(random_state=rng)
meta_estimator = SimpleGridSearch(
    estimator=sub_estimator, param_list=param_list, n_splits=4, score_func=score_func
)
callback = ProgressBar(max_estimator_depth=2)
meta_estimator.set_callbacks(callback)
meta_estimator.fit(X)
