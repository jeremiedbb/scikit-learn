"""
==============================================
Supporting callbacks in third party estimators
==============================================

.. currentmodule:: sklearn

This document shows how to make third party :term:`estimators` and
:term:`meta-estimators` compatible with the callback mechanisms supported
by scikit-learn.

Generally speaking, a callback is a function that is provided by the
user to be called at specific steps of a process, or to be
triggered by specific events. Callbacks provide a clean mechanism for inserting
custom logic like monitoring progress or metrics,
without modifying the core algorithm of the process.

In scikit-learn, callbacks take the form of classes following a protocol.
This protocol requires the callback classes to implement specific methods which
will be called at specific steps of the fitting of an estimator or a meta-estimator.
These specific methods are :meth:`~callback._base.Callback.on_fit_begin`,
:meth:`~callback._base.Callback.on_fit_task_end` and
:meth:`~callback._base.Callback.on_fit_end`, which are respectively called at
the start of the :term:`fit` method, at the end of each task in ``fit``
and at the end of the ``fit`` method.

In order to support the callbacks, estimators need to manipulate
:class:`~callback.CallbackContext` objects. As the name implies, these objects hold
the contextual information necessary to run the callback methods. They are also
responsible for triggering the methods of the callback at the right time.

First a few imports and some random data for the rest of the script.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

import numpy as np

from sklearn.base import BaseEstimator, clone
from sklearn.callback import CallbackSupportMixin, ProgressBar, with_callback_context
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
# First, let's implement our SimpleKmeans without the callback support.


class SimpleKmeans(BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, n_clusters=6, n_iter=100, random_state=None):
        self.n_clusters = n_clusters  # the number of clusters (and thus of centroids)
        self.n_iter = n_iter  # the number of iterations to find the centroids
        self.random_state = random_state  # to control the randomness of the centroids'
        # initialization

    def _get_labels(self, X):
        # Get the index of the closest centroid for each point in X.
        return np.argmin(euclidean_distances(X, self.centroids_), axis=1)

    def fit(self, X, y=None):  # `y` is not used but we need to declare it to adhere to
        # scikit-learn's estimators fit convention.
        random_state = check_random_state(self.random_state)
        # Randomnly initialize the centroids.
        self.centroids_ = random_state.rand(self.n_clusters, X.shape[1])

        for i in range(self.n_iter):
            # The fit iterations consist in getting the cluster label of each data point
            # according to their closest centroid, and then updating the centroids as
            # the center of each cluster.
            labels = self._get_labels(X)

            for k in range(self.n_clusters):
                # For each centroid, if its cluster is not empty, its coordinates
                # are updated with the coordinates of the cluster centers.
                if (labels == k).any():
                    self.centroids_[k] = X[labels == k].mean(axis=0)

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self._get_labels(X)

    def transform(self, X):
        check_is_fitted(self)
        return euclidean_distances(X, self.centroids_)


# %%
# Now let's add all the elements necessary to support callbacks.


# First things first, the estimator must inherit from the
# `CallbackSupportMixin` class.
class SimpleKMeans(CallbackSupportMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, n_clusters=6, n_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.n_iter = n_iter
        self.random_state = random_state

    def _get_labels(self, X):
        return np.argmin(euclidean_distances(X, self.centroids_), axis=1)

    # Then the `fit` function must be decorated with the `with_callback_context`
    # decorator, which will create the `CallbackContext` object.
    @with_callback_context
    def fit(self, X, y=None):
        # The `CallbackContext` object is accessible in `fit` as the
        # `_callback_fit_ctx` attribute of the estimator.
        callback_ctx = self._callback_fit_ctx
        # As soon as known (if ever), the `fit` method should set the maximum
        # number of iterative tasks as an attribute of the callback context.
        callback_ctx.max_subtasks = self.n_iter
        # Then the callback context's `eval_on_fit_begin` method must be called.
        # It will trigger all the callbacks' `on_fit_begin` methods.
        callback_ctx.eval_on_fit_begin(estimator=self)

        random_state = check_random_state(self.random_state)
        self.centroids_ = random_state.rand(self.n_clusters, X.shape[1])

        # The callback context's `eval_on_fit_task_end` method must be called
        # after each task of `fit`, here after each iteration of the loop
        # updating the centroids.

        for i in range(self.n_iter):
            # For each of these task, a subcontext must be spawned with the
            # callback context's `subcontext` method.
            subcontext = callback_ctx.subcontext(task_id=i, task_name="fit iteration")

            labels = self._get_labels(X)

            for k in range(self.n_clusters):
                if (labels == k).any():
                    self.centroids_[k] = X[labels == k].mean(axis=0)

            # After each task, the `eval_on_fit_task_end` method of its
            # callback context must be called. It will trigger all the callbacks'
            # `on_fit_task_end` methods. Data relative to the current task
            # can be provided through the `data` argument.
            if subcontext.eval_on_fit_task_end(
                estimator=self,
                data={"X_train": X, "y_train": None},
            ):
                # The `eval_on_fit_task_end` method returns a boolean, which will
                # be set to True if any of the callbacks' `on_fit_task_end` methods
                # return True. This allows to implement early stopping with
                # callbacks. Thus the `eval_on_fit_task_end` method must be used
                # in an `if` / `break` block.
                break

        # The callback context's `eval_on_fit_end` method does not need to be
        # called here as it is called automatically in the decorator, after fit
        # finishes, even if it crashes. Thus the callbacks will call their
        # `on_fit_end` methods even if `fit` does not complete correctly.
        return self

    def predict(self, X):
        check_is_fitted(self)
        return self._get_labels(X)

    def transform(self, X):
        check_is_fitted(self)
        return euclidean_distances(X, self.centroids_)


# %%
# Registering callbacks to the custom estimator
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
# Again, let's start with the implementation without the callback support.


class SimpleGridSearch(BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, estimator, param_list, n_splits, score_func):
        self.estimator = estimator
        self.param_list = param_list  # the list of parameter combinations to iterate
        # over
        self.n_splits = n_splits  # the number of splits for the KFold to apply.
        self.score_func = score_func  # the scoring function

    def fit(self, X, y=None):
        # We make a KFold split to evaluate each parameter combination on
        # multiple folds.
        kf = KFold(n_splits=self.n_splits)
        self.cv_results_ = []  # this attribute will hold the score values for each
        # parameter combination and fold.

        # We iterate on the parameter combinations and the folds, computing
        # a score value for each param combination and fold.
        for i, params in enumerate(self.param_list):
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
                # An estimator is initialized with the parameter combination.
                estimator = clone(self.estimator).set_params(**params)
                # The split of the current fold is applied to the data.
                train_X, test_X = X[train_idx], X[test_idx]
                train_y, test_y = (
                    (y[train_idx], y[test_idx]) if y is not None else (None, None)
                )
                # The estimator is fitted.
                estimator.fit(train_X, train_y)
                # Its score is computed.
                score = self.score_func(estimator, test_X, test_y)
                # The results are aggregated as a tuple in the attribute.
                self.cv_results_.append((params, f"split_{j}", score))

        return self


# %%
# Now let's update the class to support callbacks.


# The class must again inherit from `CallbackSupportMixin`.
class SimpleGridSearch(CallbackSupportMixin, BaseEstimator):  # noqa: F811
    _parameter_constraints: dict = {}

    def __init__(self, estimator, param_list, n_splits, score_func):
        self.estimator = estimator
        self.param_list = param_list
        self.n_splits = n_splits
        self.score_func = score_func

    # The `fit` method must also be decorated.
    @with_callback_context
    def fit(self, X, y=None):
        # The callback context can also be accessed as an attribute.
        callback_ctx = self._callback_fit_ctx
        # The `max_subtasks` attribute must again be declared.
        callback_ctx.max_subtasks = len(self.param_list)
        # The `eval_on_fit_begin` method must again be called.
        callback_ctx.eval_on_fit_begin(estimator=self)

        kf = KFold(n_splits=self.n_splits)
        self.cv_results_ = []

        # The tasks of the `fit` function are here the iterations on two
        # levels of nested loops. Therefore, two levels of subcontexts
        # must be used. Each level will need its subcontext to call its
        # `eval_on_fit_task_end` method. These methods can be used at any
        # level to enable early stopping through a `break`. Here it does not
        # make much sense to allow stopping between folds inside the inner loop,
        # so it is only implemented on the outer loop, but this is only a design
        # choice.

        for i, params in enumerate(self.param_list):
            # A subcontext for the first level of `fit` iterations must be spawned
            outer_subcontext = callback_ctx.subcontext(
                task_name="param iteration", task_id=i, max_subtasks=self.n_splits
            )
            for j, (train_idx, test_idx) in enumerate(kf.split(X)):
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
                score = self.score_func(estimator, test_X, test_y)
                self.cv_results_.append((params, f"split_{j}", score))

                # The inner subcontext's `eval_on_fit_task_end` must be called
                inner_subcontext.eval_on_fit_task_end(
                    estimator=self, data={"X_train": train_X, "y_train": train_y}
                )

            # The outer subcontext's `eval_on_fit_task_end` must be called as well
            # and is used with an `if` / `break` to eventually enable early
            # stopping.
            if outer_subcontext.eval_on_fit_task_end(
                estimator=self, data={"X_train": X, "y_train": y}
            ):
                break

        # Again, the callback context's `eval_on_fit_end` is taken care of
        # automatically in the decorator.
        return self


# %%
# The main difference with a simple estimator is that the callbacks must be
# propagated to the sub-estimators through the corresponding callback subcontext's
# :meth:`~callback._callback_context.CallbackContext.propagate_callbacks` method.


# %%
# Registering callbacks to the meta-estimator
# -------------------------------------------
# Callbacks are registered to a meta-estimator the same way as to regular
# estimators. The callbacks which respect the
# :class:`~callback.AutoPropagatedCallback` protocol (such as
# :class:`~callback.ProgressBar`) will be propagated to the sub-estimators.


param_list = [{"n_clusters": 5, "n_iter": 20}, {"n_clusters": 4, "n_iter": 50}]


def score_func(estimator, X, y=None):
    return np.sum(estimator.transform(X).min(axis=1))


sub_estimator = SimpleKMeans(random_state=rng)
meta_estimator = SimpleGridSearch(
    estimator=sub_estimator, param_list=param_list, n_splits=4, score_func=score_func
)
callback = ProgressBar(max_estimator_depth=2)
meta_estimator.set_callbacks(callback)
meta_estimator.fit(X)
