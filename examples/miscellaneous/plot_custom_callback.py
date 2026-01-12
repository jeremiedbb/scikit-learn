"""
===============
Custom Callback
===============

.. currentmodule:: sklearn

This document shows how to make custom :class:`~callback.Callback` to use with
:term:`estimators` that support callbacks.

Generally speaking, a callback is a function that is provided by the
user to be invoked automatically at specific steps of a process, or to be
triggered by specific events. Callbacks provide a clean mechanism for inserting
custom logic (like monitoring progress or metrics, or implementing early stopping)
without modifying the core algorithm of the process.

In scikit-learn, callbacks take the form of classes following the
:class:`~callback.Callback` protocol. This protocol requires your custom callback
class to implement three specific methods which will be called at specific steps
of the fitting of an estimator or meta-estimator.
These methods are :

* :meth:`~callback._base.Callback.on_fit_begin`, called at the start of the
  :term:`fit` method.
* :meth:`~callback._base.Callback.on_fit_task_end`, called at the end of each
  iteration in ``fit``
* :meth:`~callback._base.Callback.on_fit_end`, called at the end of the ``fit``
  method.

Additionally, if you want your callback to be propagated to sub-estimators in a
:term:`meta-estimator`, it has to also follow the
:class:`~callback.AutoPropagatedCallback` protocol, which only requires to
implement an integer ``max_estimator_depth`` property. This property controls how deep
in the sub-estimator tree the callback will be propagated, meaning it will be
automatically attached to the sub-estimators up to a depth of ``max_estimator_depth``.

For this example's estimator we will use the simplified version of KMeans
introduced in the :ref:`sphx_glr_auto_examples_miscellaneous_plot_callback_support.py`
example.
The callback we will implement will log the different positions of the
``SimpleKmeans`` centroids at each fitting iteration and then produce an animation
showing the evolution of these positions.

First a few imports and some random data for the rest of the script.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

from sklearn.base import BaseEstimator, _fit_context
from sklearn.callback import CallbackSupportMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

n_samples, n_features = 100, 4
rng = np.random.RandomState(42)
X = rng.rand(n_samples, n_features)


# %%
# Custom Callback
# ---------------
# The callback will assume that the estimator it is attached to has a ``centroids_``
# attribute, otherwise it will not do anything.


class CentroidsAnimation:
    def __init__(self, figsize=(6, 6), frame_interval=1000, show_on_fit_end=True):
        self.figsize = figsize
        self.frame_interval = frame_interval
        self.show_on_fit_end = show_on_fit_end

    # The `on_fit_begin` method is used to initialize the callback state,
    # here the attribute logging the centroids positions.
    def on_fit_begin(self, estimator):
        self.position_log = []

    # The `on_fit_task_end` method is used to implement the iterative logic,
    # here the aggregation of the centroids positions is created.
    def on_fit_task_end(self, estimator, context, **kwargs):
        if hasattr(estimator, "centroids_"):
            self.position_log.append(copy(estimator.centroids_))
            if not hasattr(self, "samples"):
                self.samples = copy(kwargs["data"]["X_train"])

    # The `on_fit_end` is used for eventual clean-up or final step of the logic.
    # Here it is used to show the animation.
    def on_fit_end(self, estimator, context):
        if self.show_on_fit_end:
            self.show_animation()

    def show_animation(self):
        if not self.position_log:
            return

        figure = plt.figure(figsize=self.figsize)
        plt.scatter(self.samples[:, 0], self.samples[:, 1], c="lightgrey")
        moving_centroids = plt.scatter(
            self.position_log[0][:, 0], self.position_log[0][:, 1], c="r"
        )

        def update_frame(frame):
            moving_centroids.set_offsets(self.position_log[frame])
            return (moving_centroids,)

        self.animation = FuncAnimation(
            figure,
            update_frame,
            frames=len(self.position_log),
            interval=self.frame_interval,
            blit=True,
        )
        plt.show()


# %%
# .. note ::
#     Note that :meth:`~callback._base.Callback.on_fit_task_end` can be used to
#     implement early stopping by returning ``True``. If the estimator to which
#     the callback is attached accepts early stopping, its `fit` process will be
#     stopped at the iteration where :meth:`~callback._base.Callback.on_fit_task_end`
#     returns ``True``.


# %%
# SimpleKMeans implementation
# ---------------------------
# Here we re-implement our simplified KMeans estimator introduced in the
# :ref:`sphx_glr_auto_examples_miscellaneous_plot_callback_support.py` example.


class SimpleKMeans(CallbackSupportMixin, BaseEstimator):
    _parameter_constraints: dict = {}

    def __init__(self, n_clusters=6, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def _get_labels(self, X):
        return np.argmin(euclidean_distances(X, self.centroids_), axis=1)

    @_fit_context(prefer_skip_nested_validation=False)
    def fit(self, X, y=None):
        callback_ctx = self._callback_fit_ctx
        callback_ctx.max_subtasks = self.max_iter
        callback_ctx.eval_on_fit_begin(estimator=self)

        random_state = check_random_state(self.random_state)
        self.centroids_ = random_state.rand(self.n_clusters, X.shape[1])

        for i in range(self.max_iter):
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

            if (
                subcontext.eval_on_fit_task_end(
                    estimator=self,
                    data={"X_train": X, "y_train": None},
                )
                or has_converged
            ):
                break

        return self

    def predict(self, X):
        check_is_fitted(self)
        return self._get_labels(X)

    def transform(self, X):
        check_is_fitted(self)
        return euclidean_distances(X, self.centroids_)


# %%
# Attaching the callback
# ----------------------
# Now we can attach our callback the estimator.

estimator = SimpleKMeans(random_state=rng)
callback = CentroidsAnimation()
estimator.set_callbacks(callback)
estimator.fit(X)
