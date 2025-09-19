# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import copy

from sklearn.callback._base import Callback
from sklearn.callback._callback_context import CallbackContext


class CallbackSupportMixin:
    """Mixin class to add callback support to an estimator."""

    def set_callbacks(self, callbacks):
        """Set callbacks for the estimator.

        Parameters
        ----------
        callbacks : callback or list of callbacks
            the callbacks to set.

        Returns
        -------
        self : estimator instance
            The estimator instance itself.
        """
        if not isinstance(callbacks, list):
            callbacks = [callbacks]

        if not all(isinstance(callback, Callback) for callback in callbacks):
            raise TypeError("callbacks must follow the Callback protocol.")

        self._skl_callbacks = callbacks

        return self

    def init_callback_context(self, task_name="fit", max_subtasks=None):
        """Initialize the callback context for the estimator.

        Parameters
        ----------
        task_name : str, default='fit'
            The name of the root task.

        max_subtasks : int or None, default=None
            The maximum number of subtasks that can be children of the root task. None
            means the maximum number of subtasks is not known in advance.

        Returns
        -------
        callback_fit_ctx : CallbackContext
            The callback context for the estimator.
        """
        # We don't initialize the callback context during _set_callbacks but in fit
        # because in the future we might want to have callbacks in predict/transform
        # which would require their own context.
        self._callback_fit_ctx = CallbackContext._from_estimator(
            estimator=self, task_name=task_name, task_id=0, max_subtasks=max_subtasks
        )

        return self._callback_fit_ctx

    def _from_reconstruction_attributes(self, *, reconstruction_attributes):
        """Return an as if fitted copy of this estimator

        Parameters
        ----------
        reconstruction_attributes : callable
            A callable that has no arguments and returns the necessary fitted attributes
            to create a working fitted estimator from this instance.

            Using a callable allows lazy evaluation of the potentially costly
            reconstruction attributes.

        Returns
        -------
        fitted_estimator : estimator instance
            The fitted copy of this estimator.
        """
        new_estimator = copy.copy(self)  # XXX deepcopy ?
        for key, val in reconstruction_attributes().items():
            setattr(new_estimator, key, val)
        return new_estimator
