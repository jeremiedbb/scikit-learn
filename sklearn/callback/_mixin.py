# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from sklearn.callback._base import Callback
from sklearn.callback._callback_context import (
    CallbackContext,
    callback_management_context,
)


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

    def __skl_init_callback_context__(self, task_name="fit", max_subtasks=None):
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
        self._callback_fit_ctx = CallbackContext._from_estimator(
            estimator=self, task_name=task_name
        )

        return self._callback_fit_ctx


def fit_callback_context(fit_method):
    """Decorator to run the fit methods within the callback context manager.

    Parameters
    ----------
    fit_method : method
        The fit method to decorate.

    Returns
    -------
    decorated_fit : method
        The decorated fit method.
    """

    def wrapper(estimator, *args, **kwargs):
        with callback_management_context(estimator, fit_method.__name__):
            fit_method(estimator, *args, **kwargs)

    return wrapper
