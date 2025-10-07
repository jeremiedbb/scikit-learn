# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import functools

from sklearn.callback._base import Callback
from sklearn.callback._callback_context import CallbackContext


class CallbackSupportMixin:
    """Mixin class to add callback support to an estimator."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for fit_method in ("fit", "fit_transform", "fit_predict", "partial_fit"):
            if hasattr(cls, fit_method):
                setattr(cls, fit_method, _fit_callback(getattr(cls, fit_method)))

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


def _fit_callback(fit_method):
    """Decorator to initialize the callback context for the fit methods."""

    @functools.wraps(fit_method)
    def callback_wrapper(estimator, *args, **kwargs):
        ctx_already_existing = hasattr(estimator, "__sklearn_callback_fit_ctx__")
        if not ctx_already_existing:
            estimator.__sklearn_callback_fit_ctx__ = CallbackContext._from_estimator(
                estimator
            )

        try:
            return fit_method(estimator, *args, **kwargs)
        finally:
            if not ctx_already_existing:
                estimator.__sklearn_callback_fit_ctx__.eval_on_fit_end(estimator)
                del estimator.__sklearn_callback_fit_ctx__
                if hasattr(estimator, "_parent_callback_ctx"):
                    del estimator._parent_callback_ctx

    return callback_wrapper
