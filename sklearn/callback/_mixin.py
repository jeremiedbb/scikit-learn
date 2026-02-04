# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import copy

from sklearn.callback._base import Callback


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

    def _from_reconstruction_attributes(self, *, reconstruction_attributes):
        """Return a copy of this estimator as if it was fitted.

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

