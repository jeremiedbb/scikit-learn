# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import pytest

from sklearn.callback.tests._utils import (
    Estimator,
    NoCloningMetaEstimator,
    NotValidCallback,
    ParentFitEstimator,
    PublicFitDecoratorEstimator,
    TestingAutoPropagatedCallback,
    TestingCallback,
)


@pytest.mark.parametrize(
    "callbacks",
    [
        TestingCallback(),
        [TestingCallback()],
        [TestingCallback(), TestingAutoPropagatedCallback()],
    ],
)
def test_set_callbacks(callbacks):
    """Sanity check for the `set_callbacks` method."""
    estimator = Estimator()

    set_callbacks_return = estimator.set_callbacks(callbacks)
    assert hasattr(estimator, "_skl_callbacks")

    expected_callbacks = [callbacks] if not isinstance(callbacks, list) else callbacks
    assert estimator._skl_callbacks == expected_callbacks

    assert set_callbacks_return is estimator


@pytest.mark.parametrize("callbacks", [None, NotValidCallback()])
def test_set_callbacks_error(callbacks):
    """Check the error message when not passing a valid callback to `set_callbacks`."""
    estimator = Estimator()

    with pytest.raises(TypeError, match="callbacks must follow the Callback protocol."):
        estimator.set_callbacks(callbacks)


def test_init_callback_context():
    """Sanity check for the `__skl_init_callback_context__` method."""
    estimator = Estimator()
    callback_ctx = estimator.__skl_init_callback_context__()

    assert hasattr(estimator, "_callback_fit_ctx")
    assert hasattr(callback_ctx, "_callbacks")


def test_callback_removed_after_fit():
    """Test that the _callback_fit_ctx attribute gets removed after fit."""
    estimator = Estimator()
    estimator.fit()
    assert not hasattr(estimator, "_callback_fit_ctx")


def test_public_fit_decorator():
    """Sanity check of the public fit decorator to manage callback contexts during
    fit."""
    estimator = PublicFitDecoratorEstimator()
    estimator.fit()
    assert not hasattr(estimator, "_callback_fit_ctx")


def test_inheritated_fit():
    """Test with an estimator that uses its parent fit function."""
    estimator = ParentFitEstimator()
    estimator.fit()
    assert not hasattr(estimator, "_callback_fit_ctx")


def test_no_parent_callback_after_fit():
    """Check that the `_parent_callback_ctx` attribute does not survive after fit."""
    estimator = Estimator()
    meta_estimator = NoCloningMetaEstimator(estimator)
    meta_estimator.set_callbacks(TestingAutoPropagatedCallback())
    meta_estimator.fit()
    assert not hasattr(estimator, "_parent_callback_ctx")
