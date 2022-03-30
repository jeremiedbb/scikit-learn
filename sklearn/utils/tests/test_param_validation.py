from numbers import Integral, Real

import numpy as np
from scipy.sparse import csr_matrix
import pytest

from sklearn.base import BaseEstimator
from sklearn.utils._param_validation import Interval
from sklearn.utils._param_validation import StrOptions
from sklearn.utils._param_validation import _ArrayLikes
from sklearn.utils._param_validation import _Callables
from sklearn.utils._param_validation import _InstancesOf
from sklearn.utils._param_validation import _NoneConstraint
from sklearn.utils._param_validation import _RandomStates
from sklearn.utils._param_validation import _SparseMatrices
from sklearn.utils._param_validation import make_constraint
from sklearn.utils._param_validation import generate_invalid_param_val
from sklearn.utils._param_validation import validate_params


# Some helpers for the tests
@validate_params({"a": [Real], "b": [Real], "c": [Real], "d": [Real]})
def _func(a, b=0, *args, c, d=None, **kwargs):
    """A function to test the validation of functions."""


class _Class:
    """A class to test the _InstancesOf constraint and the validation of methods."""

    @validate_params({"a": [Real]})
    def _method(self, a):
        """A validated method"""


class _Estimator(BaseEstimator):
    """An estimator to test the validation of estimator parameters."""

    _parameter_constraints = {"a": [Real]}

    def __init__(self, a):
        self.a = a

    def fit(self, X=None, y=None):
        self._validate_params()


@pytest.mark.parametrize("interval_type", [Integral, Real])
def test_interval_range(interval_type):
    """Check the range of values depending on closed."""
    interval = Interval(interval_type, -2, 2, closed="left")
    assert -2 in interval and 2 not in interval

    interval = Interval(interval_type, -2, 2, closed="right")
    assert -2 not in interval and 2 in interval

    interval = Interval(interval_type, -2, 2, closed="both")
    assert -2 in interval and 2 in interval

    interval = Interval(interval_type, -2, 2, closed="neither")
    assert -2 not in interval and 2 not in interval


def test_interval_inf_in_bounds():
    """Check that inf is included if a bound is closed and set to None.

    Only valid for real intervals.
    """
    interval = Interval(Real, 0, None, closed="right")
    assert np.inf in interval

    interval = Interval(Real, None, 0, closed="left")
    assert -np.inf in interval


@pytest.mark.parametrize(
    "params, error, match",
    [
        (
            {"type": Integral, "left": 1.0, "right": 2, "closed": "both"},
            TypeError,
            r"Expecting left to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": 1, "right": 2.0, "closed": "neither"},
            TypeError,
            "Expecting right to be an int for an interval over the integers",
        ),
        (
            {"type": Integral, "left": None, "right": 0, "closed": "left"},
            ValueError,
            r"left can't be None when closed == left",
        ),
        (
            {"type": Integral, "left": 0, "right": None, "closed": "right"},
            ValueError,
            r"right can't be None when closed == right",
        ),
        (
            {"type": Integral, "left": 1, "right": -1, "closed": "both"},
            ValueError,
            r"right can't be less than left",
        ),
    ],
)
def test_interval_errors(params, error, match):
    """Check that informative errors are raised for invalid combination of parameters"""
    with pytest.raises(error, match=match):
        Interval(**params)


def test_stroptions():
    """Sanity check for the StrOptions constraint"""
    options = StrOptions({"a", "b", "c"}, deprecated={"c"})
    assert options.is_satisfied_by("a")
    assert options.is_satisfied_by("c")
    assert not options.is_satisfied_by("d")

    assert "'c' (deprecated)" in str(options)


@pytest.mark.parametrize(
    "constraint",
    [
        Interval(Real, None, 0, closed="left"),
        Interval(Real, 0, None, closed="left"),
        StrOptions({"a", "b", "c"}),
    ],
)
def test_generate_invalid_param_val(constraint):
    """Check that the value generated does not satisfy the constraint"""
    bad_value = generate_invalid_param_val(constraint)
    assert not constraint.is_satisfied_by(bad_value)


@pytest.mark.parametrize(
    "constraint",
    [
        _ArrayLikes,
        _Callables,
        _InstancesOf,
        _NoneConstraint,
        _RandomStates,
        _SparseMatrices,
    ],
)
def test_generate_invalid_param_val_not_error(constraint):
    """Check that the value generated does not satisfy the constraint"""
    with pytest.raises(NotImplementedError):
        generate_invalid_param_val(constraint)


@pytest.mark.parametrize(
    "constraint_declaration, value",
    [
        (Interval(Real, 0, 1, closed="both"), 0.42),
        (Interval(Integral, 0, None, closed="neither"), 42),
        (StrOptions({"a", "b", "c"}), "b"),
        (callable, lambda x: x + 1),
        (None, None),
        ("array-like", [[1, 2], [3, 4]]),
        ("array-like", np.array([[1, 2], [3, 4]])),
        ("sparse matrix", csr_matrix([[1, 2], [3, 4]])),
        ("random_state", 0),
        ("random_state", np.random.RandomState(0)),
        ("random_state", None),
        (_Class, _Class()),
        (int, 1),
        (Real, 0.5),
    ],
)
def test_is_satisified_by(constraint_declaration, value):
    """Sanity check for the is_satisfied_by method"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.is_satisfied_by(value)


@pytest.mark.parametrize(
    "constraint_declaration, expected_constraint_class",
    [
        (Interval(Real, 0, 1, closed="both"), Interval),
        (StrOptions({"option1", "option2"}), StrOptions),
        ("array-like", _ArrayLikes),
        ("sparse matrix", _SparseMatrices),
        ("random_state", _RandomStates),
        (None, _NoneConstraint),
        (callable, _Callables),
        (int, _InstancesOf),
    ],
)
def test_make_constraint(constraint_declaration, expected_constraint_class):
    """Check that make_constraint dispaches to the appropriate constraint class"""
    constraint = make_constraint(constraint_declaration)
    assert constraint.__class__ is expected_constraint_class


def test_make_constraint_unknown():
    with pytest.raises(ValueError, match="Unknown constraint"):
        make_constraint("not a valid constraint")


def test_validate_params():
    """Check that validate_params works no matter how the arguments are passed"""
    with pytest.raises(ValueError, match="The 'a' parameter of _func must be"):
        _func("wrong", c=1)

    with pytest.raises(ValueError, match="The 'b' parameter of _func must be"):
        _func(*[1, "wrong"], c=1)

    with pytest.raises(ValueError, match="The 'c' parameter of _func must be"):
        _func(1, **{"c": "wrong"})

    with pytest.raises(ValueError, match="The 'd' parameter of _func must be"):
        _func(1, c=1, d="wrong")

    # check in the presence of extra positional and keyword args
    with pytest.raises(ValueError, match="The 'b' parameter of _func must be"):
        _func(0, *["wrong", 2, 3], c=4, **{"e": 5})

    with pytest.raises(ValueError, match="The 'c' parameter of _func must be"):
        _func(0, *[1, 2, 3], c="four", **{"e": 5})


def test_validate_params_method():
    """Check that validate_params works with methods"""
    with pytest.raises(ValueError, match="The 'a' parameter of _Class._method must be"):
        _Class()._method("wrong")


def test_validate_params_estimator():
    """Check that validate_params works with Estimator instances"""
    # no validation in init
    est = _Estimator("wrong")

    with pytest.raises(ValueError, match="The 'a' parameter of _Estimator must be"):
        est.fit()
