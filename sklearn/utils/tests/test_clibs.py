import os

import pytest

from sklearn.utils.testing import SkipTest
from sklearn.utils._clibs import (get_thread_limits, set_thread_limits,
                                  get_openblas_version, thread_limits_context,
                                  _CLibsWrapper)


SKIP_OPENBLAS = get_openblas_version() is None


@pytest.mark.parametrize("clib", _CLibsWrapper.SUPPORTED_CLIBS)
def test_set_thread_limits_dict(clib):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if clib is "openblas" and SKIP_OPENBLAS:
        raise SkipTest("Possible bug in getting maximum number of threads with"
                       " OpenBLAS < 0.2.16 and OpenBLAS does not expose it's "
                       "version before 0.3.4.")

    old_limits = get_thread_limits()

    if old_limits[clib] is not None:
        dynamic_scaling = set_thread_limits(limits={clib: 1})
        assert get_thread_limits()[clib] == 1
        assert dynamic_scaling[clib]

        set_thread_limits(limits={clib: 3})
        new_limits = get_thread_limits()
        assert new_limits[clib] in (3, os.cpu_count(), os.cpu_count() / 2)

        set_thread_limits(limits=old_limits)
        new_limits = get_thread_limits()
        assert new_limits[clib] == old_limits[clib]


@pytest.mark.parametrize("subset", ("all", "blas", "openmp"))
def test_set_thread_limits_subset(subset):
    # Check that the number of threads used by the multithreaded C-libs can be
    # modified dynamically.

    if subset == "all":
        clibs = list(_CLibsWrapper.SUPPORTED_CLIBS.keys())
    elif subset == "blas":
        clibs = ["openblas", "mkl", "mkl_win32"]
    elif subset == "openmp":
        clibs = list(c for c in _CLibsWrapper.SUPPORTED_CLIBS if "openmp" in c)

    if SKIP_OPENBLAS and "openblas" in clibs:
        clibs.remove("openblas")

    old_limits = get_thread_limits()

    dynamic_scaling = set_thread_limits(limits=1, subset=subset)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] == 1
            assert dynamic_scaling[clib]

    set_thread_limits(limits=3, subset=subset)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] in (3, os.cpu_count(), os.cpu_count() / 2)

    set_thread_limits(limits=old_limits)
    new_limits = get_thread_limits()
    for clib in clibs:
        if old_limits[clib] is not None:
            assert new_limits[clib] == old_limits[clib]


def test_set_thread_limits_bad_input():
    # Check that appropriate errors are raised for invalid arguments

    with pytest.raises(ValueError,
                       match="subset must be either 'all', 'blas' "
                             "or 'openmp'"):
        set_thread_limits(limits=1, subset="wrong")

    with pytest.raises(TypeError,
                       match="limits must either be an int or a dict"):
        set_thread_limits(limits=(1, 2, 3))


def test_thread_limit_context():
    old_limits = get_thread_limits()

    with thread_limits_context(limits=1):
        limits = get_thread_limits()
        for clib in limits:
            if old_limits[clib] is None:
                assert limits[clib] is None
            else:
                assert limits[clib] == 1

    assert get_thread_limits() == old_limits
