cimport cython
cimport numpy as np
from cython cimport floating

import numpy as np

np.import_array()

from libc.math cimport exp, fabs, log
from numpy.math cimport EULER


# numpy_float_t and numpy_float2_t are defined in order to accept two different
# dtypes for mean_change and _dirichlet_expectation_1d methods.
# By accepting two different two dtypes, unnecessary memory allocation is
# suppressed along with type conversion in prediction time.
ctypedef fused numpy_float_t:
    np.float32_t
    np.float64_t

ctypedef fused numpy_float2_t:
    np.float32_t
    np.float64_t


def mean_change(np.ndarray[ndim=1, dtype=numpy_float_t] arr_1,
                np.ndarray[ndim=1, dtype=numpy_float2_t] arr_2):
    """Calculate the mean difference between two arrays.

    Equivalent to np.abs(arr_1 - arr2).mean().
    """

    cdef np.float64_t total, diff
    cdef np.npy_intp i, size

    size = arr_1.shape[0]
    total = 0.0
    for i in range(size):
        diff = fabs(arr_1[i] - arr_2[i])
        total += diff

    return total / size


def _dirichlet_expectation_1d(np.ndarray[ndim=1, dtype=numpy_float_t] doc_topic,
                              double doc_topic_prior,
                              np.ndarray[ndim=1, dtype=numpy_float2_t] out):
    """Dirichlet expectation for a single sample:
        exp(E[log(theta)]) for theta ~ Dir(doc_topic)
    after adding doc_topic_prior to doc_topic, in-place.

    Equivalent to
        doc_topic += doc_topic_prior
        out[:] = np.exp(psi(doc_topic) - psi(np.sum(doc_topic)))
    """

    cdef np.float64_t dt, psi_total, total
    cdef np.npy_intp i, size

    size = doc_topic.shape[0]

    # Type of doc_topic and out are identical.
    # So use the type for calculation
    total = 0.0
    for i in range(size):
        dt = doc_topic[i] + doc_topic_prior
        doc_topic[i] = dt
        total += dt
    psi_total = psi(total)

    for i in range(size):
        out[i] = exp(psi(doc_topic[i]) - psi_total)


def _dirichlet_expectation_2d(np.ndarray[ndim=2, dtype=floating] arr):
    """Dirichlet expectation for multiple samples:
    E[log(theta)] for theta ~ Dir(arr).

    Equivalent to psi(arr) - psi(np.sum(arr, axis=1))[:, np.newaxis].

    Note that unlike _dirichlet_expectation_1d, this function doesn't compute
    the exp and doesn't add in the prior.
    """
    cdef np.float64_t row_total, psi_row_total
    cdef np.ndarray[ndim=2, dtype=floating] d_exp
    cdef np.npy_intp i, j, n_rows, n_cols

    n_rows = arr.shape[0]
    n_cols = arr.shape[1]

    d_exp = np.empty_like(arr)
    for i in range(n_rows):
        row_total = 0
        for j in range(n_cols):
            row_total += arr[i, j]
        psi_row_total = psi(row_total)

        for j in range(n_cols):
            d_exp[i, j] = psi(arr[i, j]) - psi_row_total

    return d_exp


# Psi function for positive arguments. Optimized for speed, not accuracy.
#
# After: J. Bernardo (1976). Algorithm AS 103: Psi (Digamma) Function.
# https://www.uv.es/~bernardo/1976AppStatist.pdf
cdef floating psi(floating x) nogil:
    if x <= 1e-6:
        # psi(x) = -EULER - 1/x + O(x)
        return -EULER - 1. / x

    cdef floating r, result = 0

    # psi(x + 1) = psi(x) + 1/x
    while x < 6:
        result -= 1. / x
        x += 1

    # psi(x) = log(x) - 1/(2x) - 1/(12x**2) + 1/(120x**4) - 1/(252x**6)
    #          + O(1/x**8)
    r = 1. / x
    result += log(x) - .5 * r
    r = r * r
    result -= r * ((1./12.) - r * ((1./120.) - r * (1./252.)))
    return result;
