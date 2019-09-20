#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
#
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Lars Buitinck
#
# License: BSD 3 clause

import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel cimport prange
from libc.math cimport fabs

np.import_array()


def _chi2_kernel_fast(floating[:, :] X,
                      floating[:, :] Y,
                      floating[:, :] result):
    cdef np.npy_intp i, j, k
    cdef np.npy_intp n_samples_X = X.shape[0]
    cdef np.npy_intp n_samples_Y = Y.shape[0]
    cdef np.npy_intp n_features = X.shape[1]
    cdef double res, nom, denom

    with nogil:
        for i in range(n_samples_X):
            for j in range(n_samples_Y):
                res = 0
                for k in range(n_features):
                    denom = (X[i, k] - Y[j, k])
                    nom = (X[i, k] + Y[j, k])
                    if nom != 0:
                        res  += denom * denom / nom
                result[i, j] = -res


def _sparse_manhattan(floating[::1] X_data, int[:] X_indices, int[:] X_indptr,
                      floating[::1] Y_data, int[:] Y_indices, int[:] Y_indptr,
                      double[:, ::1] D):
    """Pairwise L1 distances for CSR matrices.
    Usage:
    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> _sparse_manhattan(X.data, X.indices, X.indptr,
    ...                   Y.data, Y.indices, Y.indptr,
    ...                   D)
    """
    cdef np.npy_intp px, py, i, j, ix, iy
    cdef double d = 0.0

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    # We scan the matrices row by row.
    # Given row px in X and row py in Y, we find the positions (i and j respectively), in .indices where the indices
    # for the two rows start.
    # If the indices (ix and iy) are the same, the corresponding data values are processed and the cursors i and j are
    # advanced.
    # If not, the lowest index is considered. Its associated data value is processed and its cursor is advanced.
    # We proceed like this until one of the cursors hits the end for its row.
    # Then we process all remaining data values in the other row.

    # Below the avoidance of inplace operators is intentional.
    # When prange is used, the inplace operator has a special meaning, i.e. it signals a "reduction"

    for px in prange(m,nogil=True):
        for py in range(n):
            i = X_indptr[px]
            j = Y_indptr[py]
            d = 0.0
            while i < X_indptr[px + 1] and j < Y_indptr[py + 1]:
                if i < X_indptr[px + 1]:
                    ix = X_indices[i]
                if j < Y_indptr[py + 1]:
                    iy = Y_indices[j]

                if ix == iy:
                    d = d + fabs(X_data[i] - Y_data[j])
                    i = i + 1
                    j = j + 1
                elif ix < iy:
                    d = d + fabs(X_data[i])
                    i = i + 1
                else:
                    d = d + fabs(Y_data[j])
                    j = j + 1

            if i == X_indptr[px + 1]:
                while j < Y_indptr[py + 1]:
                    iy = Y_indices[j]
                    d = d + fabs(Y_data[j])
                    j = j + 1
            else:
                while i < X_indptr[px + 1]:
                    ix = X_indices[i]
                    d = d + fabs(X_data[i])
                    i = i + 1

            D[px, py] = d


def _dense_manhattan(floating[:, :] X, floating[:, :] Y, floating[:, :] out):
    cdef:
        floating s = 0.0
        int n_samples_x = X.shape[0]
        int n_samples_y = Y.shape[0]
        int n_features = X.shape[1]
        np.npy_intp i, j

    for i in prange(n_samples_x, nogil=True):
        for j in range(n_samples_y):
            out[i, j] = _dense_manhattan_1d(&X[i, 0], &Y[j, 0], n_features)

cdef floating _dense_manhattan_1d(floating *x, floating *y,
                                  int n_features) nogil:
    cdef:
        int i
        int n = n_features // 4
        int rem = n_features % 4
        floating result = 0

    for i in range(n):
        result += (fabs(x[0] - y[0])
                  +fabs(x[1] - y[1])
                  +fabs(x[2] - y[2])
                  +fabs(x[3] - y[3]))
        x += 4; y += 4

    for i in range(rem):
        result += fabs(x[i] - y[i])

    return result