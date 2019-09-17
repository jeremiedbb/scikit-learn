#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
#
# Author: Andreas Mueller <amueller@ais.uni-bonn.de>
#         Lars Buitinck
#
# License: BSD 3 clause

import numpy as np
cimport numpy as np
from cython cimport floating
from cython.parallel import prange
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
    """
    """Pairwise L1 distances for CSR matrices.
    Usage:
    >>> D = np.zeros(X.shape[0], Y.shape[0])
    >>> _sparse_manhattan(X.data, X.indices, X.indptr,
    ...                  Y.data, Y.indices, Y.indptr,
    ...                   D)
    """
    cdef np.npy_intp px, py, i, j, ix, iy
    cdef double d = 0.0

    cdef int m = D.shape[0]
    cdef int n = D.shape[1]

    with nogil:
        for px in prange(m):
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

                D[px,py] = d


def _dense_manhattan(floating[:,:] x,floating[:,:] y, floating[:,:] out):
    cdef double s = 0.0
    cdef np.npy_intp i, j, k
    with nogil:
        for i in prange(x.shape[0]):
            for j in range(y.shape[0]):
                s = 0
                for k in range(x.shape[1]):
                    s = s + fabs(x[i,k] - y[j,k])
                out[i,j]=s
