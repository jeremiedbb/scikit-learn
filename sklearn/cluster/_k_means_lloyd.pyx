# cython: profile=True, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False
# cython: language_level=3
#
# Licence: BSD 3 clause

import numpy as np
cimport numpy as np
cimport openmp
from cython cimport floating
from cython.parallel import prange, parallel
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
from libc.string cimport memset, memcpy
from libc.float cimport DBL_MAX, FLT_MAX

from ..utils.extmath import row_norms
from ..utils._cython_blas cimport _gemm
from ..utils._cython_blas cimport RowMajor, Trans, NoTrans
from ._k_means cimport _relocate_empty_clusters_dense
from ._k_means cimport _relocate_empty_clusters_sparse
from ._k_means cimport _mean_and_center_shift


np.import_array()


cpdef void _lloyd_iter_chunked_dense(np.ndarray[floating, ndim=2, mode='c'] X,
                                     floating[::1] sample_weight,
                                     floating[::1] x_squared_norms,
                                     floating[:, ::1] centers_old,
                                     floating[:, ::1] centers_new,
                                     floating[::1] centers_squared_norms,
                                     floating[::1] weight_in_clusters,
                                     int[::1] labels,
                                     floating[::1] center_shift,
                                     int n_jobs=-1,
                                     bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with dense input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : {float32, float64} array-like, shape (n_samples, n_features)
        The observations to cluster.

    sample_weight : {float32, float64} array-like, shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : {float32, float64} array-like, shape (n_samples,)
        Squared L2 norm of X.

    centers_old : {float32, float64} array-like, shape (n_clusters, n_features)
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : {float32, float64} array-like, shape (n_clusters, n_features)
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    centers_squared_norms : {float32, float64} array-like, shape (n_clusters,)
        Squared L2 norm of the centers.

    weight_in_clusters : {float32, float64} array-like, shape (n_clusters,)
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    labels : int array-like, shape (n_samples,)
        labels assignment.

    center_shift : {float32, float64} array-like, shape (n_clusters,)
        Distance between old and new centers.

    n_jobs : int
        The number of threads to be used by openmp. If -1, openmp will use as
        many as possible.

    update_centers : bool
        - If True, the labels and the new centers will be computed, i.e. runs
          the E-step and the M-step of the algorithm.
        - If False, only the labels will be computed, i.e runs the E-step of
          the algorithm.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]

        # hard-coded number of samples per chunk. Appeared to be close to
        # optimal in all situations.
        int n_samples_chunk = 256 if n_samples > 256 else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_r = n_samples % n_samples_chunk
        int chunk_idx, n_samples_chunk_eff
        int start, end
        int num_threads

        int j, k

    # If n_samples < 256 there's still one chunk of size n_samples_r
    if n_chunks == 0:
        n_chunks = 1
        n_samples_chunk = 0

    # re-initialize all arrays at each iteration
    centers_squared_norms = row_norms(centers_new, squared=True)

    if update_centers:
        memcpy(&centers_old[0, 0], &centers_new[0, 0], n_clusters * n_features * sizeof(floating))
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))

    # set number of threads to be used by openmp
    num_threads = n_jobs if n_jobs != -1 else openmp.omp_get_max_threads()

    with nogil, parallel(num_threads=num_threads):

        for chunk_idx in prange(n_chunks):
            # remaining samples added to last chunk
            if chunk_idx == n_chunks - 1:
                n_samples_chunk_eff = n_samples_chunk + n_samples_r
            else:
                n_samples_chunk_eff = n_samples_chunk

            start = chunk_idx * n_samples_chunk
            end = start + n_samples_chunk_eff

            _update_chunk_dense(
                &X[start, 0],
                sample_weight[start: end],
                x_squared_norms[start: end],
                centers_old,
                centers_new,
                centers_squared_norms,
                weight_in_clusters,
                labels[start: end],
                update_centers)

    if update_centers:
        _relocate_empty_clusters_dense(
            X, sample_weight, centers_new, weight_in_clusters, labels)

        _mean_and_center_shift(
            centers_old, centers_new, weight_in_clusters, center_shift)


cdef void _update_chunk_dense(floating *X,
                              floating[::1] sample_weight,
                              floating[::1] x_squared_norms,
                              floating[:, ::1] centers_old,
                              floating[:, ::1] centers_new,
                              floating[::1] centers_squared_norms,
                              floating[::1] weight_in_clusters,
                              int[::1] labels,
                              bint update_centers) nogil:
    """K-means combined EM step for one dense data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label

        floating *pairwise_distances_ptr = <floating*> malloc(n_samples * n_clusters * sizeof(floating))
        floating[:, ::1] pairwise_distances

    with gil:
        pairwise_distances = <floating[:n_samples, :n_clusters:1]> pairwise_distances_ptr

    # Instead of computing the full pairwise squared distances matrix,
    # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to store
    # the - 2 X.C^T + ||C||² term since the argmin for a given sample only
    # depends on the centers.
    for i in range(n_samples):
        for j in range(n_clusters):
            pairwise_distances[i, j] = centers_squared_norms[j]

    _gemm(RowMajor, NoTrans, Trans, n_samples, n_clusters, n_features,
          -2.0, X, n_features, &centers_old[0, 0], n_features,
          1.0, pairwise_distances_ptr, n_clusters)

    for i in range(n_samples):
        min_sq_dist = pairwise_distances[i, 0]
        label = 0
        for j in range(1, n_clusters):
            sq_dist = pairwise_distances[i, j]
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j
        labels[i] = label

    free(pairwise_distances_ptr)

    if update_centers:
        # The gil is necessary for that to avoid race conditions.
        with gil:
            for i in range(n_samples):
                weight_in_clusters[labels[i]] += sample_weight[i]
                for k in range(n_features):
                    centers_new[labels[i], k] += X[i * n_features + k] * sample_weight[i]


cpdef void _lloyd_iter_chunked_sparse(X,
                                      floating[::1] sample_weight,
                                      floating[::1] x_squared_norms,
                                      floating[:, ::1] centers_old,
                                      floating[:, ::1] centers_new,
                                      floating[::1] centers_squared_norms,
                                      floating[::1] weight_in_clusters,
                                      int[::1] labels,
                                      floating[::1] center_shift,
                                      int n_jobs=-1,
                                      bint update_centers=True):
    """Single iteration of K-means lloyd algorithm with sparse input.

    Update labels and centers (inplace), for one iteration, distributed
    over data chunks.

    Parameters
    ----------
    X : {float32, float64} CSR matrix, shape (n_samples, n_features)
        The observations to cluster.

    sample_weight : {float32, float64} array-like, shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : {float32, float64} array-like, shape (n_samples,)
        Squared L2 norm of X.

    centers_old : {float32, float64} array-like, shape (n_clusters, n_features)
        Centers before previous iteration, placeholder for the centers after
        previous iteration.

    centers_new : {float32, float64} array-like, shape (n_clusters, n_features)
        Centers after previous iteration, placeholder for the new centers
        computed during this iteration.

    centers_squared_norms : {float32, float64} array-like, shape (n_clusters,)
        Squared L2 norm of the centers.

    weight_in_clusters : {float32, float64} array-like, shape (n_clusters,)
        Placeholder for the sums of the weights of every observation assigned
        to each center.

    labels : int array-like, shape (n_samples,)
        labels assignment.

    center_shift : {float32, float64} array-like, shape (n_clusters,)
        Distance between old and new centers.

    n_jobs : int
        The number of threads to be used by openmp. If -1, openmp will use as
        many as possible.

    update_centers : bool
        - If True, the labels and the new centers will be computed.
        - If False, only the labels will be computed.
    """
    cdef:
        int n_samples = X.shape[0]
        int n_features = X.shape[1]
        int n_clusters = centers_new.shape[0]

        # Chosed same as for dense. Does not have the same impact since with
        # sparse data the pairwise distances matrix is not precomputed.
        # However, splitting in chunks is necessary to get parallelism.
        int n_samples_chunk = 256 if n_samples > 256 else n_samples
        int n_chunks = n_samples // n_samples_chunk
        int n_samples_r = n_samples % n_samples_chunk
        int chunk_idx, n_samples_chunk_eff = 0
        int start = 0, end = 0
        int num_threads

        int j, k
        floating alpha

        floating[::1] X_data = X.data
        int[::1] X_indices = X.indices
        int[::1] X_indptr = X.indptr

    # If n_samples < 256 there's still one chunk of size n_samples_r
    if n_chunks == 0:
        n_chunks = 1
        n_samples_chunk = 0

    # re-initialize all arrays at each iteration
    centers_squared_norms = row_norms(centers_new, squared=True)

    if update_centers:
        memcpy(&centers_old[0, 0], &centers_new[0, 0], n_clusters * n_features * sizeof(floating))
        memset(&centers_new[0, 0], 0, n_clusters * n_features * sizeof(floating))
        memset(&weight_in_clusters[0], 0, n_clusters * sizeof(floating))

    # set number of threads to be used by openmp
    num_threads = n_jobs if n_jobs != -1 else openmp.omp_get_max_threads()

    with nogil, parallel(num_threads=num_threads):

        for chunk_idx in prange(n_chunks):
            # remaining samples added to last chunk
            if chunk_idx == n_chunks - 1:
                n_samples_chunk_eff = n_samples_chunk + n_samples_r
            else:
                n_samples_chunk_eff = n_samples_chunk

            start = chunk_idx * n_samples_chunk
            end = start + n_samples_chunk_eff

            _update_chunk_sparse(
                X_data[X_indptr[start]: X_indptr[end]],
                X_indices[X_indptr[start]: X_indptr[end]],
                X_indptr[start: end],
                sample_weight[start: end],
                x_squared_norms[start: end],
                centers_old,
                centers_new,
                centers_squared_norms,
                weight_in_clusters,
                labels[start: end],
                update_centers)

    if update_centers:
        _relocate_empty_clusters_sparse(
            X_data, X_indices, X_indptr, sample_weight,
            centers_new, weight_in_clusters, labels)

        _mean_and_center_shift(
            centers_old, centers_new, weight_in_clusters, center_shift)


cdef void _update_chunk_sparse(floating[::1] X_data,
                               int[::1] X_indices,
                               int[::1] X_indptr,
                               floating[::1] sample_weight,
                               floating[::1] x_squared_norms,
                               floating[:, ::1] centers_old,
                               floating[:, ::1] centers_new,
                               floating[::1] centers_squared_norms,
                               floating[::1] weight_in_clusters,
                               int[::1] labels,
                               bint update_centers) nogil:
    """K-means combined EM step for one sparse data chunk.

    Compute the partial contribution of a single data chunk to the labels and
    centers.
    """
    cdef:
        int n_samples = labels.shape[0]
        int n_clusters = centers_old.shape[0]
        int n_features = centers_old.shape[1]

        floating sq_dist, min_sq_dist
        int i, j, k, label
        floating max_floating = FLT_MAX if floating is float else DBL_MAX
        int s = X_indptr[0]

    # XXX Precompute the pairwise distances matrix is not worth for sparse
    # currently. Should be tested when BLAS (sparse x dense) matrix
    # multiplication is available.
    for i in range(n_samples):
        min_sq_dist = max_floating
        label = 0

        for j in range(n_clusters):
            sq_dist = 0.0
            for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                sq_dist += centers_old[j, X_indices[k]] * X_data[k]

            # Instead of computing the full squared distance with each cluster,
            # ||X - C||² = ||X||² - 2 X.C^T + ||C||², we only need to compute
            # the - 2 X.C^T + ||C||² term since the argmin for a given sample
            # only depends on the centers C.
            sq_dist = centers_squared_norms[j] -2 * sq_dist
            if sq_dist < min_sq_dist:
                min_sq_dist = sq_dist
                label = j

        labels[i] = label

    if update_centers:
        # The gil is necessary for that to avoid race conditions.
        with gil:
            for i in range(n_samples):
                weight_in_clusters[labels[i]] += sample_weight[i]
                for k in range(X_indptr[i] - s, X_indptr[i + 1] - s):
                    centers_new[labels[i], X_indices[k]] += X_data[k] * sample_weight[i]
