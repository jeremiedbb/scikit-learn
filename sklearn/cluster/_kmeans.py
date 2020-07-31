"""K-means clustering"""

# Authors: Gael Varoquaux <gael.varoquaux@normalesup.org>
#          Thomas Rueckstiess <ruecksti@in.tum.de>
#          James Bergstra <james.bergstra@umontreal.ca>
#          Jan Schlueter <scikit-learn@jan-schlueter.de>
#          Nelle Varoquaux
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Olivier Grisel <olivier.grisel@ensta.org>
#          Mathieu Blondel <mathieu@mblondel.org>
#          Robert Layton <robertlayton@gmail.com>
# License: BSD 3 clause

import warnings

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from ..base import BaseEstimator, ClusterMixin, TransformerMixin
from ..metrics.pairwise import euclidean_distances
from ..utils.extmath import row_norms, stable_cumsum
from ..utils.sparsefuncs_fast import assign_rows_csr
from ..utils.sparsefuncs import mean_variance_axis
from ..utils.validation import _deprecate_positional_args
from ..utils import check_array
from ..utils import check_random_state
from ..utils import deprecated
from ..utils.validation import check_is_fitted, _check_sample_weight
from ..utils._openmp_helpers import _openmp_effective_n_threads
from ..exceptions import ConvergenceWarning
from ._k_means_common import _inertia_dense
from ._k_means_common import _inertia_sparse
from ._k_means_minibatch import _minibatch_update_dense
from ._k_means_minibatch import _minibatch_update_sparse
from ._k_means_lloyd import lloyd_iter_chunked_dense
from ._k_means_lloyd import lloyd_iter_chunked_sparse
from ._k_means_elkan import init_bounds_dense
from ._k_means_elkan import init_bounds_sparse
from ._k_means_elkan import elkan_iter_chunked_dense
from ._k_means_elkan import elkan_iter_chunked_sparse


###############################################################################
# Initialization heuristic

def _k_init(X, n_clusters, x_squared_norms, random_state, n_local_trials=None):
    """Init n_clusters seeds according to k-means++

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The data to pick seeds for. To avoid memory copy, the input data
        should be double precision (dtype=np.float64).

    n_clusters : int
        The number of seeds to choose

    x_squared_norms : ndarray of shape (n_samples,)
        Squared Euclidean norm of each data point.

    random_state : RandomState instance
        The generator used to initialize the centers.
        See :term:`Glossary <random_state>`.

    n_local_trials : int, default=None
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-mean clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, n_features = X.shape

    centers = np.empty((n_clusters, n_features), dtype=X.dtype)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    # Pick first center randomly
    center_id = random_state.randint(n_samples)
    if sp.issparse(X):
        centers[0] = X[center_id].toarray()
    else:
        centers[0] = X[center_id]

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = euclidean_distances(
        centers[0, np.newaxis], X, Y_norm_squared=x_squared_norms,
        squared=True)
    current_pot = closest_dist_sq.sum()

    # Pick the remaining n_clusters-1 points
    for c in range(1, n_clusters):
        # Choose center candidates by sampling with probability proportional
        # to the squared distance to the closest existing center
        rand_vals = random_state.random_sample(n_local_trials) * current_pot
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq),
                                        rand_vals)
        # XXX: numerical imprecision can result in a candidate_id out of range
        np.clip(candidate_ids, None, closest_dist_sq.size - 1,
                out=candidate_ids)

        # Compute distances to center candidates
        distance_to_candidates = euclidean_distances(
            X[candidate_ids], X, Y_norm_squared=x_squared_norms, squared=True)

        # update closest distances squared and potential for each candidate
        np.minimum(closest_dist_sq, distance_to_candidates,
                   out=distance_to_candidates)
        candidates_pot = distance_to_candidates.sum(axis=1)

        # Decide which candidate is the best
        best_candidate = np.argmin(candidates_pot)
        current_pot = candidates_pot[best_candidate]
        closest_dist_sq = distance_to_candidates[best_candidate]
        best_candidate = candidate_ids[best_candidate]

        # Permanently add best center candidate found in local tries
        if sp.issparse(X):
            centers[c] = X[best_candidate].toarray()
        else:
            centers[c] = X[best_candidate]

    return centers


###############################################################################
# K-means batch estimation by EM (expectation maximization)

def _tolerance(X, tol):
    """Return a tolerance which is independent of the dataset"""
    if tol == 0:
        return 0
    if sp.issparse(X):
        variances = mean_variance_axis(X, axis=0)[1]
    else:
        variances = np.var(X, axis=0)
    return np.mean(variances) * tol


@_deprecate_positional_args
def k_means(X, n_clusters, *, sample_weight=None, init='k-means++',
            precompute_distances='deprecated', n_init=10, max_iter=300,
            verbose=False, tol=1e-4, random_state=None, copy_x=True,
            n_jobs='deprecated', algorithm="auto", return_n_iter=False):
    """K-means clustering algorithm.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------
    X : {array-like, sparse} matrix of shape (n_samples, n_features)
        The observations to cluster. It must be noted that the data
        will be converted to C ordering, which will cause a memory copy
        if the given data is not C-contiguous.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    sample_weight : array-like of shape (n_samples,), default=None
        The weights for each observation in X. If None, all observations
        are assigned equal weight

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    precompute_distances : {'auto', True, False}
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances

        False : never precompute distances

        .. deprecated:: 0.23
            'precompute_distances' was deprecated in version 0.23 and will be
            removed in 0.25. It has no effect.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    n_jobs : int, default=None
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

        ``None`` or ``-1`` means using all processors.

        .. deprecated:: 0.23
            ``n_jobs`` was deprecated in version 0.23 and will be removed in
            0.25.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).

        For now "auto" (kept for backward compatibiliy) chooses "elkan" but it
        might change in the future for a better heuristic.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    best_n_iter : int
        Number of iterations corresponding to the best results.
        Returned only if `return_n_iter` is set to True.
    """
    est = KMeans(
        n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
        verbose=verbose, precompute_distances=precompute_distances, tol=tol,
        random_state=random_state, copy_x=copy_x, n_jobs=n_jobs,
        algorithm=algorithm
    ).fit(X, sample_weight=sample_weight)
    if return_n_iter:
        return est.cluster_centers_, est.labels_, est.inertia_, est.n_iter_
    else:
        return est.cluster_centers_, est.labels_, est.inertia_


def _kmeans_single_elkan(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1):
    """A single run of k-means elkan, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : array-like of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : array-like, default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_samples = X.shape[0]
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    labels = np.full(n_samples, -1, dtype=np.int32)
    center_half_distances = euclidean_distances(centers) / 2
    distance_next_center = np.partition(np.asarray(center_half_distances),
                                        kth=1, axis=0)[1]
    upper_bounds = np.zeros(n_samples, dtype=X.dtype)
    lower_bounds = np.zeros((n_samples, n_clusters), dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        init_bounds = init_bounds_sparse
        elkan_iter = elkan_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        init_bounds = init_bounds_dense
        elkan_iter = elkan_iter_chunked_dense
        _inertia = _inertia_dense

    init_bounds(X, centers, center_half_distances,
                labels, upper_bounds, lower_bounds)

    for i in range(max_iter):
        elkan_iter(X, sample_weight, centers, centers_new,
                   weight_in_clusters, center_half_distances,
                   distance_next_center, upper_bounds, lower_bounds,
                   labels, center_shift, n_threads)

        # compute new pairwise distances between centers and closest other
        # center of each center for next iterations
        center_half_distances = euclidean_distances(centers_new) / 2
        distance_next_center = np.partition(
            np.asarray(center_half_distances), kth=1, axis=0)[1]

        if verbose:
            inertia = _inertia(X, sample_weight, centers, labels, n_threads)
            print(f"Iteration {i}, inertia {inertia}")

        centers, centers_new = centers_new, centers

        center_shift_tot = (center_shift**2).sum()
        if center_shift_tot <= tol:
            if verbose:
                print(f"Converged at iteration {i}: center shift "
                      f"{center_shift_tot} within tolerance {tol}.")
            break

    if center_shift_tot > 0:
        # rerun E-step so that predicted labels match cluster centers
        elkan_iter(X, sample_weight, centers, centers, weight_in_clusters,
                   center_half_distances, distance_next_center,
                   upper_bounds, lower_bounds, labels, center_shift,
                   n_threads, update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _kmeans_single_lloyd(X, sample_weight, centers_init, max_iter=300,
                         verbose=False, x_squared_norms=None, tol=1e-4,
                         n_threads=1):
    """A single run of k-means lloyd, assumes preparation completed prior.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The observations to cluster. If sparse matrix, must be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers_init : ndarray of shape (n_clusters, n_features)
        The initial centers.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm to run.

    verbose : bool, default=False
        Verbosity mode

    x_squared_norms : ndarray of shape (n_samples,), default=None
        Precomputed x_squared_norms.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    centroid : ndarray of shape (n_clusters, n_features)
        Centroids found at the last iteration of k-means.

    label : ndarray of shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """
    n_clusters = centers_init.shape[0]

    # Buffers to avoid new allocations at each iteration.
    centers = centers_init
    centers_new = np.zeros_like(centers)
    labels = np.full(X.shape[0], -1, dtype=np.int32)
    weight_in_clusters = np.zeros(n_clusters, dtype=X.dtype)
    center_shift = np.zeros(n_clusters, dtype=X.dtype)

    if sp.issparse(X):
        lloyd_iter = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        lloyd_iter = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    # Threadpoolctl context to limit the number of threads in second level of
    # nested parallelism (i.e. BLAS) to avoid oversubsciption.
    with threadpool_limits(limits=1, user_api="blas"):
        for i in range(max_iter):
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads)

            if verbose:
                inertia = _inertia(X, sample_weight, centers, labels,
                                   n_threads)
                print(f"Iteration {i}, inertia {inertia}.")

            centers, centers_new = centers_new, centers

            center_shift_tot = (center_shift**2).sum()
            if center_shift_tot <= tol:
                if verbose:
                    print("Converged at iteration {0}: "
                          "center shift {1} within tolerance {2}"
                          .format(i, center_shift_tot, tol))
                break

        if center_shift_tot > 0:
            # rerun E-step so that predicted labels match cluster centers
            lloyd_iter(X, sample_weight, x_squared_norms, centers, centers,
                       weight_in_clusters, labels, center_shift, n_threads,
                       update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia, centers, i + 1


def _labels_inertia(X, sample_weight, x_squared_norms, centers,
                    n_threads=None):
    """E step of the K-means EM algorithm.

    Compute the labels and the inertia of the given samples and centers.

    Parameters
    ----------
    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The input samples to assign to the labels. If sparse matrix, must
        be in CSR format.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    x_squared_norms : ndarray of shape (n_samples,)
        Precomputed squared euclidean norm of each data point, to speed up
        computations.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers.

    n_threads : int, default=None
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

    Returns
    -------
    labels : ndarray of shape (n_samples,)
        The resulting assignment.

    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    n_samples = X.shape[0]
    n_clusters = centers.shape[0]

    n_threads = _openmp_effective_n_threads(n_threads)

    labels = np.full(n_samples, -1, dtype=np.int32)
    weight_in_clusters = np.zeros(n_clusters, dtype=centers.dtype)
    center_shift = np.zeros_like(weight_in_clusters)

    if sp.issparse(X):
        _labels = lloyd_iter_chunked_sparse
        _inertia = _inertia_sparse
    else:
        _labels = lloyd_iter_chunked_dense
        _inertia = _inertia_dense

    _labels(X, sample_weight, x_squared_norms, centers, centers,
            weight_in_clusters, labels, center_shift, n_threads,
            update_centers=False)

    inertia = _inertia(X, sample_weight, centers, labels, n_threads)

    return labels, inertia


def _labels_inertia_threadpool_limit(X, sample_weight, x_squared_norms,
                                     centers, n_threads=None):
    """Same as _labels_inertia but in a threadpool_limits context."""
    with threadpool_limits(limits=1, user_api="blas"):
        labels, inertia = _labels_inertia(X, sample_weight, x_squared_norms,
                                          centers, n_threads)

    return labels, inertia


class KMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """K-Means clustering.

    Read more in the :ref:`User Guide <k_means>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    n_init : int, default=10
        Number of time the k-means algorithm will be run with different
        centroid seeds. The final results will be the best output of
        n_init consecutive runs in terms of inertia.

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.
        It's not advised to set `tol=0` since convergence might never be
        declared due to rounding errors. Use a very small number instead.

    precompute_distances : {'auto', True, False}, default='auto'
        Precompute distances (faster but takes more memory).

        'auto' : do not precompute distances if n_samples * n_clusters > 12
        million. This corresponds to about 100MB overhead per job using
        double precision.

        True : always precompute distances.

        False : never precompute distances.

        .. deprecated:: 0.23
            'precompute_distances' was deprecated in version 0.22 and will be
            removed in 0.25. It has no effect.

    verbose : int, default=0
        Verbosity mode.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    copy_x : bool, default=True
        When pre-computing distances it is more numerically accurate to center
        the data first. If copy_x is True (default), then the original data is
        not modified. If False, the original data is modified, and put back
        before the function returns, but small numerical differences may be
        introduced by subtracting and then adding the data mean. Note that if
        the original data is not C-contiguous, a copy will be made even if
        copy_x is False. If the original data is sparse, but not in CSR format,
        a copy will be made even if copy_x is False.

    n_jobs : int, default=None
        The number of OpenMP threads to use for the computation. Parallelism is
        sample-wise on the main cython loop which assigns each sample to its
        closest center.

        ``None`` or ``-1`` means using all processors.

        .. deprecated:: 0.23
            ``n_jobs`` was deprecated in version 0.23 and will be removed in
            0.25.

    algorithm : {"auto", "full", "elkan"}, default="auto"
        K-means algorithm to use. The classical EM-style algorithm is "full".
        The "elkan" variation is more efficient on data with well-defined
        clusters, by using the triangle inequality. However it's more memory
        intensive due to the allocation of an extra array of shape
        (n_samples, n_clusters).

        For now "auto" (kept for backward compatibiliy) chooses "elkan" but it
        might change in the future for a better heuristic.

        .. versionchanged:: 0.18
            Added Elkan algorithm

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers. If the algorithm stops before fully
        converging (see ``tol`` and ``max_iter``), these will not be
        consistent with ``labels_``.

    labels_ : ndarray of shape (n_samples,)
        Labels of each point

    inertia_ : float
        Sum of squared distances of samples to their closest cluster center,
        weighted by the sample weights if provided.

    n_iter_ : int
        Number of iterations run.

    See also
    --------

    MiniBatchKMeans
        Alternative online implementation that does incremental updates
        of the centers positions using mini-batches.
        For large scale learning (say n_samples > 10k) MiniBatchKMeans is
        probably much faster than the default batch implementation.

    Notes
    -----
    The k-means problem is solved using either Lloyd's or Elkan's algorithm.

    The average complexity is given by O(k n T), were n is the number of
    samples and T is the number of iteration.

    The worst case complexity is given by O(n^(k+2/p)) with
    n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
    'How slow is the k-means method?' SoCG2006)

    In practice, the k-means algorithm is very fast (one of the fastest
    clustering algorithms available), but it falls in local minima. That's why
    it can be useful to restart it several times.

    If the algorithm stops before fully converging (because of ``tol`` or
    ``max_iter``), ``labels_`` and ``cluster_centers_`` will not be consistent,
    i.e. the ``cluster_centers_`` will not be the means of the points in each
    cluster. Also, the estimator will reassign ``labels_`` after the last
    iteration to make ``labels_`` consistent with ``predict`` on the training
    set.

    Examples
    --------

    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [10, 2], [10, 4], [10, 0]])
    >>> kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
    >>> kmeans.labels_
    array([1, 1, 1, 0, 0, 0], dtype=int32)
    >>> kmeans.predict([[0, 0], [12, 3]])
    array([1, 0], dtype=int32)
    >>> kmeans.cluster_centers_
    array([[10.,  2.],
           [ 1.,  2.]])
    """
    @_deprecate_positional_args
    def __init__(self, n_clusters=8, *, init='k-means++', n_init=10,
                 max_iter=300, tol=1e-4, precompute_distances='deprecated',
                 verbose=0, random_state=None, copy_x=True,
                 n_jobs='deprecated', algorithm='auto'):

        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.precompute_distances = precompute_distances
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.n_jobs = n_jobs
        self.algorithm = algorithm

    def _check_params(self, X):
        # precompute_distances
        if self.precompute_distances != 'deprecated':
            warnings.warn("'precompute_distances' was deprecated in version "
                          "0.23 and will be removed in 0.25. It has no "
                          "effect", FutureWarning)

        # n_jobs
        if self.n_jobs != 'deprecated':
            warnings.warn("'n_jobs' was deprecated in version 0.23 and will be"
                          " removed in 0.25.", FutureWarning)
            self._n_threads = self.n_jobs
        else:
            self._n_threads = None
        self._n_threads = _openmp_effective_n_threads(self._n_threads)

        # n_init
        if self.n_init <= 0:
            raise ValueError(
                f"n_init should be > 0, got {self.n_init} instead.")
        self._n_init = self.n_init

        # max_iter
        if self.max_iter <= 0:
            raise ValueError(
                f"max_iter should be > 0, got {self.max_iter} instead.")

        # n_clusters
        if X.shape[0] < self.n_clusters:
            raise ValueError(f"n_samples={X.shape[0]} should be >= "
                             f"n_clusters={self.n_clusters}.")

        # tol
        self._tol = _tolerance(X, self.tol)

        # algorithm
        if self.algorithm not in ("auto", "full", "elkan"):
            raise ValueError(f"Algorithm must be 'auto', 'full' or 'elkan', "
                             f"got {self.algorithm} instead.")

        self._algorithm = self.algorithm
        if self._algorithm == "auto":
            self._algorithm = "full" if self.n_clusters == 1 else "elkan"
        if self._algorithm == "elkan" and self.n_clusters == 1:
            warnings.warn("algorithm='elkan' doesn't make sense for a single "
                          "cluster. Using 'full' instead.", RuntimeWarning)
            self._algorithm = "full"

        # init
        if not (hasattr(self.init, '__array__') or callable(self.init)
                or (isinstance(self.init, str)
                    and self.init in ["k-means++", "random"])):
            raise ValueError(
                f"init should be either 'k-means++', 'random', a ndarray or a "
                f"callable, got '{self.init}' instead.")

        if hasattr(self.init, '__array__') and self._n_init != 1:
            warnings.warn(
                f"Explicit initial center position passed: performing only"
                f" one init in {self.__class__.__name__} instead of "
                f"n_init={self._n_init}.", RuntimeWarning, stacklevel=2)
            self._n_init = 1

    def _validate_center_shape(self, X, centers):
        """Check if centers is compatible with X and n_clusters"""
        if centers.shape[0] != self.n_clusters:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of clusters {self.n_clusters}.")
        if centers.shape[1] != X.shape[1]:
            raise ValueError(
                f"The shape of the initial centers {centers.shape} does not "
                f"match the number of features of the data {X.shape[1]}.")

    def _check_test_data(self, X):
        X = check_array(X, accept_sparse='csr', dtype=[np.float64, np.float32],
                        order='C', accept_large_sparse=False)
        n_samples, n_features = X.shape
        expected_n_features = self.cluster_centers_.shape[1]
        if not n_features == expected_n_features:
            raise ValueError(
                f"Incorrect number of features. Got {n_features} features, "
                f"expected {expected_n_features}.")

        return X

    def _init_centroids(self, X, x_squared_norms, init, random_state,
                        init_size=None):
        """Compute the initial centroids

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        x_squared_norms : ndarray of shape (n_samples,)
            Squared euclidean norm of each data point. Pass it if you have it
            at hands already to avoid it being recomputed here.

        init : {'k-means++', 'random'}, callable or ndarray of shape \
                (n_clusters, n_features)
            Method for initialization.

        random_state : RandomState instance
            Determines random number generation for centroid initialization.
            See :term:`Glossary <random_state>`.

        init_size : int, default=None
            Number of samples to randomly sample for speeding up the
            initialization (sometimes at the expense of accuracy).

        Returns
        -------
        centers : ndarray of shape (n_clusters, n_features)
        """
        n_samples = X.shape[0]
        n_clusters = self.n_clusters

        if init_size is not None and init_size < n_samples:
            init_indices = random_state.randint(0, n_samples, init_size)
            X = X[init_indices]
            x_squared_norms = x_squared_norms[init_indices]
            n_samples = X.shape[0]

        if isinstance(init, str) and init == 'k-means++':
            centers = _k_init(X, n_clusters, random_state=random_state,
                              x_squared_norms=x_squared_norms)
        elif isinstance(init, str) and init == 'random':
            seeds = random_state.permutation(n_samples)[:n_clusters]
            centers = X[seeds]
        elif hasattr(init, '__array__'):
            centers = init
        elif callable(init):
            centers = init(X, n_clusters, random_state=random_state)
            centers = check_array(
                centers, dtype=X.dtype, copy=False, order='C')
            self._validate_center_shape(X, centers)

        if sp.issparse(centers):
            centers = centers.toarray()

        return centers

    def fit(self, X, y=None, sample_weight=None):
        """Compute k-means clustering.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
            Fitted estimator.
        """
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', copy=self.copy_x,
                                accept_large_sparse=False)

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, '__array__'):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if self._algorithm == "full":
            kmeans_single = _kmeans_single_lloyd
        else:
            kmeans_single = _kmeans_single_elkan

        best_inertia = None

        for i in range(self._n_init):
            # Initialize centers
            centers_init = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=random_state)
            if self.verbose:
                print("Initialization complete")

            # run a k-means once
            labels, inertia, centers, n_iter_ = kmeans_single(
                X, sample_weight, centers_init, max_iter=self.max_iter,
                verbose=self.verbose, tol=self._tol,
                x_squared_norms=x_squared_norms, n_threads=self._n_threads)

            # determine if these results are the best so far
            if best_inertia is None or inertia < best_inertia:
                best_labels = labels
                best_centers = centers
                best_inertia = inertia
                best_n_iter = n_iter_

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning, stacklevel=2)

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """Compute cluster centers and predict cluster index for each sample.

        Convenience method; equivalent to calling fit(X) followed by
        predict(X).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        return self.fit(X, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        """Compute clustering and transform X to cluster-distance space.

        Equivalent to fit(X).transform(X), but more efficiently implemented.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        X_new : array of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        return self.fit(X, sample_weight=sample_weight)._transform(X)

    def transform(self, X):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster
        centers.  Note that even if X is sparse, the array returned by
        `transform` will typically be dense.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_clusters)
            X transformed in the new space.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        return self._transform(X)

    def _transform(self, X):
        """guts of transform method; no input validation"""
        return euclidean_distances(X, self.cluster_centers_)

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return _labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_,
            self._n_threads)[0]

    def score(self, X, y=None, sample_weight=None):
        """Opposite of the value of X on the K-means objective.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        score : float
            Opposite of the value of X on the K-means objective.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        return -_labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_,
            self._n_threads)[1]

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance(kind=zeros)':
                'zero sample_weight is not equivalent to removing samples',
            }
        }


def _mini_batch_step(X, x_squared_norms, sample_weight, centers, centers_new,
                     weight_sums, random_state, random_reassign=False,
                     reassignment_ratio=0.01, verbose=False, n_threads=1):
    """Incremental update of the centers for the Minibatch K-Means algorithm.

    Parameters
    ----------

    X : {ndarray, sparse matrix} of shape (n_samples, n_features)
        The original data array. In sparse, must be in CSR format.

    x_squared_norms : ndarray of shape (n_samples,)
        Squared euclidean norm of each data point.

    sample_weight : ndarray of shape (n_samples,)
        The weights for each observation in X.

    centers : ndarray of shape (n_clusters, n_features)
        The cluster centers before the current iteration

    centers_new : ndarray of shape (n_clusters, n_features)
        The cluster centers after the current iteration. Modified in-place.

    weight_sums : ndarray of shape (n_clusters,)
        The vector in which we keep track of the numbers of points in a
        cluster. This array is modified in place.

    random_state : RandomState instance
        Determines random number generation for low count centers reassignment.
        See :term:`Glossary <random_state>`.

    random_reassign : boolean, default=False
        If True, centers with very low counts are randomly reassigned
        to observations.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more likely to be reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    verbose : bool, default=False
        Controls the verbosity.

    n_threads : int, default=1
        The number of OpenMP threads to use for the computation.

    Returns
    -------
    inertia : float
        Sum of squared distances of samples to their closest cluster center.
    """
    # Perform label assignment to nearest centers
    labels, inertia = _labels_inertia(X, sample_weight,
                                      x_squared_norms, centers,
                                      n_threads=n_threads)

    # Update centers according to the labels
    if sp.issparse(X):
        _minibatch_update_sparse(X, sample_weight, centers, centers_new,
                                 weight_sums, labels, n_threads)
    else:
        _minibatch_update_dense(X, sample_weight, centers, centers_new,
                                weight_sums, labels, n_threads)

    # Reassign clusters that have very low weight
    if random_reassign and reassignment_ratio > 0:
        to_reassign = weight_sums < reassignment_ratio * weight_sums.max()

        # pick at most .5 * batch_size samples as new centers
        if to_reassign.sum() > .5 * X.shape[0]:
            indices_dont_reassign = \
                    np.argsort(weight_sums)[int(.5 * X.shape[0]):]
            to_reassign[indices_dont_reassign] = False
        n_reassigns = to_reassign.sum()

        if n_reassigns:
            # Pick new clusters amongst observations with uniform probability
            new_centers = random_state.choice(X.shape[0], replace=False,
                                              size=n_reassigns)
            if verbose:
                print(f"[MiniBatchKMeans] Reassigning {n_reassigns} "
                      f"cluster centers.")

            if sp.issparse(X):
                assign_rows_csr(
                        X, new_centers.astype(np.intp, copy=False),
                        np.where(to_reassign)[0].astype(np.intp, copy=False),
                        centers_new)
            else:
                centers_new[to_reassign] = X[new_centers]

        # reset counts of reassigned centers, but don't reset them too small
        # to avoid instant reassignment. This is a pretty dirty hack as it
        # also modifies the learning rates.
        weight_sums[to_reassign] = np.min(weight_sums[~to_reassign])

    return inertia


class MiniBatchKMeans(KMeans):
    """
    Mini-Batch K-Means clustering.

    Read more in the :ref:`User Guide <mini_batch_kmeans>`.

    Parameters
    ----------

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    init : {'k-means++', 'random'}, callable or array-like of shape \
            (n_clusters, n_features), default='k-means++'
        Method for initialization:

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose `n_clusters` observations (rows) at random from data
        for the initial centroids.

        If an array is passed, it should be of shape (n_clusters, n_features)
        and gives the initial centers.

        If a callable is passed, it should take arguments X, n_clusters and a
        random state and return an initialization.

    max_iter : int, default=100
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.

    batch_size : int, default=100
        Size of the mini batches.

    verbose : int, default=0
        Verbosity mode.

    compute_labels : bool, default=True
        Compute label assignment and inertia for the complete dataset
        once the minibatch optimization has converged in fit.

    random_state : int, RandomState instance, default=None
        Determines random number generation for centroid initialization and
        random reassignment. Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    tol : float, default=0.0
        Control early stopping based on the relative center changes as
        measured by a smoothed, variance-normalized of the mean center
        squared position changes. This early stopping heuristics is
        closer to the one used for the batch variant of the algorithms
        but induces a slight computational and memory overhead over the
        inertia heuristic.

        To disable convergence detection based on normalized center
        change, set tol to 0.0 (default).

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini
        batches that does not yield an improvement on the smoothed inertia.

        To disable convergence detection based on inertia, set
        max_no_improvement to None.

    init_size : int, default=None
        Number of samples to randomly sample for speeding up the
        initialization (sometimes at the expense of accuracy): the
        only algorithm is initialized by running a batch KMeans on a
        random subset of the data. This needs to be larger than n_clusters.

        If `None`, the heuristic is `init_size = 3 * batch_size` if
        `3 * batch_size < n_clusters`, else `init_size = 3 * n_clusters`.

    n_init : int, default=3
        Number of random initializations that are tried.
        In contrast to KMeans, the algorithm is only run once, using the
        best of the ``n_init`` initializations as measured by inertia.

    reassignment_ratio : float, default=0.01
        Control the fraction of the maximum number of counts for a
        center to be reassigned. A higher value means that low count
        centers are more easily reassigned, which means that the
        model will take longer to converge, but should converge in a
        better clustering.

    Attributes
    ----------

    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Coordinates of cluster centers

    labels_ : ndarray of shape (n_samples)
        Labels of each point (if compute_labels is set to True).

    inertia_ : float
        The value of the inertia criterion associated with the chosen
        partition (if compute_labels is set to True). The inertia is
        defined as the sum of square distances of samples to their cluster
        center, weighted by the sample weights if provided.

    n_iter_ : int
        Number of batches processed.

    counts_ : ndarray of shape (n_clusters,)
        Weigth sum of each cluster.

        .. deprecated:: 0.24
           This attribute is deprecated in 0.24 and will be removed in 0.26.

    init_size_ : int
        The effective number of samples used for the initialization.

        .. deprecated:: 0.24
           This attribute is deprecated in 0.24 and will be removed in 0.26.

    See Also
    --------
    KMeans
        The classic implementation of the clustering method based on the
        Lloyd's algorithm. It consumes the whole set of input data at each
        iteration.

    Notes
    -----
    See https://www.eecs.tufts.edu/~dsculley/papers/fastkmeans.pdf

    Examples
    --------
    >>> from sklearn.cluster import MiniBatchKMeans
    >>> import numpy as np
    >>> X = np.array([[1, 2], [1, 4], [1, 0],
    ...               [4, 2], [4, 0], [4, 4],
    ...               [4, 5], [0, 1], [2, 2],
    ...               [3, 2], [5, 5], [1, -1]])
    >>> # manually fit on batches
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6)
    >>> kmeans = kmeans.partial_fit(X[0:6,:])
    >>> kmeans = kmeans.partial_fit(X[6:12,:])
    >>> kmeans.cluster_centers_
    array([[2. , 1. ],
           [3.5, 4.5]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    >>> # fit on the whole data
    >>> kmeans = MiniBatchKMeans(n_clusters=2,
    ...                          random_state=0,
    ...                          batch_size=6,
    ...                          max_iter=10).fit(X)
    >>> kmeans.cluster_centers_
    array([[2.32394366, 1.16901408],
           [3.4       , 4.36      ]])
    >>> kmeans.predict([[0, 0], [4, 4]])
    array([0, 1], dtype=int32)
    """
    @_deprecate_positional_args
    def __init__(self, n_clusters=8, *, init='k-means++', max_iter=100,
                 batch_size=100, verbose=0, compute_labels=True,
                 random_state=None, tol=0.0, max_no_improvement=10,
                 init_size=None, n_init=3, reassignment_ratio=0.01, mode=0):

        super().__init__(
            n_clusters=n_clusters, init=init, max_iter=max_iter,
            verbose=verbose, random_state=random_state, tol=tol, n_init=n_init)

        self.max_no_improvement = max_no_improvement
        self.batch_size = batch_size
        self.compute_labels = compute_labels
        self.init_size = init_size
        self.reassignment_ratio = reassignment_ratio
        self.mode = mode

    @deprecated("The attribute 'counts_' is deprecated in 0.24"  # type: ignore
                " and will be removed in 0.26.")
    @property
    def counts_(self):
        return self._counts

    @deprecated("The attribute 'init_size_' is deprecated in "  # type: ignore
                "0.24 and will be removed in 0.26.")
    @property
    def init_size_(self):
        return self._init_size

    @deprecated("The attribute 'random_state_' is deprecated "  # type: ignore
                "in 0.24 and will be removed in 0.26.")
    @property
    def random_state_(self):
        return getattr(self, "_random_state", None)

    def _check_params(self, X):
        super()._check_params(X)

        # max_no_improvement
        if self.max_no_improvement is not None and self.max_no_improvement < 0:
            raise ValueError(
                f"max_no_improvement should be >= 0, got "
                f"{self.max_no_improvement} instead.")

        # batch_size
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be > 0, got {self.batch_size} instead.")

        # init_size
        if self.init_size is not None and self.init_size <= 0:
            raise ValueError(
                f"init_size should be > 0, got {self.init_size} instead.")
        self._init_size = self.init_size
        if self._init_size is None:
            self._init_size = 3 * self.batch_size
            if self._init_size < self.n_clusters:
                self._init_size = 3 * self.n_clusters
        elif self._init_size < self.n_clusters:
            warnings.warn(
                f"init_size={self._init_size} should be larger than "
                f"n_clusters={self.n_clusters}. Setting it to "
                f"min(3*n_clusters, n_samples)",
                RuntimeWarning, stacklevel=2)
            self._init_size = 3 * self.n_clusters
        self._init_size = min(self._init_size, X.shape[0])

        # reassignment_ratio
        if self.reassignment_ratio < 0:
            raise ValueError(
                f"reassignment_ratio should be >= 0, got "
                f"{self.reassignment_ratio} instead.")

    def _mini_batch_convergence(self, iteration_idx, n_iter, n_samples,
                                centers_squared_diff, batch_inertia):
        """Helper function to encapsulate the early stopping logic"""
        # Normalize inertia to be able to compare values when
        # batch_size changes
        batch_inertia /= self.batch_size

        # Ignore first iteration because it's inertia from initialization.
        if iteration_idx == 0:
            if self.verbose:
                print(f"Minibatch iteration {iteration_idx + 1}/{n_iter}: "
                      f"mean batch inertia: {batch_inertia}")
            return False

        # Compute an Exponentially Weighted Average of the inertia to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        ewa_inertia = self._ewa_inertia
        if ewa_inertia is None:
            ewa_inertia = batch_inertia
        else:
            alpha = self.batch_size * 2.0 / (n_samples + 1)
            alpha = min(alpha, 1)
            ewa_inertia = ewa_inertia * (1 - alpha) + batch_inertia * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(f"Minibatch iteration {iteration_idx + 1}/{n_iter}: "
                  f"mean batch inertia: {batch_inertia}, ewa inertia: "
                  f"{ewa_inertia}")

        # Early stopping based on absolute tolerance on squared change of
        # centers position (using EWA smoothing)
        if self._tol > 0.0 and centers_squared_diff <= self._tol:
            if self.verbose:
                print(f"Converged (small centers change) at iteration "
                      f"{iteration_idx + 1}/{n_iter}")
            return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # inertia
        ewa_inertia_min = self._ewa_inertia_min
        no_improvement = self._no_improvement
        if ewa_inertia_min is None or ewa_inertia < ewa_inertia_min:
            no_improvement = 0
            ewa_inertia_min = ewa_inertia
        else:
            no_improvement += 1

        if (self.max_no_improvement is not None
                and no_improvement >= self.max_no_improvement):
            if self.verbose:
                print(f"Converged (lack of improvement in inertia) at "
                      f"iteration {iteration_idx}/{n_iter}")
            return True

        # update the convergence context to maintain state across successive
        # calls:
        self._ewa_inertia = ewa_inertia
        self._ewa_inertia_min = ewa_inertia_min
        self._no_improvement = no_improvement
        return False

    def _random_reassign(self):
        """Check if a random reassignment needs to be done.

        Do random reassignments each time 10 * n_clusters samples have been
        processed.
        """
        self._n_since_last_reassign += self.batch_size
        if self._n_since_last_reassign >= (10 * self.n_clusters):
            self._n_since_last_reassign = 0
            return True
        return False

    def fit(self, X, y=None, sample_weight=None):
        """Compute the centroids on X by chunking it into mini-batches.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory copy
            if the given data is not C-contiguous.
            If a sparse matrix is passed, a copy will be made if it's not in
            CSR format.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

            .. versionadded:: 0.20

        Returns
        -------
        self
        """
        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', accept_large_sparse=False)

        self._check_params(X)
        random_state = check_random_state(self.random_state)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)
        n_samples, n_features = X.shape

        # Validate init array
        init = self.init
        if hasattr(init, '__array__'):
            init = check_array(init, dtype=X.dtype, copy=True, order='C')
            self._validate_center_shape(X, init)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        validation_indices = random_state.randint(0, n_samples,
                                                  self._init_size)
        X_valid = X[validation_indices]
        sample_weight_valid = sample_weight[validation_indices]
        x_squared_norms_valid = x_squared_norms[validation_indices]

        # perform several inits with random sub-sets
        best_inertia = None
        for init_idx in range(self._n_init):
            if self.verbose:
                print(f"Init {init_idx + 1}/{self._n_init} with method {init}")

            # Initialize the centers using only a fraction of the data as we
            # expect n_samples to be very large when using MiniBatchKMeans.
            cluster_centers = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=random_state, init_size=self._init_size)

            # Compute inertia on a validation set.
            _, inertia = _labels_inertia_threadpool_limit(
                X_valid, sample_weight_valid, x_squared_norms_valid,
                cluster_centers, n_threads=self._n_threads)

            if self.verbose:
                print(f"Inertia for init {init_idx + 1}/{self._n_init}: "
                      f"{inertia}")
            if best_inertia is None or inertia < best_inertia:
                init_centers = cluster_centers
                best_inertia = inertia

        centers = init_centers
        centers_new = np.empty_like(centers)

        # Initialize counts
        self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

        # Attributes to monitor the convergence
        self._ewa_inertia = None
        self._ewa_inertia_min = None
        self._no_improvement = 0

        n_batches = int(np.ceil(float(n_samples) / self.batch_size))
        n_iter = int(self.max_iter * n_batches)

        self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            # Perform the iterative optimization until convergence
            for i in range(n_iter):
                # Sample a minibatch from the full dataset
                minibatch_indices = random_state.randint(0, n_samples,
                                                         self.batch_size)

                # Perform the actual update step on the minibatch data
                batch_inertia = _mini_batch_step(
                    X=X[minibatch_indices],
                    x_squared_norms=x_squared_norms[minibatch_indices],
                    sample_weight=sample_weight[minibatch_indices],
                    centers=centers,
                    centers_new=centers_new,
                    weight_sums=self._counts,
                    random_state=random_state,
                    random_reassign=self._random_reassign(),
                    reassignment_ratio=self.reassignment_ratio,
                    verbose=self.verbose,
                    n_threads=self._n_threads)

                if self._tol > 0.0:
                    centers_squared_diff = np.sum((centers_new - centers)**2)
                else:
                    centers_squared_diff = 0

                centers, centers_new = centers_new, centers

                # Monitor convergence and do early stopping if necessary
                if self._mini_batch_convergence(
                        i, n_iter, n_samples, centers_squared_diff,
                        batch_inertia):
                    break

        self.cluster_centers_ = centers

        self.n_iter_ = i + 1

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X, sample_weight, x_squared_norms, self.cluster_centers_,
                n_threads=self._n_threads)

        return self

    def partial_fit(self, X, y=None, sample_weight=None):
        """Update k means estimate on a single mini-batch X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Coordinates of the data points to cluster. It must be noted that
            X will be copied if it is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        self
        """
        is_first_call_to_partial_fit = not hasattr(self, 'cluster_centers_')

        X = self._validate_data(X, accept_sparse='csr',
                                dtype=[np.float64, np.float32],
                                order='C', accept_large_sparse=False,
                                reset=is_first_call_to_partial_fit)

        self._random_state = getattr(self, "_random_state",
                                     check_random_state(self.random_state))
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        if is_first_call_to_partial_fit:
            # this is the first call to partial_fit on this object
            self._check_params(X)

            # Validate init array
            init = self.init
            if hasattr(init, '__array__'):
                init = check_array(init, dtype=X.dtype, copy=True, order='C')
                self._validate_center_shape(X, init)

            # initialize the cluster centers
            self.cluster_centers_ = self._init_centroids(
                X, x_squared_norms=x_squared_norms, init=init,
                random_state=self._random_state, init_size=self._init_size)

            # Initialize counts
            self._counts = np.zeros(self.n_clusters, dtype=X.dtype)

            self._n_since_last_reassign = 0

        with threadpool_limits(limits=1, user_api="blas"):
            _mini_batch_step(X,
                             x_squared_norms=x_squared_norms,
                             sample_weight=sample_weight,
                             centers=self.cluster_centers_,
                             centers_new=self.cluster_centers_,
                             weight_sums=self._counts,
                             random_state=self._random_state,
                             random_reassign=self._random_reassign(),
                             reassignment_ratio=self.reassignment_ratio,
                             verbose=self.verbose,
                             n_threads=self._n_threads)

        if self.compute_labels:
            self.labels_, self.inertia_ = _labels_inertia_threadpool_limit(
                X, sample_weight, x_squared_norms, self.cluster_centers_,
                n_threads=self._n_threads)

        return self

    def predict(self, X, sample_weight=None):
        """Predict the closest cluster each sample in X belongs to.

        In the vector quantization literature, `cluster_centers_` is called
        the code book and each value returned by `predict` is the index of
        the closest code in the code book.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        sample_weight : array-like of shape (n_samples,), default=None
            The weights for each observation in X. If None, all observations
            are assigned equal weight.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Index of the cluster each sample belongs to.
        """
        check_is_fitted(self)

        X = self._check_test_data(X)
        x_squared_norms = row_norms(X, squared=True)
        sample_weight = _check_sample_weight(sample_weight, X, dtype=X.dtype)

        labels, _ = _labels_inertia_threadpool_limit(
            X, sample_weight, x_squared_norms, self.cluster_centers_,
            n_threads=self._n_threads)

        return labels

    def _more_tags(self):
        return {
            '_xfail_checks': {
                'check_sample_weights_invariance(kind=zeros)':
                'zero sample_weight is not equivalent to removing samples',
            }
        }
