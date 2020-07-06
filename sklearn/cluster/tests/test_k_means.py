"""Testing for K-means"""
import sys

import numpy as np
from scipy import sparse as sp
from threadpoolctl import threadpool_limits

import pytest

from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning

from sklearn.utils.extmath import row_norms
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.cluster import v_measure_score
from sklearn.cluster import KMeans, k_means
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster._kmeans import _mini_batch_step
from sklearn.cluster._kmeans import _labels_inertia
from sklearn.cluster._k_means_common import _relocate_empty_clusters_dense
from sklearn.cluster._k_means_common import _relocate_empty_clusters_sparse
from sklearn.cluster._k_means_common import _euclidean_dense_dense_wrapper
from sklearn.cluster._k_means_common import _euclidean_sparse_dense_wrapper
from sklearn.cluster._k_means_common import _inertia_dense
from sklearn.cluster._k_means_common import _inertia_sparse
from sklearn.datasets import make_blobs
from io import StringIO


# non centered, sparse centers to check the
centers = np.array([
    [0.0, 5.0, 0.0, 0.0, 0.0],
    [1.0, 1.0, 4.0, 0.0, 0.0],
    [1.0, 0.0, 0.0, 5.0, 1.0],
])
n_samples = 100
n_clusters, n_features = centers.shape
X, true_labels = make_blobs(n_samples=n_samples, centers=centers,
                            cluster_std=1., random_state=42)
X_csr = sp.csr_matrix(X)


def _check_fitted_model(km):
    # check that the number of clusters centers and distinct labels match
    # the expectation
    centers = km.cluster_centers_
    assert centers.shape == (n_clusters, n_features)

    labels = km.labels_
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert km.inertia_ > 0.0


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("algo", ["full", "elkan"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kmeans_results(array_constr, algo, dtype):
    # Checks that KMeans works as intended on toy dataset by comparing with
    # expected results computed by hand.
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]], dtype=dtype)
    sample_weight = [3, 1, 1, 3]
    init_centers = np.array([[0, 0], [1, 1]], dtype=dtype)

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.375
    expected_centers = np.array([[0.125, 0], [0.875, 1]], dtype=dtype)
    expected_n_iter = 2

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X, sample_weight=sample_weight)

    assert_array_equal(kmeans.labels_, expected_labels)
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert_allclose(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("algo", ["full", "elkan"])
def test_kmeans_relocated_clusters(array_constr, algo):
    # check that empty clusters are relocated as expected
    X = array_constr([[0, 0], [0.5, 0], [0.5, 1], [1, 1]])

    # second center too far from others points will be empty at first iter
    init_centers = np.array([[0.5, 0.5], [3, 3]])

    expected_labels = [0, 0, 1, 1]
    expected_inertia = 0.25
    expected_centers = [[0.25, 0], [0.75, 1]]
    expected_n_iter = 3

    kmeans = KMeans(n_clusters=2, n_init=1, init=init_centers, algorithm=algo)
    kmeans.fit(X)

    assert_array_equal(kmeans.labels_, expected_labels)
    assert_allclose(kmeans.inertia_, expected_inertia)
    assert_allclose(kmeans.cluster_centers_, expected_centers)
    assert kmeans.n_iter_ == expected_n_iter


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
def test_relocate_empty_clusters(array_constr):
    # test for the _relocate_empty_clusters_(dense/sparse) helpers

    # Synthetic dataset with 3 obvious clusters of different sizes
    X = np.array(
        [-10., -9.5, -9, -8.5, -8, -1, 1, 9, 9.5, 10]).reshape(-1, 1)
    X = array_constr(X)
    sample_weight = np.ones(10)

    # centers all initialized to the first point of X
    centers_old = np.array([-10., -10, -10]).reshape(-1, 1)

    # With this initialization, all points will be assigned to the first center
    # At this point a center in centers_new is the weighted sum of the points
    # it contains if it's not empty, otherwise it is the same as before.
    centers_new = np.array([-16.5, -10, -10]).reshape(-1, 1)
    weight_in_clusters = np.array([10., 0, 0])
    labels = np.zeros(10, dtype=np.int32)

    if array_constr is np.array:
        _relocate_empty_clusters_dense(X, sample_weight, centers_old,
                                       centers_new, weight_in_clusters, labels)
    else:
        _relocate_empty_clusters_sparse(X.data, X.indices, X.indptr,
                                        sample_weight, centers_old,
                                        centers_new, weight_in_clusters,
                                        labels)

    # The relocation scheme will take the 2 points farthest from the center and
    # assign them to the 2 empty clusters, i.e. points at 10 and at 9.9. The
    # first center will be updated to contain the other 8 points.
    assert_array_equal(weight_in_clusters, [8, 1, 1])
    assert_allclose(centers_new, [[-36], [10], [9.5]])


@pytest.mark.parametrize("data", [X, X_csr], ids=["dense", "sparse"])
@pytest.mark.parametrize("init", ["random", "k-means++", centers,
                                  lambda X, k, random_state: centers],
                         ids=["random", "k-means++", "ndarray", "callable"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_all_init(estimator, data, init):
    # Check KMeans and MiniBatchKMeans with all possible init.
    n_init = 10 if type(init) is str else 1
    km = estimator(init=init, n_clusters=n_clusters, random_state=42,
                   n_init=n_init).fit(data)
    _check_fitted_model(km)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_result_equal_in_diff_n_threads(estimator):
    # Check that KMeans/MiniBatchKMeans give the same results in parallel mode
    # than in sequential mode.
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(50, 10))

    with threadpool_limits(limits=1, user_api="openmp"):
        result_1 = estimator(
            n_clusters=n_clusters, random_state=0).fit(X).labels_
    with threadpool_limits(limits=2, user_api="openmp"):
        result_2 = estimator(
            n_clusters=n_clusters, random_state=0).fit(X).labels_
    assert_array_equal(result_1, result_2)


def _sort_centers(centers):
    return np.sort(centers, axis=0)


def test_weighted_vs_repeated():
    # Check that a sample weight of N should yield the same result as an N-fold
    # repetition of the sample. Valid only if init is precomputed, otherwise
    # rng produces different results. Not valid for MinibatchKMeans due to rng
    # to extract minibatches.
    sample_weight = np.random.RandomState(0).randint(1, 5, size=n_samples)
    X_repeat = np.repeat(X, sample_weight, axis=0)

    km = KMeans(init=centers, n_init=1, n_clusters=n_clusters, random_state=0)

    km_weighted = clone(km).fit(X, sample_weight=sample_weight)
    repeated_labels = np.repeat(km_weighted.labels_, sample_weight)
    km_repeated = clone(km).fit(X_repeat)

    assert_array_equal(km_repeated.labels_, repeated_labels)
    assert_allclose(km_weighted.inertia_, km_repeated.inertia_)
    assert_allclose(_sort_centers(km_weighted.cluster_centers_),
                    _sort_centers(km_repeated.cluster_centers_))


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_unit_weights_vs_no_weights(estimator):
    # Check that not passing sample weights should be equivalent to passing
    # sample weights all equal to one.
    sample_weight = np.ones(n_samples)

    km = estimator(n_clusters=n_clusters, random_state=42, n_init=1)
    km_none = clone(km).fit(X, sample_weight=None)
    km_ones = clone(km).fit(X, sample_weight=sample_weight)

    assert_array_equal(km_none.labels_, km_ones.labels_)
    assert_allclose(km_none.cluster_centers_, km_ones.cluster_centers_)


@pytest.mark.parametrize("data", [X, X_csr], ids=["dense", "sparse"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_scaled_weights(estimator, data):
    # Check that scaling all sample weights by a common factor
    # shouldn't change the result
    sample_weight = np.random.uniform(n_samples)

    km = estimator(n_clusters=n_clusters, random_state=42, n_init=1)
    km_orig = clone(km).fit(data, sample_weight=sample_weight)
    km_scaled = clone(km).fit(data, sample_weight=0.5 * sample_weight)

    assert_array_equal(km_orig.labels_, km_scaled.labels_)
    assert_allclose(km_orig.cluster_centers_, km_scaled.cluster_centers_)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_dense_sparse(estimator):
    # Check that the results are the same for dense and sparse input.
    sample_weight = np.random.RandomState(0).random_sample((n_samples,))
    km_dense = estimator(n_clusters=n_clusters, random_state=0, n_init=1)
    km_dense.fit(X, sample_weight=sample_weight)
    km_sparse = estimator(n_clusters=n_clusters, random_state=0, n_init=1)
    km_sparse.fit(X_csr, sample_weight=sample_weight)

    assert_array_equal(km_dense.labels_, km_sparse.labels_)
    assert_allclose(km_dense.cluster_centers_, km_sparse.cluster_centers_)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_fortran_aligned_data(estimator):
    # Check that KMeans works with fortran-aligned data.
    X_fortran = np.asfortranarray(X)
    centers_fortran = np.asfortranarray(centers)

    km_c = estimator(n_clusters=n_clusters, init=centers, n_init=1,
                     random_state=42).fit(X)
    km_f = estimator(n_clusters=n_clusters, init=centers_fortran, n_init=1,
                     random_state=42).fit(X_fortran)
    assert_allclose(km_c.cluster_centers_, km_f.cluster_centers_)
    assert_array_equal(km_c.labels_, km_f.labels_)


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_centers_not_mutated(estimator, dtype):
    # Check that KMeans and MiniBatchKMeans won't mutate the user provided
    # init centers silently even if input data and init centers have the same
    # type.
    X_new_type = X.astype(dtype, copy=True)
    centers_new_type = centers.astype(dtype, copy=True)

    km = estimator(init=centers_new_type, n_clusters=n_clusters, n_init=1)
    km.fit(X_new_type)

    assert not np.may_share_memory(km.cluster_centers_, centers)


@pytest.mark.parametrize("data", [X, X_csr], ids=["dense", "sparse"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_float_precision(estimator, data):
    # Check that the results are the same for single and double precision.
    km = estimator(n_init=1, random_state=0)

    inertia = {}
    Xt = {}
    centers = {}
    labels = {}

    for dtype in [np.float64, np.float32]:
        X = data.astype(dtype)
        km.fit(X)

        inertia[dtype] = km.inertia_
        Xt[dtype] = km.transform(X)
        centers[dtype] = km.cluster_centers_
        labels[dtype] = km.labels_

        # dtype of cluster centers has to be the dtype of the input data
        assert km.cluster_centers_.dtype == dtype

        # same with partial_fit
        if estimator is MiniBatchKMeans:
            km.partial_fit(X[0:3])
            assert km.cluster_centers_.dtype == dtype

    # compare arrays with low precision since the difference between
    # 32 and 64 bit sometimes makes a difference up to the 4th decimal
    # place
    assert_allclose(inertia[np.float32], inertia[np.float64], rtol=1e-5)
    assert_allclose(Xt[np.float32], Xt[np.float64], rtol=1e-5)
    assert_allclose(centers[np.float32], centers[np.float64], rtol=1e-5)
    assert_array_equal(labels[np.float32], labels[np.float64])


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_score_max_iter(estimator):
    # Check that fitting KMeans or MiniBatchKMeans with more iterations gives
    # better score
    X = np.random.RandomState(0).randn(100, 10)

    km1 = estimator(n_init=1, random_state=42, max_iter=1)
    s1 = km1.fit(X).score(X)
    km2 = estimator(n_init=1, random_state=42, max_iter=10)
    s2 = km2.fit(X).score(X)
    assert s2 > s1


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.int32, np.int64])
@pytest.mark.parametrize("init", ["k-means++", "ndarray"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_integer_input(estimator, array_constr, dtype, init):
    # Check that KMeans and MiniBatchKMeans work with integer input.
    X_dense = np.array([[0, 0], [10, 10], [12, 9], [-1, 1], [2, 0], [8, 10]])
    X = array_constr(X_dense, dtype=dtype)

    n_init = 1 if init == "ndarray" else 10
    init = X_dense[:2] if init == "ndarray" else init

    km = estimator(n_clusters=2, init=init, n_init=n_init, random_state=0)
    if estimator is MiniBatchKMeans:
        km.set_params(batch_size=2)

    km.fit(X)

    # Internally integer input should be converted to float64
    assert km.cluster_centers_.dtype == np.float64

    expected_labels = [0, 1, 1, 0, 0, 1]
    assert_allclose(v_measure_score(km.labels_, expected_labels), 1)

    # Same with partial_fit (#14314)
    if estimator is MiniBatchKMeans:
        km = clone(km).partial_fit(X)
        assert km.cluster_centers_.dtype == np.float64


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("init", ["random", "k-means++", "ndarray"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_predict(estimator, init, dtype, array_constr):
    # Check the predict method and the equivalence between fit.predict and
    # fit_predict.
    if sys.platform == "darwin":
        pytest.xfail(
            "Known failures on MacOS, See "
            "https://github.com/scikit-learn/scikit-learn/issues/12644")

    X, _ = make_blobs(n_samples=500, n_features=10, centers=10, random_state=0)

    n_init = 1 if init == "ndarray" else 10
    init = X[:10] if init == "ndarray" else init
    X = array_constr(X)

    km = estimator(n_clusters=10, init=init, n_init=n_init,
                   random_state=0).fit(X)
    labels = km.labels_

    # Due to randomness in the order in which chunks of data are processed when
    # using more than one thread, there might be different rounding errors for
    # the computation of the inertia for each init between 2 runs. This might
    # result in a different ranking of the inits, hence a different labeling,
    # which should still correspond to the same clustering

    # re-predict labels for training set using predict
    pred = km.predict(X)
    assert_allclose(v_measure_score(pred, labels), 1)

    # re-predict labels for training set using fit_predict
    pred = km.fit_predict(X)
    assert_allclose(v_measure_score(pred, labels), 1)

    # predict centroid labels
    pred = km.predict(km.cluster_centers_)
    assert_allclose(v_measure_score(pred, np.arange(10)), 1)


@pytest.mark.parametrize("init", ["random", "k-means++", centers],
                         ids=["random", "k-means++", "ndarray"])
@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_predict_dense_sparse(estimator, init):
    # check that models trained on sparse input also works for dense input at
    # predict time and vice versa.
    n_init = 10 if type(init) is str else 1
    km = estimator(n_clusters=n_clusters, init=init, n_init=n_init,
                   random_state=0)

    km.fit(X_csr)
    assert_array_equal(km.predict(X), km.labels_)

    km.fit(X)
    assert_array_equal(km.predict(X_csr), km.labels_)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_transform(estimator):
    # Check the transform method
    km = estimator(n_clusters=n_clusters).fit(X)

    # Transorfming cluster_centers_ should return the pairwise distances
    # between centers
    Xt = km.transform(km.cluster_centers_)
    assert_allclose(Xt, pairwise_distances(km.cluster_centers_))
    # In particular, diagonal must be 0
    assert_array_equal(Xt.diagonal(), np.zeros(n_clusters))

    # Transorfming X should return the pairwise distances between X and the
    # centers
    Xt = km.transform(X)
    assert_allclose(Xt, pairwise_distances(X, km.cluster_centers_))


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_fit_transform(estimator):
    # Check equivalence between fit.transform and fit_transform
    X1 = estimator(random_state=0, n_init=1).fit(X).transform(X)
    X2 = estimator(random_state=0, n_init=1).fit_transform(X)
    assert_allclose(X1, X2)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_sample_weight_unchanged(estimator):
    # Check that sample_weight is not modified in place by KMeans (#17204)
    X = np.array([[1], [2], [4]])
    sample_weight = np.array([0.5, 0.2, 0.3])
    estimator(n_clusters=2, random_state=0).fit(X, sample_weight=sample_weight)

    # internally, sample_weight is rescale to sum up to n_samples = 3
    assert_array_equal(sample_weight, np.array([0.5, 0.2, 0.3]))


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_verbose(estimator):
    # Check verbose mode of KMeans and MiniBatchKMeans for better coverage.
    km = estimator(n_clusters=n_clusters, random_state=42, verbose=1)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        km.fit(X)
    finally:
        sys.stdout = old_stdout


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("algo", ["full", "elkan"])
def test_k_means_1_iteration(array_constr, algo):
    # check the results after a single iteration (E-step M-step E-step) by
    # comparing against a pure python implementation.
    X = np.random.RandomState(0).uniform(size=(100, 5))
    init_centers = X[:5]
    X = array_constr(X)

    def py_kmeans(X, init):
        new_centers = init.copy()
        labels = pairwise_distances_argmin(X, init)
        for label in range(init.shape[0]):
            new_centers[label] = X[labels == label].mean(axis=0)
        labels = pairwise_distances_argmin(X, new_centers)
        return labels, new_centers

    py_labels, py_centers = py_kmeans(X, init_centers)

    cy_kmeans = KMeans(n_clusters=5, n_init=1, init=init_centers,
                       algorithm=algo, max_iter=1).fit(X)
    cy_labels = cy_kmeans.labels_
    cy_centers = cy_kmeans.cluster_centers_

    assert_array_equal(py_labels, cy_labels)
    assert_allclose(py_centers, cy_centers)


@pytest.mark.parametrize("distribution", ["normal", "blobs"])
@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
@pytest.mark.parametrize("tol", [1e-2, 1e-4, 1e-8])
def test_kmeans_elkan_results(distribution, array_constr, tol):
    # Check that results are identical between lloyd and elkan algorithms
    rnd = np.random.RandomState(0)
    if distribution == "normal":
        X = rnd.normal(size=(5000, 10))
    else:
        X, _ = make_blobs(random_state=rnd)
    X[X < 0] = 0
    X = array_constr(X)

    km_full = KMeans(algorithm="full", n_clusters=5,
                     random_state=0, n_init=1, tol=tol)
    km_elkan = KMeans(algorithm="elkan", n_clusters=5,
                      random_state=0, n_init=1, tol=tol)

    km_full.fit(X)
    km_elkan.fit(X)
    assert_allclose(km_elkan.cluster_centers_, km_full.cluster_centers_)
    assert_array_equal(km_elkan.labels_, km_full.labels_)
    assert km_elkan.n_iter_ == km_full.n_iter_
    assert km_elkan.inertia_ == pytest.approx(km_full.inertia_, rel=1e-6)


@pytest.mark.parametrize("algorithm", ["full", "elkan"])
def test_kmeans_convergence(algorithm):
    # Check that KMeans stops when convergence is reached when tol=0. (#16075)
    # We can only ensure that if the number of threads is not to large,
    # otherwise the roundings errors coming from the unpredictability of
    # the order in which chunks are processed make the convergence criterion
    # to never be exactly 0.
    rnd = np.random.RandomState(0)
    X = rnd.normal(size=(5000, 10))

    with threadpool_limits(limits=1, user_api="openmp"):
        km = KMeans(algorithm=algorithm, n_clusters=5, random_state=0,
                    n_init=1, tol=0, max_iter=300).fit(X)

    assert km.n_iter_ < 300


def test_kmeans_copyx():
    # Check that copy_x=False returns nearly equal X after de-centering.
    my_X = X.copy()
    km = KMeans(copy_x=False, n_clusters=n_clusters, random_state=42)
    km.fit(my_X)
    _check_fitted_model(km)

    # check that my_X is de-centered
    assert_allclose(my_X, X)


@pytest.mark.parametrize("data", [X, X_csr], ids=["dense", "sparse"])
def test_kmeans_init_fitted_centers(data):
    # Check that starting fitting from a local optimum shouldn't change the
    # solution
    km1 = KMeans(n_clusters=n_clusters).fit(data)
    km2 = KMeans(n_clusters=n_clusters, init=km1.cluster_centers_,
                 n_init=1).fit(data)

    assert_allclose(km1.cluster_centers_, km2.cluster_centers_)


def test_kmeans_elkan_iter_attribute():
    # Regression test on bad n_iter_ value. Previous bug n_iter_ was one off
    # it's right value (#11340).
    km = KMeans(algorithm="elkan", max_iter=1).fit(X)
    assert km.n_iter_ == 1


@pytest.mark.parametrize("array_constr", [np.array, sp.csr_matrix],
                         ids=["dense", "sparse"])
def test_kmeans_empty_cluster_relocated(array_constr):
    # check that empty clusters are correctly relocated when using sample
    # weights (#13486)
    X = array_constr([[-1], [1]])
    sample_weight = [1.9, 0.1]
    init = np.array([[-1], [10]])

    km = KMeans(n_clusters=2, init=init, n_init=1)
    km.fit(X, sample_weight=sample_weight)

    assert len(set(km.labels_)) == 2
    assert_allclose(km.cluster_centers_, [[-1], [1]])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("squared", [True, False])
def test_euclidean_distance(dtype, squared):
    # Check that the _euclidean_(dense/sparse)_dense helpers produce correct
    # results
    rng = np.random.RandomState(0)
    a_sparse = sp.random(1, 100, density=0.5, format="csr", random_state=rng,
                         dtype=dtype)
    a_dense = a_sparse.toarray().reshape(-1)
    b = rng.randn(100).astype(dtype, copy=False)
    b_squared_norm = (b**2).sum()

    expected = ((a_dense - b)**2).sum()
    expected = expected if squared else np.sqrt(expected)

    distance_dense_dense = _euclidean_dense_dense_wrapper(a_dense, b, squared)
    distance_sparse_dense = _euclidean_sparse_dense_wrapper(
        a_sparse.data, a_sparse.indices, b, b_squared_norm, squared)

    assert_allclose(distance_dense_dense, distance_sparse_dense, rtol=1e-6)
    assert_allclose(distance_dense_dense, expected, rtol=1e-6)
    assert_allclose(distance_sparse_dense, expected, rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_inertia(dtype):
    # Check that the _inertia_(dense/sparse) helpers produce correct results.
    rng = np.random.RandomState(0)
    X_sparse = sp.random(100, 10, density=0.5, format="csr", random_state=rng,
                         dtype=dtype)
    X_dense = X_sparse.toarray()
    sample_weight = rng.randn(100).astype(dtype, copy=False)
    centers = rng.randn(5, 10).astype(dtype, copy=False)
    labels = rng.randint(5, size=100, dtype=np.int32)

    distances = ((X_dense - centers[labels])**2).sum(axis=1)
    expected = np.sum(distances * sample_weight)

    inertia_dense = _inertia_dense(
        X_dense, sample_weight, centers, labels, 1)
    inertia_sparse = _inertia_sparse(
        X_sparse, sample_weight, centers, labels, 1)

    assert_allclose(inertia_dense, inertia_sparse, rtol=1e-6)
    assert_allclose(inertia_dense, expected, rtol=1e-6)
    assert_allclose(inertia_sparse, expected, rtol=1e-6)


def test_k_means_function():
    # test calling the k_means function directly
    cluster_centers, labels, inertia = k_means(X, n_clusters=n_clusters,
                                               sample_weight=None)

    assert cluster_centers.shape == (n_clusters, n_features)
    assert np.unique(labels).shape[0] == n_clusters

    # check that the labels assignment are perfect (up to a permutation)
    assert_allclose(v_measure_score(true_labels, labels), 1.0)
    assert inertia > 0.0


def test_minibatch_update_consistency():
    # Check that dense and sparse minibatch update give the same results
    rng = np.random.RandomState(42)

    centers_old = centers + rng.normal(size=centers.shape)
    centers_old_csr = centers_old.copy()

    centers_new = np.zeros_like(centers_old)
    centers_new_csr = np.zeros_like(centers_old_csr)

    weight_sums = np.zeros(centers_old.shape[0], dtype=X.dtype)
    weight_sums_csr = np.zeros(centers_old.shape[0], dtype=X.dtype)

    x_squared_norms = (X ** 2).sum(axis=1)
    x_squared_norms_csr = row_norms(X_csr, squared=True)

    sample_weight = np.ones(X.shape[0], dtype=X.dtype)

    # extract a small minibatch
    X_mb = X[:10]
    X_mb_csr = X_csr[:10]
    x_mb_squared_norms = x_squared_norms[:10]
    x_mb_squared_norms_csr = x_squared_norms_csr[:10]
    sample_weight_mb = sample_weight[:10]

    # step 1: compute the dense minibatch update
    old_inertia = _mini_batch_step(
        X_mb, x_mb_squared_norms, sample_weight_mb, centers_old, centers_new,
        weight_sums, np.random.RandomState(0), random_reassign=False)
    assert old_inertia > 0.0

    # compute the new inertia on the same batch to check that it decreased
    labels, new_inertia = _labels_inertia(
        X_mb, sample_weight_mb, x_mb_squared_norms, centers_new)
    assert new_inertia > 0.0
    assert new_inertia < old_inertia

    # step 2: compute the sparse minibatch update
    old_inertia_csr = _mini_batch_step(
        X_mb_csr, x_mb_squared_norms_csr, sample_weight_mb, centers_old_csr,
        centers_new_csr, weight_sums_csr, np.random.RandomState(0),
        random_reassign=False)
    assert old_inertia_csr > 0.0

    # compute the new inertia on the same batch to check that it decreased
    labels_csr, new_inertia_csr = _labels_inertia(
        X_mb_csr, sample_weight_mb, x_mb_squared_norms_csr, centers_new_csr)
    assert new_inertia_csr > 0.0
    assert new_inertia_csr < old_inertia_csr

    # step 3: check that sparse and dense updates lead to the same results
    assert_array_equal(labels, labels_csr)
    assert_allclose(centers_new, centers_new_csr)
    assert_allclose(old_inertia, old_inertia_csr)
    assert_allclose(new_inertia, new_inertia_csr)


def test_minibatch_kmeans_init_size():
    # Check the internal _init_size attribute of MiniBatchKMeans

    # default init size should be 3 * batch_size
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1).fit(X)
    assert km._init_size == 15

    # if 3 * batch size < n_clusters, it should then be 3 * n_clusters
    km = MiniBatchKMeans(n_clusters=10, batch_size=1, n_init=1).fit(X)
    assert km._init_size == 30

    # it should not be larger than n_samples
    km = MiniBatchKMeans(n_clusters=10, batch_size=5, n_init=1,
                         init_size=n_samples + 1).fit(X)
    assert km._init_size == n_samples


def test_minibatch_sensible_reassign():
    # check that identical initial clusters are reassigned
    # also a regression test for when there are more desired reassignments than
    # samples.
    zeroed_X, true_labels = make_blobs(n_samples=100, centers=5,
                                       random_state=42)
    zeroed_X[::2, :] = 0

    km = MiniBatchKMeans(n_clusters=20, batch_size=10, random_state=42,
                         init="random").fit(zeroed_X)
    # there should not be too many exact zero cluster centers
    assert km.cluster_centers_.any(axis=1).sum() > 10

    # do the same with batch-size > X.shape[0] (regression test)
    km = MiniBatchKMeans(n_clusters=20, batch_size=200, random_state=42,
                         init="random").fit(zeroed_X)
    # there should not be too many exact zero cluster centers
    assert km.cluster_centers_.any(axis=1).sum() > 10

    # do the same with partial_fit API
    km = MiniBatchKMeans(n_clusters=20, random_state=42, init="random")
    for i in range(100):
        km.partial_fit(zeroed_X)
    # there should not be too many exact zero cluster centers
    assert km.cluster_centers_.any(axis=1).sum() > 10


@pytest.mark.parametrize("data", [X, X_csr], ids=["dense", "sparse"])
def test_minibatch_reassign(data):
    # Check the reassignment part of the minibatch step with very high or very
    # low reassignment ratio.
    perfect_centers = np.empty((n_clusters, n_features))
    for i in range(n_clusters):
        perfect_centers[i] = X[true_labels == i].mean(axis=0)

    x_squared_norms = row_norms(data, squared=True)
    sample_weight = np.ones(n_samples)
    centers_new = np.empty_like(perfect_centers)

    # Give a perfect initialization, but a large reassignment_ratio, as a
    # result many centers should be reassigned and the model should no longer
    # be good
    score_before = - _labels_inertia(data, sample_weight, x_squared_norms,
                                     perfect_centers, 1)[1]

    _mini_batch_step(data, x_squared_norms, sample_weight, perfect_centers,
                     centers_new, np.zeros(n_clusters),
                     np.random.RandomState(0), random_reassign=True,
                     reassignment_ratio=1)

    score_after = - _labels_inertia(data, sample_weight, x_squared_norms,
                                    centers_new, 1)[1]

    assert score_before > score_after

    # Give a perfect initialization, with a small reassignment_ratio,
    # no center should be reassigned.
    _mini_batch_step(data, x_squared_norms, sample_weight, perfect_centers,
                     centers_new, np.zeros(n_clusters),
                     np.random.RandomState(0), random_reassign=True,
                     reassignment_ratio=1e-15)

    assert_allclose(centers_new, perfect_centers)


def test_minibatch_with_many_reassignments():
    # Test for the case that the number of clusters to reassign is bigger
    # than the batch_size. Run the test with 100 clusters and a batch_size of
    # 10 because it turned out that these values ensure that the number of
    # clusters to reassign is always bigger than the batch_size.
    MiniBatchKMeans(n_clusters=100,
                    batch_size=10,
                    init_size=n_samples,
                    random_state=42,
                    verbose=True).fit(X)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
@pytest.mark.parametrize("param, match", [
    ({"n_init": 0}, r"n_init should be > 0"),
    ({"max_iter": 0}, r"max_iter should be > 0"),
    ({"n_clusters": n_samples + 1}, r"n_samples.* should be >= n_clusters"),
    ({"tol": -1}, r"tol should be >= 0"),
    ({"init": X[:2]},
     r"The shape of the initial centers .* does not match"
     r" the number of clusters"),
    ({"init": lambda X_, k, random_state: X_[:2]},
     r"The shape of the initial centers .* does not match"
     r" the number of clusters"),
    ({"init": X[:8, :2]},
     r"The shape of the initial centers .* does not match"
     r" the number of features of the data"),
    ({"init": lambda X_, k, random_state: X_[:8, :2]},
     r"The shape of the initial centers .* does not match"
     r" the number of features of the data"),
    ({"init": "wrong"},
     r"init should be either 'k-means\+\+', 'random', "
     r"a ndarray or a callable")]
)
def test_wrong_params(estimator, param, match):
    # Check that error are raised with clear error message when wrong values
    # are passed for the parameters
    with pytest.raises(ValueError, match=match):
        estimator(**param).fit(X)


@pytest.mark.parametrize("param, match", [
    ({"algorithm": "wrong"}, r"Algorithm must be 'auto', 'full' or 'elkan'")]
)
def test_kmeans_wrong_params(param, match):
    # Check that error are raised with clear error message when wrong values
    # are passed for the KMeans specific parameters
    with pytest.raises(ValueError, match=match):
        KMeans(**param).fit(X)


@pytest.mark.parametrize("param, match", [
    ({"max_no_improvement": -1}, r"max_no_improvement should be >= 0"),
    ({"batch_size": -1}, r"batch_size should be > 0"),
    ({"init_size": -1}, r"init_size should be > 0"),
    ({"reassignment_ratio": -1}, r"reassignment_ratio should be >= 0")]
)
def test_minibatch_kmeans_wrong_params(param, match):
    # Check that error are raised with clear error message when wrong values
    # are passed for the MiniBatchKMeans specific parameters
    with pytest.raises(ValueError, match=match):
        MiniBatchKMeans(**param).fit(X)


@pytest.mark.parametrize("estimator", [KMeans, MiniBatchKMeans])
def test_warnings(estimator):
    # Check warning messages common to KMeans and MiniBatchKMeans
    with pytest.warns(RuntimeWarning,
                      match="Explicit initial center position passed: "
                            "performing only one init"):
        estimator(init=centers, n_clusters=n_clusters).fit(X)


def test_kmeans_warnings():
    # Check warning messages specific to KMeans
    with pytest.warns(RuntimeWarning,
                      match="algorithm='elkan' doesn't make sense for a single"
                            " cluster"):
        KMeans(n_clusters=1, algorithm="elkan").fit(X)


def test_kmeans_warns_less_centers_than_unique_points():
    # Check KMeans when the number of found clusters is smaller than expected
    X = np.asarray([[0, 0],
                    [0, 1],
                    [1, 0],
                    [1, 0]])  # last point is duplicated
    km = KMeans(n_clusters=4)

    # KMeans should warn that fewer labels than cluster centers have been used
    msg = (r"Number of distinct clusters \(3\) found smaller than "
           r"n_clusters \(4\). Possibly due to duplicate points in X.")
    with pytest.warns(ConvergenceWarning, match=msg):
        km.fit(X)
        # only three distinct points, so only three clusters
        # can have points assigned to them
        assert set(km.labels_) == set(range(3))


def test_minibatch_kmeans_warnings():
    # Check warning messages specific to MiniBatchKMeans
    with pytest.warns(RuntimeWarning,
                      match=r"init_size.* should be larger than n_clusters"):
        MiniBatchKMeans(init_size=10, n_clusters=20).fit(X)


@pytest.mark.parametrize("precompute_distances", ["auto", False, True])
def test_precompute_distance_deprecated(precompute_distances):
    # FIXME: remove in 0.25
    depr_msg = ("'precompute_distances' was deprecated in version 0.23 and "
                "will be removed in 0.25.")
    X, _ = make_blobs(n_samples=10, n_features=2, centers=2, random_state=0)
    kmeans = KMeans(n_clusters=2, n_init=1, init="random", random_state=0,
                    precompute_distances=precompute_distances)

    with pytest.warns(FutureWarning, match=depr_msg):
        kmeans.fit(X)


@pytest.mark.parametrize("n_jobs", [None, 1])
def test_n_jobs_deprecated(n_jobs):
    # FIXME: remove in 0.25
    depr_msg = ("'n_jobs' was deprecated in version 0.23 and will be removed "
                "in 0.25.")
    X, _ = make_blobs(n_samples=10, n_features=2, centers=2, random_state=0)
    kmeans = KMeans(n_clusters=2, n_init=1, init="random", random_state=0,
                    n_jobs=n_jobs)

    with pytest.warns(FutureWarning, match=depr_msg):
        kmeans.fit(X)
