import numpy as np
import scipy.sparse as sp

from scipy import linalg
from sklearn.decomposition import NMF, MiniBatchNMF
from sklearn.decomposition import non_negative_factorization
from sklearn.decomposition import _nmf as nmf  # For testing internals
from scipy.sparse import csc_matrix

import pytest

from sklearn.utils._testing import assert_raise_message
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_almost_equal
from sklearn.utils._testing import assert_allclose
from sklearn.utils._testing import ignore_warnings
from sklearn.utils.extmath import squared_norm
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning


@pytest.mark.parametrize(['estimator', 'solver'],
                         [[NMF, 'cd'], [NMF, 'mu'],
                          [MiniBatchNMF, 'mu']])
@pytest.mark.parametrize('regularization',
                         [None, 'both', 'components', 'transformation'])
def test_convergence_warning(estimator, solver, regularization):
    convergence_warning = ("Maximum number of iterations 1 reached. "
                           "Increase it to improve convergence.")
    A = np.ones((2, 2))
    with pytest.warns(ConvergenceWarning, match=convergence_warning):
        estimator(
            solver=solver, regularization=regularization, max_iter=1
        ).fit(A)


def test_initialize_nn_output():
    # Test that initialization does not return negative values
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    for init in ('random', 'nndsvd', 'nndsvda', 'nndsvdar'):
        W, H, _, _ = nmf._initialize_nmf(data, 10, init=init, random_state=0)
        assert not ((W < 0).any() or (H < 0).any())


def test_parameter_checking():
    A = np.ones((2, 2))
    name = 'spam'
    msg = "Invalid solver parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, NMF(solver=name).fit, A)
    msg = "Invalid solver parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, MiniBatchNMF(solver=name).fit, A)
    msg = "Invalid init parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, NMF(init=name).fit, A)
    msg = "Invalid regularization parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, NMF(regularization=name).fit, A)
    msg = "Invalid beta_loss parameter: got 'spam' instead of one"
    assert_raise_message(ValueError, msg, NMF(solver='mu',
                                              beta_loss=name).fit, A)
    msg = "Invalid beta_loss parameter: got 'spam' instead of one"
    assert_raise_message(
        ValueError, msg, MiniBatchNMF(solver='mu', beta_loss=name).fit, A
    )
    msg = ("Coordinate descent algorithm is not available for MiniBatchNMF. "
           "Please set solver to 'mu'.")
    assert_raise_message(
        ValueError, msg,
        MiniBatchNMF(solver='cd', beta_loss='frobenius').fit, A
    )
    msg = "Invalid beta_loss parameter: solver 'cd' does not handle "
    msg += "beta_loss = 1.0"
    assert_raise_message(ValueError, msg, NMF(solver='cd',
                                              beta_loss=1.0).fit, A)

    msg = "Negative values in data passed to"
    assert_raise_message(ValueError, msg, NMF().fit, -A)
    assert_raise_message(ValueError, msg, MiniBatchNMF().fit, -A)
    assert_raise_message(ValueError, msg, nmf._initialize_nmf, -A,
                         2, 'nndsvd')
    clf = NMF(2, tol=0.1).fit(A)
    assert_raise_message(ValueError, msg, clf.transform, -A)

    for init in ['nndsvd', 'nndsvda', 'nndsvdar']:
        msg = ("init = '{}' can only be used when "
               "n_components <= min(n_samples, n_features)"
               .format(init))
        assert_raise_message(ValueError, msg, NMF(3, init=init).fit, A)
        assert_raise_message(ValueError, msg,
                             MiniBatchNMF(3, init=init).fit, A)
        assert_raise_message(ValueError, msg, nmf._initialize_nmf, A,
                             3, init)


def test_initialize_close():
    # Test NNDSVD error
    # Test that _initialize_nmf error is less than the standard deviation of
    # the entries in the matrix.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    W, H, _, _ = nmf._initialize_nmf(A, 10, init='nndsvd')
    error = linalg.norm(np.dot(W, H) - A)
    sdev = linalg.norm(A - A.mean())
    assert error <= sdev


def test_initialize_variants():
    # Test NNDSVD variants correctness
    # Test that the variants 'nndsvda' and 'nndsvdar' differ from basic
    # 'nndsvd' only where the basic version has zeros.
    rng = np.random.mtrand.RandomState(42)
    data = np.abs(rng.randn(10, 10))
    W0, H0, _, _ = nmf._initialize_nmf(data, 10, init='nndsvd')
    Wa, Ha, _, _ = nmf._initialize_nmf(data, 10, init='nndsvda')
    War, Har, _, _ = nmf._initialize_nmf(data, 10, init='nndsvdar',
                                         random_state=0)

    for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
        assert_almost_equal(evl[ref != 0], ref[ref != 0])


# ignore UserWarning raised when both solver='mu' and init='nndsvd'
@ignore_warnings(category=UserWarning)
@pytest.mark.parametrize(['estimator', 'solver'],
                         [[NMF, 'cd'], [NMF, 'mu'],
                          [MiniBatchNMF, 'mu']])
@pytest.mark.parametrize('init',
                         (None, 'nndsvd', 'nndsvda', 'nndsvdar', 'random'))
@pytest.mark.parametrize('regularization',
                         (None, 'both', 'components', 'transformation'))
def test_nmf_fit_nn_output(estimator, solver, init, regularization):
    # Test that the decomposition does not contain negative values
    A = np.c_[5. - np.arange(1, 6),
              5. + np.arange(1, 6)]
    model = estimator(n_components=2, solver=solver, init=init,
                      regularization=regularization, random_state=0)
    transf = model.fit_transform(A)
    assert not((model.components_ < 0).any() or
               (transf < 0).any())


@pytest.mark.parametrize(['estimator', 'solver'],
                         [[NMF, 'cd'], [NMF, 'mu'],
                          [MiniBatchNMF, 'mu']])
@pytest.mark.parametrize('regularization',
                         (None, 'both', 'components', 'transformation'))
def test_nmf_fit_close(estimator, solver, regularization):
    rng = np.random.mtrand.RandomState(42)
    # Test that the fit is not too far away
    pnmf = estimator(5, solver=solver, init='nndsvdar', random_state=0,
                     regularization=regularization, max_iter=600)
    X = np.abs(rng.randn(6, 5))
    assert pnmf.fit(X).reconstruction_err_ < 0.1


@pytest.mark.parametrize('solver', ('cd', 'mu'))
@pytest.mark.parametrize('regularization',
                         (None, 'both', 'components', 'transformation'))
def test_nmf_transform(solver, regularization):
    # Test that NMF.transform returns close values
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(6, 5))
    m = NMF(solver=solver, n_components=3, init='random',
            regularization=regularization, random_state=0, tol=1e-5)
    ft = m.fit_transform(A)
    t = m.transform(A)
    assert_array_almost_equal(ft, t, decimal=2)


def test_nmf_transform_custom_init():
    # Smoke test that checks if NMF.transform works with custom initialization
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 5))
    n_components = 4
    avg = np.sqrt(A.mean() / n_components)
    H_init = np.abs(avg * random_state.randn(n_components, 5))
    W_init = np.abs(avg * random_state.randn(6, n_components))

    m = NMF(solver='cd', n_components=n_components, init='custom',
            random_state=0)
    m.fit_transform(A, W=W_init, H=H_init)
    m.transform(A)


@pytest.mark.parametrize('solver', ('cd', 'mu'))
@pytest.mark.parametrize('regularization',
                         (None, 'both', 'components', 'transformation'))
def test_nmf_inverse_transform(solver, regularization):
    # Test that NMF.inverse_transform returns close values
    random_state = np.random.RandomState(0)
    A = np.abs(random_state.randn(6, 4))
    m = NMF(solver=solver, n_components=4, init='random', random_state=0,
            regularization=regularization, max_iter=1000)
    ft = m.fit_transform(A)
    A_new = m.inverse_transform(ft)
    assert_array_almost_equal(A, A_new, decimal=2)


def test_n_components_greater_n_features():
    # Smoke test for the case of more components than features.
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(30, 10))
    NMF(n_components=15, random_state=0, tol=1e-2).fit(A)


@pytest.mark.parametrize('solver', ['cd', 'mu'])
@pytest.mark.parametrize('regularization',
                         [None, 'both', 'components', 'transformation'])
def test_nmf_sparse_input(solver, regularization):
    # Test that sparse matrices are accepted as input
    from scipy.sparse import csc_matrix

    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0
    A_sparse = csc_matrix(A)

    est1 = NMF(solver=solver, n_components=5, init='random',
               regularization=regularization, random_state=0,
               tol=1e-2)
    est2 = clone(est1)

    W1 = est1.fit_transform(A)
    W2 = est2.fit_transform(A_sparse)
    H1 = est1.components_
    H2 = est2.components_

    assert_array_almost_equal(W1, W2)
    assert_array_almost_equal(H1, H2)


def test_nmf_sparse_transform():
    # Test that transform works on sparse data.  Issue #2124
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(3, 2))
    A[1, 1] = 0
    A = csc_matrix(A)

    for solver in ('cd', 'mu'):
        model = NMF(solver=solver, random_state=0, n_components=2,
                    max_iter=400)
        A_fit_tr = model.fit_transform(A)
        A_tr = model.transform(A)
        assert_array_almost_equal(A_fit_tr, A_tr, decimal=1)


@pytest.mark.parametrize('init', ['random', 'nndsvd'])
@pytest.mark.parametrize('solver', ('cd', 'mu'))
@pytest.mark.parametrize('regularization',
                         (None, 'both', 'components', 'transformation'))
def test_non_negative_factorization_consistency(init, solver, regularization):
    # Test that the function is called in the same way, either directly
    # or through the NMF class
    rng = np.random.mtrand.RandomState(42)
    A = np.abs(rng.randn(10, 10))
    A[:, 2 * np.arange(5)] = 0

    W_nmf, H, _ = non_negative_factorization(
        A, init=init, solver=solver,
        regularization=regularization, random_state=1, tol=1e-2)
    W_nmf_2, _, _ = non_negative_factorization(
        A, H=H, update_H=False, init=init, solver=solver,
        regularization=regularization, random_state=1, tol=1e-2)

    model_class = NMF(init=init, solver=solver,
                      regularization=regularization,
                      random_state=1, tol=1e-2)
    W_cls = model_class.fit_transform(A)
    W_cls_2 = model_class.transform(A)

    assert_array_almost_equal(W_nmf, W_cls, decimal=10)
    assert_array_almost_equal(W_nmf_2, W_cls_2, decimal=10)


def test_non_negative_factorization_checking():
    A = np.ones((2, 2))
    # Test parameters checking is public function
    nnmf = non_negative_factorization
    msg = ("Number of components must be a positive integer; "
           "got (n_components=1.5)")
    assert_raise_message(ValueError, msg, nnmf, A, A, A, 1.5, init='random')
    msg = ("Number of components must be a positive integer; "
           "got (n_components='2')")
    assert_raise_message(ValueError, msg, nnmf, A, A, A, '2', init='random')
    msg = "Negative values in data passed to NMF (input H)"
    assert_raise_message(ValueError, msg, nnmf, A, A, -A, 2, init='custom')
    msg = "Negative values in data passed to NMF (input W)"
    assert_raise_message(ValueError, msg, nnmf, A, -A, A, 2, init='custom')
    msg = "Array passed to NMF (input H) is full of zeros"
    assert_raise_message(ValueError, msg, nnmf, A, A, 0 * A, 2, init='custom')
    msg = "Invalid regularization parameter: got 'spam' instead of one of"
    assert_raise_message(ValueError, msg, nnmf, A, A, 0 * A, 2, init='custom',
                         regularization='spam')


def _beta_divergence_dense(X, W, H, beta):
    """Compute the beta-divergence of X and W.H for dense array only.

    Used as a reference for testing nmf._beta_divergence.
    """
    WH = np.dot(W, H)

    if beta == 2:
        return squared_norm(X - WH) / 2

    WH_Xnonzero = WH[X != 0]
    X_nonzero = X[X != 0]
    np.maximum(WH_Xnonzero, 1e-9, out=WH_Xnonzero)

    if beta == 1:
        res = np.sum(X_nonzero * np.log(X_nonzero / WH_Xnonzero))
        res += WH.sum() - X.sum()

    elif beta == 0:
        div = X_nonzero / WH_Xnonzero
        res = np.sum(div) - X.size - np.sum(np.log(div))
    else:
        res = (X_nonzero ** beta).sum()
        res += (beta - 1) * (WH ** beta).sum()
        res -= beta * (X_nonzero * (WH_Xnonzero ** (beta - 1))).sum()
        res /= beta * (beta - 1)

    return res


def test_beta_divergence():
    # Compare _beta_divergence with the reference _beta_divergence_dense
    n_samples = 20
    n_features = 10
    n_components = 5
    beta_losses = [0., 0.5, 1., 1.5, 2.]

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)
    W, H, _, _ = nmf._initialize_nmf(X, n_components, init='random',
                                     random_state=42)

    for beta in beta_losses:
        ref = _beta_divergence_dense(X, W, H, beta)
        loss = nmf._beta_divergence(X, W, H, beta)
        loss_csr = nmf._beta_divergence(X_csr, W, H, beta)

        assert_almost_equal(ref, loss, decimal=7)
        assert_almost_equal(ref, loss_csr, decimal=7)


def test_special_sparse_dot():
    # Test the function that computes np.dot(W, H), only where X is non zero.
    n_samples = 10
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)

    W = np.abs(rng.randn(n_samples, n_components))
    H = np.abs(rng.randn(n_components, n_features))

    WH_safe = nmf._special_sparse_dot(W, H, X_csr)
    WH = nmf._special_sparse_dot(W, H, X)

    # test that both results have same values, in X_csr nonzero elements
    ii, jj = X_csr.nonzero()
    WH_safe_data = np.asarray(WH_safe[ii, jj]).ravel()
    assert_array_almost_equal(WH_safe_data, WH[ii, jj], decimal=10)

    # test that WH_safe and X_csr have the same sparse structure
    assert_array_equal(WH_safe.indices, X_csr.indices)
    assert_array_equal(WH_safe.indptr, X_csr.indptr)
    assert_array_equal(WH_safe.shape, X_csr.shape)


@ignore_warnings(category=ConvergenceWarning)
def test_nmf_multiplicative_update_sparse():
    # Compare sparse and dense input in multiplicative update NMF
    # Also test continuity of the results with respect to beta_loss parameter
    n_samples = 20
    n_features = 10
    n_components = 5
    alpha = 0.1
    l1_ratio = 0.5
    n_iter = 20

    # initialization
    rng = np.random.mtrand.RandomState(1337)
    X = rng.randn(n_samples, n_features)
    X = np.abs(X)
    X_csr = sp.csr_matrix(X)
    W0, H0, _, _ = nmf._initialize_nmf(X, n_components, init='random',
                                       random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1., 2., 2.5):
        # Reference with dense array X
        W, H = W0.copy(), H0.copy()
        W1, H1, _ = non_negative_factorization(
            X, W, H, n_components, init='custom', update_H=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        # Compare with sparse X
        W, H = W0.copy(), H0.copy()
        W2, H2, _ = non_negative_factorization(
            X_csr, W, H, n_components, init='custom', update_H=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        assert_array_almost_equal(W1, W2, decimal=7)
        assert_array_almost_equal(H1, H2, decimal=7)

        # Compare with almost same beta_loss, since some values have a specific
        # behavior, but the results should be continuous w.r.t beta_loss
        beta_loss -= 1.e-5
        W, H = W0.copy(), H0.copy()
        W3, H3, _ = non_negative_factorization(
            X_csr, W, H, n_components, init='custom', update_H=True,
            solver='mu', beta_loss=beta_loss, max_iter=n_iter, alpha=alpha,
            l1_ratio=l1_ratio, regularization='both', random_state=42)

        assert_array_almost_equal(W1, W3, decimal=4)
        assert_array_almost_equal(H1, H3, decimal=4)


def test_nmf_negative_beta_loss():
    # Test that an error is raised if beta_loss < 0 and X contains zeros.
    # Test that the output has not NaN values when the input contains zeros.
    n_samples = 6
    n_features = 5
    n_components = 3

    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.clip(X, 0, None, out=X)
    X_csr = sp.csr_matrix(X)

    def _assert_nmf_no_nan(X, beta_loss):
        W, H, _ = non_negative_factorization(
            X, init='random', n_components=n_components, solver='mu',
            beta_loss=beta_loss, random_state=0, max_iter=1000)
        assert not np.any(np.isnan(W))
        assert not np.any(np.isnan(H))

    msg = "When beta_loss <= 0 and X contains zeros, the solver may diverge."
    for beta_loss in (-0.6, 0.):
        assert_raise_message(ValueError, msg, _assert_nmf_no_nan, X, beta_loss)
        _assert_nmf_no_nan(X + 1e-9, beta_loss)

    for beta_loss in (0.2, 1., 1.2, 2., 2.5):
        _assert_nmf_no_nan(X, beta_loss)
        _assert_nmf_no_nan(X_csr, beta_loss)


def test_nmf_regularization():
    # Test the effect of L1 and L2 regularizations
    n_samples = 6
    n_features = 5
    n_components = 3
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(n_samples, n_features))

    # L1 regularization should increase the number of zeros
    l1_ratio = 1.
    for solver in ['cd', 'mu']:
        regul = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0.5, l1_ratio=l1_ratio, random_state=42)
        model = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0., l1_ratio=l1_ratio, random_state=42)

        W_regul = regul.fit_transform(X)
        W_model = model.fit_transform(X)

        H_regul = regul.components_
        H_model = model.components_

        W_regul_n_zeros = W_regul[W_regul == 0].size
        W_model_n_zeros = W_model[W_model == 0].size
        H_regul_n_zeros = H_regul[H_regul == 0].size
        H_model_n_zeros = H_model[H_model == 0].size

        assert W_regul_n_zeros > W_model_n_zeros
        assert H_regul_n_zeros > H_model_n_zeros

    # L2 regularization should decrease the mean of the coefficients
    l1_ratio = 0.
    for solver in ['cd', 'mu']:
        regul = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0.5, l1_ratio=l1_ratio, random_state=42)
        model = nmf.NMF(n_components=n_components, solver=solver,
                        alpha=0., l1_ratio=l1_ratio, random_state=42)

        W_regul = regul.fit_transform(X)
        W_model = model.fit_transform(X)

        H_regul = regul.components_
        H_model = model.components_

        assert W_model.mean() > W_regul.mean()
        assert H_model.mean() > H_regul.mean()


@ignore_warnings(category=ConvergenceWarning)
def test_nmf_decreasing():
    # test that the objective function is decreasing at each iteration
    n_samples = 20
    n_features = 15
    n_components = 10
    alpha = 0.1
    l1_ratio = 0.5
    tol = 0.

    # initialization
    rng = np.random.mtrand.RandomState(42)
    X = rng.randn(n_samples, n_features)
    np.abs(X, X)
    W0, H0, _, _ = nmf._initialize_nmf(X, n_components, init='random',
                                       random_state=42)

    for beta_loss in (-1.2, 0, 0.2, 1., 2., 2.5):
        for solver in ('cd', 'mu'):
            if solver != 'mu' and beta_loss != 2:
                # not implemented
                continue
            W, H = W0.copy(), H0.copy()
            previous_loss = None
            for _ in range(30):
                # one more iteration starting from the previous results
                W, H, _ = non_negative_factorization(
                    X, W, H, beta_loss=beta_loss, init='custom',
                    n_components=n_components, max_iter=1, alpha=alpha,
                    solver=solver, tol=tol, l1_ratio=l1_ratio, verbose=0,
                    regularization='both', random_state=0, update_H=True)

                loss = nmf._beta_divergence(X, W, H, beta_loss)
                if previous_loss is not None:
                    assert previous_loss > loss
                previous_loss = loss


def test_nmf_underflow():
    # Regression test for an underflow issue in _beta_divergence
    rng = np.random.RandomState(0)
    n_samples, n_features, n_components = 10, 2, 2
    X = np.abs(rng.randn(n_samples, n_features)) * 10
    W = np.abs(rng.randn(n_samples, n_components)) * 10
    H = np.abs(rng.randn(n_components, n_features))

    X[0, 0] = 0
    ref = nmf._beta_divergence(X, W, H, beta=1.0)
    X[0, 0] = 1e-323
    res = nmf._beta_divergence(X, W, H, beta=1.0)
    assert_almost_equal(res, ref)


@pytest.mark.parametrize("dtype_in, dtype_out", [
    (np.float32, np.float32),
    (np.float64, np.float64),
    (np.int32, np.float64),
    (np.int64, np.float64)])
@pytest.mark.parametrize("solver", ["cd", "mu"])
@pytest.mark.parametrize("regularization",
                         (None, "both", "components", "transformation"))
def test_nmf_dtype_match(dtype_in, dtype_out, solver, regularization):
    # Check that NMF preserves dtype (float32 and float64)
    X = np.random.RandomState(0).randn(20, 15).astype(dtype_in, copy=False)
    np.abs(X, out=X)
    nmf = NMF(solver=solver, regularization=regularization)

    assert nmf.fit(X).transform(X).dtype == dtype_out
    assert nmf.fit_transform(X).dtype == dtype_out
    assert nmf.components_.dtype == dtype_out


@pytest.mark.parametrize("solver", ["cd", "mu"])
@pytest.mark.parametrize("regularization",
                         (None, "both", "components", "transformation"))
def test_nmf_float32_float64_consistency(solver, regularization):
    # Check that the result of NMF is the same between float32 and float64
    X = np.random.RandomState(0).randn(50, 7)
    np.abs(X, out=X)
    nmf32 = NMF(solver=solver, regularization=regularization, random_state=0)
    W32 = nmf32.fit_transform(X.astype(np.float32))
    nmf64 = NMF(solver=solver, regularization=regularization, random_state=0)
    W64 = nmf64.fit_transform(X)

    assert_allclose(W32, W64, rtol=1e-6, atol=1e-5)


def test_nmf_custom_init_dtype_error():
    # Check that an error is raise if custom H and/or W don't have the same
    # dtype as X.
    rng = np.random.RandomState(0)
    X = rng.random_sample((20, 15))
    H = rng.random_sample((15, 15)).astype(np.float32)
    W = rng.random_sample((20, 15))

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        NMF(init='custom').fit(X, H=H, W=W)

    with pytest.raises(TypeError, match="should have the same dtype as X"):
        non_negative_factorization(X, H=H, update_H=False)


def test_nmf_close_minibatch_nmf():
    # Test that the decomposition with standard and minbatch nmf
    # gives close results
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(48, 5))
    nmf = NMF(5, solver='mu', init='nndsvdar', random_state=0,
              max_iter=2000, beta_loss='kullback-leibler')
    mbnmf = MiniBatchNMF(5, solver='mu', init='nndsvdar', random_state=0,
                         max_iter=2000, beta_loss='kullback-leibler',
                         batch_size=48)
    W = nmf.fit_transform(X)
    mbW = mbnmf.fit_transform(X)
    assert_array_almost_equal(W, mbW)


def test_nmf_online_partial_fit():
    rng = np.random.mtrand.RandomState(42)
    X = np.abs(rng.randn(48, 5))
    mbnmf1 = MiniBatchNMF(5, solver='mu', init='nndsvdar', random_state=0,
                          max_iter=1, beta_loss='kullback-leibler',
                          batch_size=48).fit(X)
    mbnmf2 = MiniBatchNMF(5, solver='mu', init='nndsvdar', random_state=0,
                          max_iter=1,beta_loss='kullback-leibler',
                          batch_size=48)
    mbnmf2.partial_fit(X)

    assert mbnmf1.n_iter_ == mbnmf2.n_iter_
    assert_array_almost_equal(mbnmf1.components_, mbnmf2.components_,
                              decimal=2)
