""" Non-negative matrix factorization.
"""
# Author: Vlad Niculae
#         Lars Buitinck
#         Mathieu Blondel <mathieu@mblondel.org>
#         Tom Dupre la Tour
# License: BSD 3 clause

import numbers
import numpy as np
import scipy.sparse as sp
import time
import itertools
import warnings
from math import sqrt

from ._cdnmf_fast import _update_cdnmf_fast
from .._config import config_context
from ..base import BaseEstimator, TransformerMixin
from ..exceptions import ConvergenceWarning
from ..utils import check_random_state, check_array, gen_batches
from ..utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from ..utils.validation import check_is_fitted, check_non_negative

EPSILON = np.finfo(np.float32).eps


def norm(x):
    """Dot product-based Euclidean norm implementation.

    See: http://fa.bianp.net/blog/2011/computing-the-vector-norm/

    Parameters
    ----------
    x : array-like
        Vector for which to compute the norm.
    """
    return sqrt(squared_norm(x))


def trace_dot(X, Y):
    """Trace of np.dot(X, Y.T).

    Parameters
    ----------
    X : array-like
        First matrix.
    Y : array-like
        Second matrix.
    """
    return np.dot(X.ravel(), Y.ravel())


def _check_init(A, shape, whom):
    A = check_array(A)
    if np.shape(A) != shape:
        raise ValueError(
            "Array with wrong shape passed to %s. Expected %s, "
            "but got %s " % (whom, shape, np.shape(A))
        )
    check_non_negative(A, whom)
    if np.max(A) == 0:
        raise ValueError("Array passed to %s is full of zeros." % whom)


def _beta_divergence(X, W, H, beta, square_root=False):
    """Compute the beta-divergence of X and dot(W, H).

    Parameters
    ----------
    X : float or array-like of shape (n_samples, n_features)

    W : float or array-like of shape (n_samples, n_components)

    H : float or array-like of shape (n_components, n_features)

    beta : float or {'frobenius', 'kullback-leibler', 'itakura-saito'}
        Parameter of the beta-divergence.
        If beta == 2, this is half the Frobenius *squared* norm.
        If beta == 1, this is the generalized Kullback-Leibler divergence.
        If beta == 0, this is the Itakura-Saito divergence.
        Else, this is the general beta-divergence.

    square_root : bool, default=False
        If True, return np.sqrt(2 * res)
        For beta == 2, it corresponds to the Frobenius norm.

    Returns
    -------
        res : float
            Beta divergence of X and np.dot(X, H).
    """
    beta = _beta_loss_to_float(beta)

    # The method can be called with scalars
    if not sp.issparse(X):
        X = np.atleast_2d(X)
    W = np.atleast_2d(W)
    H = np.atleast_2d(H)

    # Frobenius norm
    if beta == 2:
        # Avoid the creation of the dense np.dot(W, H) if X is sparse.
        if sp.issparse(X):
            norm_X = np.dot(X.data, X.data)
            norm_WH = trace_dot(np.linalg.multi_dot([W.T, W, H]), H)
            cross_prod = trace_dot((X * H.T), W)
            res = (norm_X + norm_WH - 2.0 * cross_prod) / 2.0
        else:
            res = squared_norm(X - np.dot(W, H)) / 2.0

        if square_root:
            return np.sqrt(res * 2)
        else:
            return res

    if sp.issparse(X):
        # compute np.dot(W, H) only where X is nonzero
        WH_data = _special_sparse_dot(W, H, X).data
        X_data = X.data
    else:
        WH = np.dot(W, H)
        WH_data = WH.ravel()
        X_data = X.ravel()

    # do not affect the zeros: here 0 ** (-1) = 0 and not infinity
    indices = X_data > EPSILON
    WH_data = WH_data[indices]
    X_data = X_data[indices]

    # used to avoid division by zero
    WH_data[WH_data == 0] = EPSILON

    # generalized Kullback-Leibler divergence
    if beta == 1:
        # fast and memory efficient computation of np.sum(np.dot(W, H))
        sum_WH = np.dot(np.sum(W, axis=0), np.sum(H, axis=1))
        # computes np.sum(X * log(X / WH)) only where X is nonzero
        div = X_data / WH_data
        res = np.dot(X_data, np.log(div))
        # add full np.sum(np.dot(W, H)) - np.sum(X)
        res += sum_WH - X_data.sum()

    # Itakura-Saito divergence
    elif beta == 0:
        div = X_data / WH_data
        res = np.sum(div) - np.product(X.shape) - np.sum(np.log(div))

    # beta-divergence, beta not in (0, 1, 2)
    else:
        if sp.issparse(X):
            # slow loop, but memory efficient computation of :
            # np.sum(np.dot(W, H) ** beta)
            sum_WH_beta = 0
            for i in range(X.shape[1]):
                sum_WH_beta += np.sum(np.dot(W, H[:, i]) ** beta)

        else:
            sum_WH_beta = np.sum(WH ** beta)

        sum_X_WH = np.dot(X_data, WH_data ** (beta - 1))
        res = (X_data ** beta).sum() - beta * sum_X_WH
        res += sum_WH_beta * (beta - 1)
        res /= beta * (beta - 1)

    if square_root:
        return np.sqrt(2 * res)
    else:
        return res


def _special_sparse_dot(W, H, X):
    """Computes np.dot(W, H), only where X is non zero."""
    if sp.issparse(X):
        ii, jj = X.nonzero()
        n_vals = ii.shape[0]
        dot_vals = np.empty(n_vals)
        n_components = W.shape[1]

        batch_size = max(n_components, n_vals // n_components)
        for start in range(0, n_vals, batch_size):
            batch = slice(start, start + batch_size)
            dot_vals[batch] = np.multiply(W[ii[batch], :], H.T[jj[batch], :]).sum(
                axis=1
            )

        WH = sp.coo_matrix((dot_vals, (ii, jj)), shape=X.shape)
        return WH.tocsr()
    else:
        return np.dot(W, H)


def _compute_regularization(alpha, l1_ratio, regularization):
    """Compute L1 and L2 regularization coefficients for W and H."""
    alpha_H = 0.0
    alpha_W = 0.0
    if regularization in ("both", "components"):
        alpha_H = float(alpha)
    if regularization in ("both", "transformation"):
        alpha_W = float(alpha)

    l1_reg_W = alpha_W * l1_ratio
    l1_reg_H = alpha_H * l1_ratio
    l2_reg_W = alpha_W * (1.0 - l1_ratio)
    l2_reg_H = alpha_H * (1.0 - l1_ratio)
    return l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H


def _beta_loss_to_float(beta_loss):
    """Convert string beta_loss to float."""
    allowed_beta_loss = {"frobenius": 2, "kullback-leibler": 1, "itakura-saito": 0}
    if isinstance(beta_loss, str) and beta_loss in allowed_beta_loss:
        beta_loss = allowed_beta_loss[beta_loss]

    if not isinstance(beta_loss, numbers.Number):
        raise ValueError(
            "Invalid beta_loss parameter: got %r instead "
            "of one of %r, or a float." % (beta_loss, allowed_beta_loss.keys())
        )
    return beta_loss


def _initialize_nmf(X, n_components, init="warn", eps=1e-6, random_state=None):
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'nndsvda', 'nndsvdar'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - None: 'nndsvd' if n_components <= min(n_samples, n_features),
            otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H

    eps : float, default=1e-6
        Truncate all values less then this in output to zero.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : array-like of shape (n_samples, n_components)
        Initial guesses for solving X ~= WH.

    H : array-like of shape (n_components, n_features)
        Initial guesses for solving X ~= WH.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd
    """
    if init == "warn":
        warnings.warn(
            (
                "The 'init' value, when 'init=None' and "
                "n_components is less than n_samples and "
                "n_features, will be changed from 'nndsvd' to "
                "'nndsvda' in 1.1 (renaming of 0.26)."
            ),
            FutureWarning,
        )
        init = None

    check_non_negative(X, "NMF initialization")
    n_samples, n_features = X.shape

    if (
        init is not None
        and init != "random"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvd"
        else:
            init = "random"

    # Random initialization
    if init == "random":
        avg = np.sqrt(X.mean() / n_components)
        rng = check_random_state(random_state)
        H = avg * rng.randn(n_components, n_features).astype(X.dtype, copy=False)
        W = avg * rng.randn(n_samples, n_components).astype(X.dtype, copy=False)
        np.abs(H, out=H)
        np.abs(W, out=W)
        return W, H

    # NNDSVD initialization
    U, S, V = randomized_svd(X, n_components, random_state=random_state)
    W = np.zeros_like(U)
    H = np.zeros_like(V)

    # The leading singular triplet is non-negative
    # so it can be used as is for initialization.
    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(V[0, :])

    for j in range(1, n_components):
        x, y = U[:, j], V[j, :]

        # extract positive and negative parts of column vectors
        x_p, y_p = np.maximum(x, 0), np.maximum(y, 0)
        x_n, y_n = np.abs(np.minimum(x, 0)), np.abs(np.minimum(y, 0))

        # and their norms
        x_p_nrm, y_p_nrm = norm(x_p), norm(y_p)
        x_n_nrm, y_n_nrm = norm(x_n), norm(y_n)

        m_p, m_n = x_p_nrm * y_p_nrm, x_n_nrm * y_n_nrm

        # choose update
        if m_p > m_n:
            u = x_p / x_p_nrm
            v = y_p / y_p_nrm
            sigma = m_p
        else:
            u = x_n / x_n_nrm
            v = y_n / y_n_nrm
            sigma = m_n

        lbd = np.sqrt(S[j] * sigma)
        W[:, j] = lbd * u
        H[j, :] = lbd * v

    W[W < eps] = 0
    H[H < eps] = 0

    if init == "nndsvd":
        pass
    elif init == "nndsvda":
        avg = X.mean()
        W[W == 0] = avg
        H[H == 0] = avg
    elif init == "nndsvdar":
        rng = check_random_state(random_state)
        avg = X.mean()
        W[W == 0] = abs(avg * rng.randn(len(W[W == 0])) / 100)
        H[H == 0] = abs(avg * rng.randn(len(H[H == 0])) / 100)
    else:
        raise ValueError(
            "Invalid init parameter: got %r instead of one of %r"
            % (init, (None, "random", "nndsvd", "nndsvda", "nndsvdar"))
        )

    return W, H


def _update_coordinate_descent(X, W, Ht, l1_reg, l2_reg, shuffle, random_state):
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).

    """
    n_components = Ht.shape[1]

    HHt = np.dot(Ht.T, Ht)
    XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.0:
        # adds l2_reg only on the diagonal
        HHt.flat[:: n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.0:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)


def _fit_coordinate_descent(
    X,
    W,
    H,
    tol=1e-4,
    max_iter=200,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
    shuffle=False,
    random_state=None,
):
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent

    The objective function is minimized with an alternating minimization of W
    and H. Each minimization is done with a cyclic (up to a permutation of the
    features) Coordinate Descent.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    random_state : int, RandomState instance or None, default=None
        Used to randomize the coordinates in the CD solver, when
        ``shuffle`` is set to ``True``. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    References
    ----------
    Cichocki, Andrzej, and Phan, Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.
    """
    # so W and Ht are both in C order in memory
    Ht = check_array(H.T, order="C")
    X = check_array(X, accept_sparse="csr")

    rng = check_random_state(random_state)

    for n_iter in range(1, max_iter + 1):
        violation = 0.0

        # Update W
        violation += _update_coordinate_descent(
            X, W, Ht, l1_reg_W, l2_reg_W, shuffle, rng
        )
        # Update H
        if update_H:
            violation += _update_coordinate_descent(
                X.T, Ht, W, l1_reg_H, l2_reg_H, shuffle, rng
            )

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        if verbose:
            print("violation:", violation / violation_init)

        if violation / violation_init <= tol:
            if verbose:
                print("Converged at iteration", n_iter + 1)
            break

    return W, Ht.T, n_iter


def _multiplicative_update_w(
    X,
    W,
    H,
    beta_loss,
    l1_reg_W,
    l2_reg_W,
    gamma,
    H_sum=None,
    HHt=None,
    XHt=None,
    update_H=True,
):
    """Update W in Multiplicative Update NMF."""
    if beta_loss == 2:
        # Numerator
        if XHt is None:
            XHt = safe_sparse_dot(X, H.T)
        if update_H:
            # avoid a copy of XHt, which will be re-computed (update_H=True)
            numerator = XHt
        else:
            # preserve the XHt, which is not re-computed (update_H=False)
            numerator = XHt.copy()

        numerator = numerator[0 : W.shape[0], 0 : W.shape[1]]

        # Denominator
        if HHt is None:
            HHt = np.dot(H, H.T)
        denominator = np.dot(W, HHt)

    else:
        # Numerator
        # if X is sparse, compute WH only where X is non zero
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH == 0] = EPSILON

        # to avoid taking a negative power of zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(X * (dot(W, H) ** (beta_loss - 2)), H.T)
        numerator = safe_sparse_dot(WH_safe_X, H.T)
        numerator = numerator[0 : W.shape[0], 0 : W.shape[1]]

        # Denominator
        if beta_loss == 1:
            if H_sum is None:
                H_sum = np.sum(H, axis=1)  # shape(n_components, )
            denominator = H_sum[np.newaxis, :]

        else:
            # computation of WHHt = dot(dot(W, H) ** beta_loss - 1, H.T)
            if sp.issparse(X):
                # memory efficient computation
                # (compute row by row, avoiding the dense matrix WH)
                WHHt = np.empty(W.shape)
                for i in range(X.shape[0]):
                    WHi = np.dot(W[i, :], H)
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WHHt[i, :] = np.dot(WHi, H.T)
            else:
                WH **= beta_loss - 1
                WHHt = np.dot(WH, H.T)
            denominator = WHHt

    # Add L1 and L2 regularization
    if l1_reg_W > 0:
        denominator += l1_reg_W
    if l2_reg_W > 0:
        denominator = denominator + l2_reg_W * W
    denominator[denominator == 0] = EPSILON

    numerator /= denominator
    delta_W = numerator

    # gamma is in ]0, 1]
    if gamma != 1:
        delta_W **= gamma

    return delta_W, H_sum, HHt, XHt


def _multiplicative_update_h(X, W, H, A, B, beta_loss, l1_reg_H, l2_reg_H, gamma, rho):

    """update H in Multiplicative Update NMF.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    A : array-like of shape (n_components, n_features)
        Initial guess for the numerator auxiliary function.
        Used in the batch case only.

    B : array-like of shape (n_components, n_features)
        Initial guess for the denominator auxiliary function.
        Used in the batch case only.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. When
        `batch_size` is not `None` `beta_loss` cannot be `'frobenius'`.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    gamma : float, default=1.
        Exponent for Maximization-Minimization (MM) algorithm
        [Fevotte 2011].

    rho : float.
        Scaling factor for past information for online and minibatch
        algorithm.

    Returns
    -------
    H : ndarray of shape (n_components, n_features)
        Updated matrix H.

    A : array-like of shape (n_components, n_features)
        Numerator auxiliary function, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.

    B : array-like of shape (n_components, n_features)
        Denominator auxiliary function, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.
    """

    if beta_loss == 2:
        numerator = safe_sparse_dot(W.T, X)
        denominator = np.linalg.multi_dot([W.T, W, H])

    else:
        # Numerator
        WH_safe_X = _special_sparse_dot(W, H, X)
        if sp.issparse(X):
            WH_safe_X_data = WH_safe_X.data
            X_data = X.data
        else:
            WH_safe_X_data = WH_safe_X
            X_data = X
            # copy used in the Denominator
            WH = WH_safe_X.copy()
            if beta_loss - 1.0 < 0:
                WH[WH == 0] = EPSILON

        # to avoid division by zero
        if beta_loss - 2.0 < 0:
            WH_safe_X_data[WH_safe_X_data == 0] = EPSILON

        if beta_loss == 1:
            np.divide(X_data, WH_safe_X_data, out=WH_safe_X_data)
        elif beta_loss == 0:
            # speeds up computation time
            # refer to /numpy/numpy/issues/9363
            WH_safe_X_data **= -1
            WH_safe_X_data **= 2
            # element-wise multiplication
            WH_safe_X_data *= X_data
        else:
            WH_safe_X_data **= beta_loss - 2
            # element-wise multiplication
            WH_safe_X_data *= X_data

        # here numerator = dot(W.T, (dot(W, H) ** (beta_loss - 2)) * X)
        numerator = safe_sparse_dot(W.T, WH_safe_X)

        # Denominator
        if beta_loss == 1:
            W_sum = np.sum(W, axis=0)  # shape(n_components, )
            W_sum[W_sum == 0] = 1.0
            denominator = W_sum[:, np.newaxis]

        # beta_loss not in (1, 2)
        else:
            # computation of WtWH = dot(W.T, dot(W, H) ** beta_loss - 1)
            if sp.issparse(X):
                # memory efficient computation
                # (compute column by column, avoiding the dense matrix WH)
                WtWH = np.empty(H.shape)
                for i in range(X.shape[1]):
                    WHi = np.dot(W, H[:, i])
                    if beta_loss - 1 < 0:
                        WHi[WHi == 0] = EPSILON
                    WHi **= beta_loss - 1
                    WtWH[:, i] = np.dot(W.T, WHi)
            else:
                WH **= beta_loss - 1
                WtWH = np.dot(W.T, WH)
            denominator = WtWH

    # Add L1 and L2 regularization
    if l1_reg_H > 0:
        denominator += l1_reg_H
    if l2_reg_H > 0:
        denominator = denominator + l2_reg_H * H
    denominator[denominator == 0] = EPSILON

    if A is not None and B is not None:
        if gamma != 1:
            H **= 1 / gamma
        numerator *= H
        A *= rho
        B *= rho
        A += numerator
        B += denominator
        H = A / B

        if gamma != 1:
            H **= gamma
    else:
        delta_H = numerator
        delta_H /= denominator
        if gamma != 1:
            delta_H **= gamma
        H *= delta_H

    return H, A, B


def _fit_multiplicative_update(
    X,
    W,
    H,
    A,
    B,
    beta_loss="frobenius",
    batch_size=None,
    iter_offset=0,
    max_iter=200,
    tol=1e-4,
    l1_reg_W=0,
    l1_reg_H=0,
    l2_reg_W=0,
    l2_reg_H=0,
    update_H=True,
    verbose=0,
    forget_factor=None,
):
    """Compute Non-negative Matrix Factorization with Multiplicative Update.

    The objective function is _beta_divergence(X, WH) and is minimized with an
    alternating minimization of W and H. Each minimization is done with a
    Multiplicative Update.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant input matrix.

    W : array-like of shape (n_samples, n_components)
        Initial guess for the solution.

    H : array-like of shape (n_components, n_features)
        Initial guess for the solution.

    A : array-like of shape (n_components, n_features)
        Initial guess for the numerator auxiliary function

    B : array-like of shape (n_components, n_features)
        Initial guess for the denominator auxiliary function

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        String must be in {'frobenius', 'kullback-leibler', 'itakura-saito'}.
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. When `batch_size` is not `None`
        `beta_loss` cannot be `'frobenius'`.

    batch_size : int, default=None
        Number of samples in each mini-batch.
        Used in the batch case only.

    iter_offset : int, default=0
        Number of previous iterations completed used for
        initialization, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.

    max_iter : int, default=200
        Number of iterations.

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    l1_reg_W : float, default=0.
        L1 regularization parameter for W.

    l1_reg_H : float, default=0.
        L1 regularization parameter for H.

    l2_reg_W : float, default=0.
        L2 regularization parameter for W.

    l2_reg_H : float, default=0.
        L2 regularization parameter for H.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    verbose : int, default=0
        The verbosity level.

    forget_factor : float, default=None
        Amount of rescaling of past information. Its value is 1 for batch
        NMF algorithm, it could be <1 for online NMF algorithm.
        When r<0.5 the solution is unstable.

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    iter_offset : int
        The number of iteration on data batches that has been
        performed, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.

    A : array-like of shape (n_components, n_features)
        Numerator auxiliary function, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.

    B : array-like of shape (n_components, n_features)
        Denominator auxiliary function, only used in
        :class:`sklearn.decomposition.MiniBatchNMF`.

    References
    ----------
    Lee, D. D., & Seung, H., S. (2001). Algorithms for Non-negative Matrix
    Factorization. Adv. Neural Inform. Process. Syst.. 13.
    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """
    start_time = time.time()

    n_samples = X.shape[0]

    rho = 0.0
    if forget_factor is not None:
        rho = forget_factor ** (batch_size / n_samples)

    beta_loss = _beta_loss_to_float(beta_loss)

    # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
    if beta_loss < 1:
        gamma = 1.0 / (2.0 - beta_loss)
    elif beta_loss > 2:
        gamma = 1.0 / (beta_loss - 1.0)
    else:
        gamma = 1.0

    # used for the convergence criterion
    error_at_init = _beta_divergence(X, W, H, beta_loss, square_root=True)
    previous_error = error_at_init

    H_sum, HHt, XHt = None, None, None

    if batch_size is None:
        batch_size = n_samples

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)
    n_batches = int(np.ceil(n_samples / batch_size))
    n_steps = max_iter * n_batches
    for n_i, batch in zip(range(n_steps), batches):
        # update W
        # H_sum, HHt are saved and reused if not update_H
        delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
            X[batch],
            W[batch],
            H,
            beta_loss,
            l1_reg_W,
            l2_reg_W,
            gamma,
            H_sum,
            HHt,
            XHt,
            update_H,
        )
        W[batch] *= delta_W

        # necessary for stability with beta_loss < 1
        if beta_loss < 1:
            W[batch][W[batch] < np.finfo(np.float64).eps] = 0.0

        # update H
        if update_H:
            H, A, B = _multiplicative_update_h(
                X[batch], W[batch], H, A, B, beta_loss, l1_reg_H, l2_reg_H, gamma, rho
            )

            # These values will be recomputed since H changed
            H_sum, HHt, XHt = None, None, None

            # necessary for stability with beta_loss < 1
            if beta_loss <= 1:
                H[H < np.finfo(np.float64).eps] = 0.0

        # XHt is updated if batch_size is smaller than n_samples
        if batch_size < n_samples:
            XHt = None

        # test convergence criterion every 10 iterations
        if tol > 0 and n_i % (10 * n_batches) == 0:
            error = _beta_divergence(X, W, H, beta_loss, square_root=True)
            if verbose:
                iter_time = time.time()
                print(
                    "Epoch %02d reached after %.3f seconds, error: %f"
                    % (n_i, iter_time - start_time, error)
                )

            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    # do not print if we have already printed in the convergence test
    if verbose and (tol == 0 or n_i % (10 * n_batches) != 0):
        end_time = time.time()
        print("Epoch %02d reached after %.3f seconds." % (n_i, end_time - start_time))

    if forget_factor is None:
        n_iter = n_i + 1
        return W, H, n_iter
    else:
        n_iter = int(np.ceil((n_i + 1) / n_batches))
        iter_offset = n_i - (n_iter * n_batches)
        return W, H, n_iter, iter_offset, A, B


def non_negative_factorization(
    X,
    W=None,
    H=None,
    n_components=None,
    *,
    init="warn",
    update_H=True,
    solver="cd",
    batch_size=None,
    beta_loss="frobenius",
    tol=1e-4,
    max_iter=200,
    alpha=0.0,
    l1_ratio=0.0,
    regularization=None,
    random_state=None,
    verbose=0,
    shuffle=False,
    forget_factor=None,
):
    """Compute Non-negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            0.5 * ||X - WH||_{loss}^2 + alpha * l1_{ratio} * ||vec(W)||_1

            + alpha * l1_{ratio} * ||vec(H)||_1

            + 0.5 * alpha * (1 - l1_{ratio}) * ||W||_{Fro}^2

            + 0.5 * alpha * (1 - l1_{ratio}) * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of W
    and H. If H is given and update_H=False, it solves for W only.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Constant matrix.

    W : array-like of shape (n_samples, n_components), default=None
        If init='custom', it is used as initial guess for the solution.

    H : array-like of shape (n_components, n_features), default=None
        If init='custom', it is used as initial guess for the solution.
        If update_H=False, it is used as a constant, to solve for W only.

    n_components : int, default=None
        Number of components, if n_components is not set all features
        are kept.

    batch_size : int, default=None
        Number of samples per batch: setting `batch_size != None`
        will select the MiniBatch implementation.

        .. versionadded:: 1.0

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.

        Valid options:

        - None: 'nndsvd' if n_components < n_features, otherwise 'random'.

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components)

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'nndsvda': NNDSVD with zeros filled with the average of X
            (better when sparsity is not desired)

        - 'nndsvdar': NNDSVD with zeros filled with small random values
            (generally faster, less accurate alternative to NNDSVDa
            for when sparsity is not desired)

        - 'custom': use custom matrices W and H if `update_H=True`. If
          `update_H=False`, then only custom matrix H is used.

        .. versionchanged:: 0.23
            The default value of `init` changed from 'random' to None in 0.23.

    update_H : bool, default=True
        Set to True, both W and H will be estimated from initial guesses.
        Set to False, only W will be estimated.

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:

        - 'cd' is a Coordinate Descent solver that uses Fast Hierarchical
            Alternating Least Squares (Fast HALS).

        - 'mu' is a Multiplicative Update solver
            This is the only solver available when `batch_size` is not `None`.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver. When
        `batch_size` is not `None` `beta_loss` cannot be `'frobenius'`.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    alpha : float, default=0.
        Constant that multiplies the regularization terms.

    l1_ratio : float, default=0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    regularization : {'both', 'components', 'transformation'}, default=None
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.

    random_state : int, RandomState instance or None, default=None
        Used for NMF initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    verbose : int, default=0
        The verbosity level.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

    forget_factor : float, default=None.
        Amount of rescaling of past information. Only for
        MiniBatch implementation.

        .. versionadded:: 1.0

    Returns
    -------
    W : ndarray of shape (n_samples, n_components)
        Solution to the non-negative least squares problem.

    H : ndarray of shape (n_components, n_features)
        Solution to the non-negative least squares problem.

    n_iter : int
        Actual number of iterations.

    iter_offset : int
        The number of iteration on data batches that has been
        performed. Only returned if `batch_size` is not `None`.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import non_negative_factorization
    >>> W, H, n_iter = non_negative_factorization(X, n_components=2,
    ... init='random', random_state=0)

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).

    Lefevre, A., Bach, F., Fevotte, C. (2011). Online algorithms for
    nonnegative matrix factorization with the Itakura-Saito divergence.
    WASPA (https://doi.org/10.1109/ASPAA.2011.6082314,
    https://hal.archives-ouvertes.fr/hal-00602050)
    """
    X = check_array(X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32])

    if batch_size is None:
        est = NMF(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=shuffle,
            regularization=regularization,
        )

        with config_context(assume_finite=True):
            W, H, n_iter = est._fit_transform(X, W=W, H=H, update_H=update_H)

        return W, H, n_iter
    else:
        est = MiniBatchNMF(
            n_components=n_components,
            init=init,
            batch_size=batch_size,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            forget_factor=forget_factor,
            verbose=verbose,
            regularization=regularization,
        )

        with config_context(assume_finite=True):
            W, H, n_iter, iter_offset, A, B = est._fit_transform(
                X, W=W, H=H, update_H=update_H
            )

        return W, H, n_iter, iter_offset, A, B


class NMF(TransformerMixin, BaseEstimator):
    """Non-Negative Matrix Factorization (NMF).

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            0.5 * ||X - WH||_{loss}^2 + alpha * l1_{ratio} * ||vec(W)||_1

            + alpha * l1_{ratio} * ||vec(H)||_1

            + 0.5 * alpha * (1 - l1_{ratio}) * ||W||_{Fro}^2

            + 0.5 * alpha * (1 - l1_{ratio}) * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of W
    and H.

    Read more in the :ref:`User Guide <NMF>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of components, if n_components is not set all features
        are kept.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - `None`: 'nndsvd' if n_components <= min(n_samples, n_features),
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          sqrt(X.mean() / n_components)

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)

        - `'custom'`: use custom matrices W and H

    solver : {'cd', 'mu'}, default='cd'
        Numerical solver to use:
        'cd' is a Coordinate Descent solver.
        'mu' is a Multiplicative Update solver.

        .. versionadded:: 0.17
           Coordinate Descent solver.

        .. versionadded:: 0.19
           Multiplicative Update solver.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros. Used only in 'mu' solver.

        .. versionadded:: 0.19

    tol : float, default=1e-4
        Tolerance of the stopping condition.

    max_iter : int, default=200
        Maximum number of iterations before timing out.

    random_state : int, RandomState instance or None, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    alpha : float, default=0.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.

        .. versionadded:: 0.17
           *alpha* used in the Coordinate Descent solver.

    l1_ratio : float, default=0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

        .. versionadded:: 0.17
           Regularization parameter *l1_ratio* used in the Coordinate Descent
           solver.

    verbose : int, default=0
        Whether to be verbose.

    shuffle : bool, default=False
        If true, randomize the order of coordinates in the CD solver.

        .. versionadded:: 0.17
           *shuffle* parameter used in the Coordinate Descent solver.

    regularization : {'both', 'components', 'transformation', None}, \
                     default='both'
        Select whether the regularization affects the components (H), the
        transformation (W), both or none of them.

        .. versionadded:: 0.24

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : int
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : float
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).
    """

    def __init__(
        self,
        n_components=None,
        *,
        init="warn",
        solver="cd",
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        shuffle=False,
        regularization="both",
    ):
        self.n_components = n_components
        self.init = init
        self.solver = solver
        self.beta_loss = beta_loss
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.verbose = verbose
        self.shuffle = shuffle
        self.regularization = regularization

    def _more_tags(self):
        return {"requires_positive_X": True}

    def _check_params(self, X):
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]
        if (
            not isinstance(self._n_components, numbers.Integral)
            or self._n_components <= 0
        ):
            raise ValueError(
                "Number of components must be a positive integer;"
                " got (n_components=%r)" % self._n_components
            )
        if not isinstance(self.max_iter, numbers.Integral) or self.max_iter < 0:
            raise ValueError(
                "Maximum number of iterations must be a positive "
                "integer; got (max_iter=%r)" % self.max_iter
            )
        if not isinstance(self.tol, numbers.Number) or self.tol < 0:
            raise ValueError(
                "Tolerance for stopping criteria must be "
                "positive; got (tol=%r)" % self.tol
            )
        allowed_solver = ("cd", "mu")
        if self.solver not in allowed_solver:
            raise ValueError(
                "Invalid solver parameter: got %r instead of one of %r"
                % (self.solver, allowed_solver)
            )

        allowed_regularization = ("both", "components", "transformation", None)
        if self.regularization not in allowed_regularization:
            raise ValueError(
                "Invalid regularization parameter: got %r instead of "
                "one of %r" % (self.regularization, allowed_regularization)
            )

        # 'mu' is the only solver that handles other beta losses
        # than 'frobenius'
        if self.solver != "mu" and self.beta_loss not in (2, "frobenius"):
            raise ValueError(
                "Invalid beta_loss parameter: solver %r does not handle "
                "beta_loss = %r" % (self.solver, self.beta_loss)
            )

        if self.solver == "mu" and self.init == "nndsvd":
            warnings.warn(
                "The multiplicative update ('mu') solver cannot "
                "update zeros present in the initialization, "
                "and so leads to poorer results when used jointly "
                "with init='nndsvd'. You may try init='nndsvda' "
                "or init='nndsvdar' instead.",
                UserWarning,
            )

        self._beta_loss = _beta_loss_to_float(self.beta_loss)

        return self

    def _check_w_h(self, X, W, H, update_H):
        # check W and H, or initialize them
        n_samples, n_features = X.shape
        if self.init == "custom" and update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            _check_init(W, (n_samples, self._n_components), "NMF (input W)")
            if H.dtype != X.dtype or W.dtype != X.dtype:
                raise TypeError(
                    "H and W should have the same dtype as X. Got "
                    "H.dtype = {} and W.dtype = {}.".format(H.dtype, W.dtype)
                )
        elif not update_H:
            _check_init(H, (self._n_components, n_features), "NMF (input H)")
            if H.dtype != X.dtype:
                raise TypeError(
                    "H should have the same dtype as X. Got "
                    "H.dtype = {}.".format(H.dtype)
                )
            # 'mu' solver should not be initialized by zeros
            if self.solver == "mu":
                avg = np.sqrt(X.mean() / self._n_components)
                W = np.full((n_samples, self._n_components), avg, dtype=X.dtype)
            else:
                W = np.zeros((n_samples, self._n_components), dtype=X.dtype)
        else:
            W, H = _initialize_nmf(
                X, self._n_components, init=self.init, random_state=self.random_state
            )
        return W, H

    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        with config_context(assume_finite=True):
            W, H, n_iter = self._fit_transform(X, W=W, H=H)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
            If update_H=False, it is used as a constant, to solve for W only.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.
        """
        check_non_negative(X, "NMF (input X)")
        # check parameters
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        n_samples, n_features = X.shape

        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)

        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
            self.alpha, self.l1_ratio, self.regularization
        )

        if self.solver == "cd":
            W, H, n_iter = _fit_coordinate_descent(
                X,
                W,
                H,
                self.tol,
                self.max_iter,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H=update_H,
                verbose=self.verbose,
                shuffle=self.shuffle,
                random_state=self.random_state,
            )
        elif self.solver == "mu":
            W, H, n_iter, *_ = _fit_multiplicative_update(
                X,
                W,
                H,
                None,
                None,
                self._beta_loss,
                None,
                0,
                self.max_iter,
                self.tol,
                l1_reg_W,
                l1_reg_H,
                l2_reg_W,
                l2_reg_H,
                update_H,
                self.verbose,
                None,
            )
        else:
            raise ValueError("Invalid solver parameter '%s'." % self.solver)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        return W, H, n_iter

    def fit(self, X, y=None, **params):
        """Learn a NMF model for the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        Returns
        -------
        self
        """
        self.fit_transform(X, **params)
        return self

    def transform(self, X):
        """Transform the data X according to the fitted NMF model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be transformed by the model.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32], reset=False
        )

        with config_context(assume_finite=True):
            W, *_ = self._fit_transform(X, H=self.components_, update_H=False)

        return W

    def inverse_transform(self, W):
        """Transform data back to its original space.

        Parameters
        ----------
        W : {ndarray, sparse matrix} of shape (n_samples, n_components)
            Transformed data matrix.

        Returns
        -------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data matrix of original shape.

        .. versionadded:: 0.18
        """
        check_is_fitted(self)
        return np.dot(W, self.components_)


class MiniBatchNMF(NMF):
    """Mini-Batch and online Non-Negative Matrix Factorization (NMF)

    .. versionadded:: 1.0

    Find two non-negative matrices (W, H) whose product approximates the non-
    negative matrix X. This factorization can be used for example for
    dimensionality reduction, source separation or topic extraction.

    The objective function is:

        .. math::

            0.5 * ||X - WH||_{loss}^2 + alpha * l1_{ratio} * ||vec(W)||_1

            + alpha * l1_{ratio} * ||vec(H)||_1

            + 0.5 * alpha * (1 - l1_{ratio}) * ||W||_{Fro}^2

            + 0.5 * alpha * (1 - l1_{ratio}) * ||H||_{Fro}^2

    Where:

    :math:`||A||_{Fro}^2 = \\sum_{i,j} A_{ij}^2` (Frobenius norm)

    :math:`||vec(A)||_1 = \\sum_{i,j} abs(A_{ij})` (Elementwise L1 norm)

    The generic norm :math:`||X - WH||_{loss}^2` may represent
    the Frobenius norm or another supported beta-divergence loss.
    The choice between options is controlled by the `beta_loss` parameter.

    The objective function is minimized with an alternating minimization of W
    and H.

    Read more in the :ref:`User Guide <MiniBatchNMF>`.

    Parameters
    ----------
    n_components : int or None
        Number of components, if n_components is not set all features
        are kept.

    init : {'random', 'nndsvd', 'nndsvda', 'nndsvdar', 'custom'}, default=None
        Method used to initialize the procedure.
        Default: None.
        Valid options:

        - `None`: 'nndsvd' if n_components <= min(n_samples, n_features),
          otherwise random.

        - `'random'`: non-negative random matrices, scaled with:
          sqrt(X.mean() / n_components)

        - `'nndsvd'`: Nonnegative Double Singular Value Decomposition (NNDSVD)
          initialization (better for sparseness)

        - `'nndsvda'`: NNDSVD with zeros filled with the average of X
          (better when sparsity is not desired)

        - `'nndsvdar'` NNDSVD with zeros filled with small random values
          (generally faster, less accurate alternative to NNDSVDa
          for when sparsity is not desired)

        - `'custom'`: use custom matrices W and H

    batch_size : int, default=1024
        Number of samples in each mini-batch. Large batch sizes
        give better long-term convergence at the cost of a slower start.

    solver : 'mu'
        Numerical solver to use:
        'mu' is a Multiplicative Update solver.
        For now, this is the only available solver in the
        MiniBatch implementation.

    beta_loss : float or {'frobenius', 'kullback-leibler', \
            'itakura-saito'}, default='frobenius'
        Beta divergence to be minimized, measuring the distance between X
        and the dot product WH. Note that values different from 'frobenius'
        (or 2) and 'kullback-leibler' (or 1) lead to significantly slower
        fits. Note that for beta_loss <= 0 (or 'itakura-saito'), the input
        matrix X cannot contain zeros.

    tol : float, default: 1e-4
        Tolerance of the stopping condition.

    max_iter : integer, default: 200
        Maximum number of iterations over the complete dataset before
        timing out.

    random_state : int, RandomState instance, default=None
        Used for initialisation (when ``init`` == 'nndsvdar' or
        'random'), and in Coordinate Descent. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    alpha : double, default: 0.
        Constant that multiplies the regularization terms. Set it to zero to
        have no regularization.

    l1_ratio : double, default: 0.
        The regularization mixing parameter, with 0 <= l1_ratio <= 1.
        For l1_ratio = 0 the penalty is an elementwise L2 penalty
        (aka Frobenius Norm).
        For l1_ratio = 1 it is an elementwise L1 penalty.
        For 0 < l1_ratio < 1, the penalty is a combination of L1 and L2.

    verbose : bool, default=False
        Whether to be verbose.

    forget_factor : float, default=0.7.
        Amount of rescaling of past information. Its value could be =1 with
        finite datasets. Choosing values <1 is recommended with online
        learning as more recent batches will weight more than past batches.

    Attributes
    ----------
    components_ : array, [n_components, n_features]
        Factorization matrix, sometimes called 'dictionary'.

    n_components_ : integer
        The number of components. It is same as the `n_components` parameter
        if it was given. Otherwise, it will be same as the number of
        features.

    reconstruction_err_ : number
        Frobenius norm of the matrix difference, or beta-divergence, between
        the training data ``X`` and the reconstructed data ``WH`` from
        the fitted model.

    n_iter_ : int
        Actual number of iterations.

    iter_offset_ : int
        The number of iteration on data batches that has been
        performed.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import MiniBatchNMF
    >>> model = MiniBatchNMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_

    References
    ----------
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. "Fast local algorithms for
    large scale nonnegative matrix and tensor factorizations."
    IEICE transactions on fundamentals of electronics, communications and
    computer sciences 92.3: 708-721, 2009.

    Fevotte, C., & Idier, J. (2011). Algorithms for nonnegative matrix
    factorization with the beta-divergence. Neural Computation, 23(9).

    Lefevre, A., Bach, F., Fevotte, C. (2011). Online algorithms for
    nonnegative matrix factorization with the Itakura-Saito divergence.
    WASPA (https://doi.org/10.1109/ASPAA.2011.6082314,
    https://hal.archives-ouvertes.fr/hal-00602050)
    """

    def __init__(
        self,
        n_components=None,
        *,
        init=None,
        solver="mu",
        batch_size=1024,
        beta_loss="frobenius",
        tol=1e-4,
        max_iter=200,
        random_state=None,
        alpha=0.0,
        l1_ratio=0.0,
        verbose=0,
        regularization="both",
        forget_factor=0.7,
    ):

        super().__init__(
            n_components=n_components,
            init=init,
            solver=solver,
            beta_loss=beta_loss,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
            alpha=alpha,
            l1_ratio=l1_ratio,
            verbose=verbose,
            shuffle=False,
            regularization=regularization,
        )

        self.batch_size = batch_size
        self.forget_factor = forget_factor

    def _check_params(self, X):
        super()._check_params(X)

        # solver
        if not isinstance(self.solver, str) or self.solver != 'mu':
            raise ValueError(f"Invalid solver parameter '{self.solver}'. "
                             f"Only solver='mu' is accepted.")

        # batch_size
        self._batch_size = self.batch_size
        if not isinstance(
            self._batch_size, numbers.Integral
        ) or self._batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got "
                             f"{self._batch_size!r} instead.")
        self._batch_size = min(self._batch_size, X.shape[0])

        # forget_factor
        # TODO
        self._rho = self.forget_factor ** (self._batch_size / X.shape[0])

        # gamma for Maximization-Minimization (MM) algorithm [Fevotte 2011]
        if self._beta_loss < 1:
            self._gamma = 1. / (2. - self._beta_loss)
        elif self._beta_loss > 2:
            self._gamma = 1. / (self._beta_loss - 1.)
        else:
            self._gamma = 1.

        return self

    def fit_transform(self, X, y=None, W=None, H=None):
        """Learn a NMF model for the data X and returns the transformed data.

        This is more efficient than calling fit followed by transform.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like, shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like, shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.

        Returns
        -------
        W : array, shape (n_samples, n_components)
            Transformed data.
        """
        X = self._validate_data(
            X, accept_sparse=("csr", "csc"), dtype=[np.float64, np.float32]
        )

        with config_context(assume_finite=True):
            W, H, n_iter, n_steps, A, B = self._fit_transform(X, W=W, H=H)

        if n_iter == self.max_iter and self.tol > 0:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning,
            )

        self.reconstruction_err_ = _beta_divergence(
            X, W, H, self._beta_loss, square_root=True
        )

        self.n_components_ = H.shape[0]
        self.components_ = H
        self.n_iter_ = n_iter
        self.n_steps_ = n_steps
        self._components_numerator = A
        self._components_denominator = B

        return W

    def _fit_transform(self, X, y=None, W=None, H=None, update_H=True):
        """Learn a NMF model for the data X and returns the transformed data.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Data matrix to be decomposed

        y : Ignored

        W : array-like of shape (n_samples, n_components)
            If init='custom', it is used as initial guess for the solution.

        H : array-like of shape (n_components, n_features)
            If init='custom', it is used as initial guess for the solution.
            If update_H=False, it is used as a constant, to solve for W only.

        update_H : bool, default=True
            If True, both W and H will be estimated from initial guesses,
            this corresponds to a call to the 'fit_transform' method.
            If False, only W will be estimated, this corresponds to a call
            to the 'transform' method.

        Returns
        -------
        W : ndarray of shape (n_samples, n_components)
            Transformed data.

        H : ndarray of shape (n_components, n_features)
            Factorization matrix, sometimes called 'dictionary'.

        n_iter_ : int
            Actual number of iterations.

        iter_offset : int, default=0
            Number of previous iterations completed used for
            initialization, only used in
            :class:`sklearn.decomposition.MiniBatchNMF`.

        A : array-like of shape (n_components, n_features)
            Initial guess for the numerator auxiliary function

        B : array-like of shape (n_components, n_features)
            Initial guess for the denominator auxiliary function
        """
        check_non_negative(X, "NMF (input X)")
        self._check_params(X)

        if X.min() == 0 and self._beta_loss <= 0:
            raise ValueError(
                "When beta_loss <= 0 and X contains zeros, "
                "the solver may diverge. Please add small values "
                "to X, or use a positive beta_loss."
            )

        n_samples, n_features = X.shape
        # initialize or check W and H
        W, H = self._check_w_h(X, W, H, update_H)

        l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
            self.alpha, self.l1_ratio, self.regularization
        )

        # Initialize auxiliary matrices
        A = H.copy()
        B = np.ones(H.shape, dtype=H.dtype)

        batches = gen_batches(n_samples, self._batch_size)
        batches = itertools.cycle(batches)
        n_steps_per_epoch = int(np.ceil(n_samples / self._batch_size))
        n_steps = self.max_iter * n_steps_per_epoch

        for i, batch in zip(range(n_steps), batches):
            # update W
            delta_W, H_sum, HHt, XHt = _multiplicative_update_w(
                X[batch], W[batch], H, self._beta_loss, l1_reg_W, l2_reg_W,
                self._gamma, update_H=update_H)
            W[batch] *= delta_W

            # necessary for stability with beta_loss < 1
            if self._beta_loss < 1:
                W[batch][W[batch] < np.finfo(np.float64).eps] = 0.

            # update H
            if update_H:
                H, A, B = _multiplicative_update_h(
                    X[batch], W[batch], H, A, B, self._beta_loss,
                    l1_reg_H, l2_reg_H, self._gamma, self._rho)

                # necessary for stability with beta_loss < 1
                if self._beta_loss <= 1:
                    H[H < np.finfo(np.float64).eps] = 0.

        n_steps = i + 1
        n_iter = int(np.ceil((i + 1) / n_steps_per_epoch))

        return W, H, n_iter, n_steps, A, B

    def partial_fit(self, X, y=None, **params):
        has_components = hasattr(self, "components_")

        if has_components:
            with config_context(assume_finite=True):
                X = self._validate_data(
                    X,
                    accept_sparse=("csr", "csc"),
                    dtype=[np.float64, np.float32],
                    reset=False,
                )
                # initialize W and H
                H = self.components_
                W = None
                # Compute W given H and X using transform
                W, *_ = self._fit_transform(X, H=H, update_H=False)

                # Add 1 iteration to the current estimation
                l1_reg_W, l1_reg_H, l2_reg_W, l2_reg_H = _compute_regularization(
                    self.alpha, self.l1_ratio, self.regularization
                )

                W, H, n_iter, iter_offset, A, B = _fit_multiplicative_update(
                    X,
                    W,
                    self.components_,
                    self._components_numerator,
                    self._components_denominator,
                    self._beta_loss,
                    self._batch_size,
                    self.iter_offset_,
                    1,
                    self.tol,
                    l1_reg_W,
                    l1_reg_H,
                    l2_reg_W,
                    l2_reg_H,
                    True,
                    self.verbose,
                    self.forget_factor,
                )

            self.n_components_ = H.shape[0]
            self.components_ = H
            self.n_iter_ += n_iter
            self.iter_offset_ += iter_offset
            self._components_numerator = A
            self._components_denominator = B

        else:
            self.fit_transform(X, **params)

        return self
