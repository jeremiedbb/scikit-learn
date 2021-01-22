""" Dictionary learning.
"""
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time
import sys
import itertools
import warnings

from math import ceil

import numpy as np
from scipy import linalg
from joblib import Parallel, effective_n_jobs

from ..base import BaseEstimator, TransformerMixin
from ..utils import deprecated
from ..utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches)
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted, _deprecate_positional_args
from ..utils.fixes import delayed
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars


def _check_positive_coding(method, positive):
    if positive and method in ["omp", "lars"]:
        raise ValueError(
                "Positive constraint not supported for '{}' "
                "coding method.".format(method)
            )


def _sparse_encode(X, dictionary, gram, cov=None, algorithm='lasso_lars',
                   regularization=None, copy_cov=True,
                   init=None, max_iter=1000, check_input=True, verbose=0,
                   positive=False):
    """Generic sparse coding.

    Each column of the result is the solution to a Lasso problem.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows.

    gram : ndarray of shape (n_components, n_components) or None
        Precomputed Gram matrix, `dictionary * dictionary'`
        gram can be `None` if method is 'threshold'.

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary * X'`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    regularization : int or float, default=None
        The regularization parameter. It corresponds to alpha when
        algorithm is `'lasso_lars'`, `'lasso_cd'` or `'threshold'`.
        Otherwise it corresponds to `n_nonzero_coefs`.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse code. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    check_input : bool, default=True
        If `False`, the input arrays `X` and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive: bool, default=False
        Whether to enforce a positivity constraint on the sparse code.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_components, n_features)
        The sparse codes.

    See Also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if X.ndim == 1:
        X = X[:, np.newaxis]
    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]
    if dictionary.shape[1] != X.shape[1]:
        raise ValueError("Dictionary and X have different numbers of features:"
                         "dictionary.shape: {} X.shape{}".format(
                             dictionary.shape, X.shape))
    if cov is None and algorithm != 'lasso_cd':
        # overwriting cov is safe
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    _check_positive_coding(algorithm, positive)

    if algorithm == 'lasso_lars':
        alpha = float(regularization) / n_features  # account for scaling
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lasso_lars = LassoLars(alpha=alpha, fit_intercept=False,
                                   verbose=verbose, normalize=False,
                                   precompute=gram, fit_path=False,
                                   positive=positive, max_iter=max_iter)
            lasso_lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lasso_lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'lasso_cd':
        alpha = float(regularization) / n_features  # account for scaling

        # TODO: Make verbosity argument for Lasso?
        # sklearn.linear_model.coordinate_descent.enet_path has a verbosity
        # argument that we could pass in from Lasso.
        clf = Lasso(alpha=alpha, fit_intercept=False, normalize=False,
                    precompute=gram, max_iter=max_iter, warm_start=True,
                    positive=positive)

        if init is not None:
            clf.coef_ = init

        clf.fit(dictionary.T, X.T, check_input=check_input)
        new_code = clf.coef_

    elif algorithm == 'lars':
        try:
            err_mgt = np.seterr(all='ignore')

            # Not passing in verbose=max(0, verbose-1) because Lars.fit already
            # corrects the verbosity level.
            lars = Lars(fit_intercept=False, verbose=verbose, normalize=False,
                        precompute=gram, n_nonzero_coefs=int(regularization),
                        fit_path=False)
            lars.fit(dictionary.T, X.T, Xy=cov)
            new_code = lars.coef_
        finally:
            np.seterr(**err_mgt)

    elif algorithm == 'threshold':
        new_code = ((np.sign(cov) *
                    np.maximum(np.abs(cov) - regularization, 0)).T)
        if positive:
            np.clip(new_code, 0, None, out=new_code)

    elif algorithm == 'omp':
        new_code = orthogonal_mp_gram(
            Gram=gram, Xy=cov, n_nonzero_coefs=int(regularization),
            tol=None, norms_squared=row_norms(X, squared=True),
            copy_Xy=copy_cov).T
    else:
        raise ValueError('Sparse coding method must be "lasso_lars" '
                         '"lasso_cd", "lasso", "threshold" or "omp", got %s.'
                         % algorithm)
    if new_code.ndim != 2:
        return new_code.reshape(n_samples, n_components)
    return new_code


# XXX : could be moved to the linear_model module
@_deprecate_positional_args
def sparse_encode(X, dictionary, *, gram=None, cov=None,
                  algorithm='lasso_lars', n_nonzero_coefs=None, alpha=None,
                  copy_cov=True, init=None, max_iter=1000, n_jobs=None,
                  check_input=True, verbose=0, positive=False):
    """Sparse coding

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    dictionary : ndarray of shape (n_components, n_features)
        The dictionary matrix against which to solve the sparse coding of
        the data. Some of the algorithms assume normalized rows for meaningful
        output.

    gram : ndarray of shape (n_components, n_components), default=None
        Precomputed Gram matrix, `dictionary * dictionary'`.

    cov : ndarray of shape (n_components, n_samples), default=None
        Precomputed covariance, `dictionary' * X`.

    algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'}, \
            default='lasso_lars'
        The algorithm used:

        * `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        * `'lasso_lars'`: uses Lars to compute the Lasso solution;
        * `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). lasso_lars will be faster if
          the estimated components are sparse;
        * `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        * `'threshold'`: squashes to zero all coefficients less than
          regularization from the projection `dictionary * data'`.

    n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `n_nonzero_coefs=int(n_features / 10)`.

    alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    copy_cov : bool, default=True
        Whether to copy the precomputed covariance matrix; if `False`, it may
        be overwritten.

    init : ndarray of shape (n_samples, n_components), default=None
        Initialization value of the sparse codes. Only used if
        `algorithm='lasso_cd'`.

    max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    check_input : bool, default=True
        If `False`, the input arrays X and dictionary will not be checked.

    verbose : int, default=0
        Controls the verbosity; the higher, the more messages.

    positive : bool, default=False
        Whether to enforce positivity when finding the encoding.

        .. versionadded:: 0.20

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse codes

    See Also
    --------
    sklearn.linear_model.lars_path
    sklearn.linear_model.orthogonal_mp
    sklearn.linear_model.Lasso
    SparseCoder
    """
    if check_input:
        if algorithm == 'lasso_cd':
            dictionary = check_array(dictionary, order='C', dtype='float64')
            X = check_array(X, order='C', dtype='float64')
        else:
            dictionary = check_array(dictionary)
            X = check_array(X)

    n_samples, n_features = X.shape
    n_components = dictionary.shape[0]

    if gram is None and algorithm != 'threshold':
        gram = np.dot(dictionary, dictionary.T)

    if cov is None and algorithm != 'lasso_cd':
        copy_cov = False
        cov = np.dot(dictionary, X.T)

    if algorithm in ('lars', 'omp'):
        regularization = n_nonzero_coefs
        if regularization is None:
            regularization = min(max(n_features / 10, 1), n_components)
    else:
        regularization = alpha
        if regularization is None:
            regularization = 1.

    if effective_n_jobs(n_jobs) == 1 or algorithm == 'threshold':
        code = _sparse_encode(X,
                              dictionary, gram, cov=cov,
                              algorithm=algorithm,
                              regularization=regularization, copy_cov=copy_cov,
                              init=init,
                              max_iter=max_iter,
                              check_input=False,
                              verbose=verbose,
                              positive=positive)
        return code

    # Enter parallel code block
    code = np.empty((n_samples, n_components))
    slices = list(gen_even_slices(n_samples, effective_n_jobs(n_jobs)))

    code_views = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_sparse_encode)(
            X[this_slice], dictionary, gram,
            cov[:, this_slice] if cov is not None else None,
            algorithm,
            regularization=regularization, copy_cov=copy_cov,
            init=init[this_slice] if init is not None else None,
            max_iter=max_iter,
            check_input=False,
            verbose=verbose,
            positive=positive)
        for this_slice in slices)
    for this_slice, this_view in zip(slices, code_views):
        code[this_slice] = this_view
    return code


def _update_dict(dictionary, Y, code, verbose=False, return_r2=False,
                 random_state=None, positive=False):
    """Update the dense dictionary factor in place.

    Parameters
    ----------
    dictionary : ndarray of shape (n_features, n_components)
        Value of the dictionary at the previous iteration.

    Y : ndarray of shape (n_features, n_samples)
        Data matrix.

    code : ndarray of shape (n_components, n_samples)
        Sparse coding of the data against which to optimize the dictionary.

    verbose: bool, default=False
        Degree of output the procedure will print.

    return_r2 : bool, default=False
        Whether to compute and return the residual sum of squares corresponding
        to the computed solution.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    Returns
    -------
    dictionary : ndarray of shape (n_features, n_components)
        Updated dictionary.
    """
    n_components = len(code)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)

    used_atoms = np.ones(n_components, dtype=bool)

    # Get BLAS functions
    gemm, = linalg.get_blas_funcs(('gemm',), (dictionary, code, Y))
    ger, = linalg.get_blas_funcs(('ger',), (dictionary, code))
    nrm2, = linalg.get_blas_funcs(('nrm2',), (dictionary,))
    # Residuals, computed with BLAS for speed and efficiency
    # R <- -1.0 * U * V^T + 1.0 * Y
    # Outputs R as Fortran array for efficiency
    R = gemm(-1.0, dictionary, code, 1.0, Y)
    for k in range(n_components):
        # R <- 1.0 * U_k * V_k^T + R
        R = ger(1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] = R[:, k]
        #dictionary[:, k] = np.dot(R, code[k, :])
        if positive:
            np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
        # Scale k'th atom
        # (U_k * U_k) ** 0.5
        atom_norm = nrm2(dictionary[:, k])
        # if atom_norm < 1e-10:
        #     if verbose == 1:
        #         sys.stdout.write("+")
        #         sys.stdout.flush()
        #     elif verbose:
        #         print("Adding new random atom")
        #     dictionary[:, k] = random_state.randn(n_features)
        #     if positive:
        #         np.clip(dictionary[:, k], 0, None, out=dictionary[:, k])
        #     # Setting corresponding coefs to 0
        #     code[k, :] = 0.0
        #     # (U_k * U_k) ** 0.5
        #     atom_norm = nrm2(dictionary[:, k])
        #     dictionary[:, k] /= atom_norm
        #     used_atoms[k] = False
        # else:
        #     dictionary[:, k] /= atom_norm
        #     # R <- -1.0 * U_k * V_k^T + R
        #     R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
        dictionary[:, k] /= atom_norm
        # R <- -1.0 * U_k * V_k^T + R
        R = ger(-1.0, dictionary[:, k], code[k, :], a=R, overwrite_a=True)
    if return_r2:
        R = nrm2(R) ** 2.0
        return dictionary, R
    return dictionary, used_atoms


def _update_dict2(dictionary, Y, code, verbose=False, return_r2=False,
                 random_state=None, positive=False):
    n_components = len(code)
    n_features = Y.shape[0]
    random_state = check_random_state(random_state)

    B, A = Y, code

    used_atoms = np.ones(n_components, dtype=bool)

    for j in range(n_components):
        Ajjuj = B[:, j] - dictionary.dot(A[j]) + A[j, j] * dictionary[:,j]
        dictionary[:, j] = Ajjuj / np.sqrt(np.sum(Ajjuj**2))
        # if A[j, j] != 0:
        #     dictionary[:, j] += (B[:, j] - dictionary.dot(A[j])) / A[j, j]
        #     dictionary[:, j] /= max(np.sqrt((dictionary[:, j]**2).sum()), 1)

    return dictionary, used_atoms


@_deprecate_positional_args
def dict_learning(X, n_components, *, alpha, max_iter=100, tol=1e-8,
                  method='lars', n_jobs=None, dict_init=None, code_init=None,
                  callback=None, verbose=False, random_state=None,
                  return_n_iter=False, positive_dict=False,
                  positive_code=False, method_max_iter=1000):
    """Solves a dictionary learning matrix factorization problem.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    n_components : int
        Number of dictionary atoms to extract.

    alpha : int
        Sparsity controlling parameter.

    max_iter : int, default=100
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for the stopping condition.

    method : {'lars', 'cd'}, default='lars'
        The method used:

        * `'lars'`: uses the least angle regression method to solve the lasso
           problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the sparse code for warm restart scenarios.

    callback : callable, default=None
        Callable that gets invoked every five iterations

    verbose : bool, default=False
        To control the verbosity of the procedure.

    random_state : int, RandomState instance or None, default=None
        Used for randomly initializing the dictionary. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform.

        .. versionadded:: 0.22

    Returns
    -------
    code : ndarray of shape (n_samples, n_components)
        The sparse code factor in the matrix factorization.

    dictionary : ndarray of shape (n_components, n_features),
        The dictionary factor in the matrix factorization.

    errors : array
        Vector of errors at each iteration.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to True.

    See Also
    --------
    dict_learning_online
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    if method not in ('lars', 'cd'):
        raise ValueError('Coding method %r not supported as a fit algorithm.'
                         % method)

    _check_positive_coding(method, positive_code)

    method = 'lasso_' + method

    t0 = time.time()
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init the code and the dictionary with SVD of Y
    if code_init is not None and dict_init is not None:
        code = np.array(code_init, order='F')
        # Don't copy V, it will happen below
        dictionary = dict_init
    else:
        code, S, dictionary = linalg.svd(X, full_matrices=False)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    # Fortran-order dict, as we are going to access its row vectors
    dictionary = np.array(dictionary, order='F')

    residuals = 0

    errors = []
    current_cost = np.nan

    if verbose == 1:
        print('[dict_learning]', end=' ')

    # If max_iter is 0, number of iterations returned should be zero
    ii = -1

    for ii in range(max_iter):
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            print("Iteration % 3i "
                  "(elapsed time: % 3is, % 4.1fmn, current cost % 7.3f)"
                  % (ii, dt, dt / 60, current_cost))

        # Update code
        code = sparse_encode(X, dictionary, algorithm=method, alpha=alpha,
                             init=code, n_jobs=n_jobs, positive=positive_code,
                             max_iter=method_max_iter, verbose=verbose)
        # Update dictionary
        dictionary, residuals = _update_dict(dictionary.T, X.T, code.T,
                                             verbose=verbose, return_r2=True,
                                             random_state=random_state,
                                             positive=positive_dict)
        dictionary = dictionary.T

        # Cost function
        current_cost = 0.5 * residuals + alpha * np.sum(np.abs(code))
        errors.append(current_cost)

        if ii > 0:
            dE = errors[-2] - errors[-1]
            # assert(dE >= -tol * errors[-1])
            if dE < tol * errors[-1]:
                if verbose == 1:
                    # A line return
                    print("")
                elif verbose:
                    print("--- Convergence reached after %d iterations" % ii)
                break
        if ii % 5 == 0 and callback is not None:
            callback(locals())

    if return_n_iter:
        return code, dictionary, errors, ii + 1
    else:
        return code, dictionary, errors


@_deprecate_positional_args
def dict_learning_online(X, n_components=2, *, alpha=1, n_iter="deprecated",
                         max_iter=None, return_code=True, dict_init=None,
                         callback=None, batch_size=3, verbose=False,
                         shuffle=True, n_jobs=None, method='lars',
                         iter_offset="deprecated", random_state=None,
                         return_inner_stats="deprecated",
                         inner_stats="deprecated", return_n_iter="deprecated",
                         positive_dict=False, positive_code=False,
                         method_max_iter=1000, max_no_improvement=10):
    """Solves a dictionary learning matrix factorization problem online.

    Finds the best dictionary and the corresponding sparse code for
    approximating the data matrix X by solving::

        (U^*, V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                     (U,V)
                     with || V_k ||_2 = 1 for all  0 <= k < n_components

    where V is the dictionary and U is the sparse code. This is
    accomplished by repeatedly iterating over mini-batches by slicing
    the input data.

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Data matrix.

    n_components : int, default=2
        Number of dictionary atoms to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    n_iter : int, default=100
        Number of mini-batch iterations to perform.

        .. deprecated:: 1.0
           ``n_iter`` is deprecated in 1.0 and will be removed in 1.2. Use
           ``max_iter`` instead.

    max_iter : int, default=None
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
        If ``max_iter`` is not None, ``n_iter`` is ignored.

        .. versionadded:: 1.0

    return_code : bool, default=True
        Whether to also return the code U or just the dictionary `V`.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial value for the dictionary for warm restart scenarios.

    callback : callable, default=None
        callable that gets invoked at the end of each iteration.

    batch_size : int, default=3
        The number of samples to take in each batch.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    shuffle : bool, default=True
        Whether to shuffle the data before splitting it in batches.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    method : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    iter_offset : int, default=0
        Number of previous iterations completed on the dictionary used for
        initialization.

        .. deprecated:: 1.0
           ``iter_offset`` serves internal purpose only and will be removed
           in 1.2.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    return_inner_stats : bool, default=False
        Return the inner statistics A (dictionary covariance) and B
        (data approximation). Useful to restart the algorithm in an
        online setting. If `return_inner_stats` is `True`, `return_code` is
        ignored.

        .. deprecated:: 1.0
           ``return_inner_stats`` serves internal purpose only and will be
           removed in 1.2.

    inner_stats : tuple of (A, B) ndarrays, default=None
        Inner sufficient statistics that are kept by the algorithm.
        Passing them at initialization is useful in online settings, to
        avoid losing the history of the evolution.
        `A` `(n_components, n_components)` is the dictionary covariance matrix.
        `B` `(n_features, n_components)` is the data approximation matrix.

        .. deprecated:: 1.0
           ``inner_stats`` serves internal purpose only and will be removed
           in 1.2.

    return_n_iter : bool, default=False
        Whether or not to return the number of iterations.

        .. deprecated:: 1.0
           ``return_n_iter`` will be removed in 1.2 and n_iter will always be
           returned.

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    method_max_iter : int, default=1000
        Maximum number of iterations to perform when solving the lasso problem.

        .. versionadded:: 0.22

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function. To
        disable early stopping set `max_no_improvement` to None. Only used if
        `max_iter` is not None.

        .. versionadded:: 1.0

    Returns
    -------
    code : ndarray of shape (n_samples, n_components),
        The sparse code (only returned if `return_code=True`).

    dictionary : ndarray of shape (n_components, n_features),
        The solutions to the dictionary learning problem.

    n_iter : int
        Number of iterations run. Returned only if `return_n_iter` is
        set to `True`.

    n_steps : int
        The number of iteration on data batches that has been
        performed before. Returned only if `max_iter` is not None.

    See Also
    --------
    dict_learning
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    deps = (return_n_iter, return_inner_stats, iter_offset, inner_stats)
    if max_iter is not None and not all(arg == "deprecated" for arg in deps):
        raise ValueError(
            "the following args are incompatible with 'max_iter': "
            "return_n_iter, return_inner_stats, iter_offset, inner_stats")

    if iter_offset != "deprecated":
        warnings.warn("'iter_offset' is deprecated in version 1.0 and "
                      "will be removed in version 1.2.", FutureWarning)
    else:
        iter_offset = 0

    if return_inner_stats != "deprecated":
        warnings.warn("'return_inner_stats' is deprecated in version 1.0 and "
                      "will be removed in version 1.2.", FutureWarning)
    else:
        return_inner_stats = False

    if inner_stats != "deprecated":
        warnings.warn("'inner_stats' is deprecated in version 1.0 and "
                      "will be removed in version 1.2.", FutureWarning)
    else:
        inner_stats = None

    if return_n_iter != "deprecated":
        warnings.warn("'return_n_iter' is deprecated in version 1.0 and "
                      "will be removed in version 1.2. From version 1.2 n_iter"
                      " will always be returned", FutureWarning)
    else:
        return_n_iter = False

    if max_iter is not None:
        # TODO: split method into fit & transform ?
        transform_algorithm = "lasso_" + method

        # TODO: add split_sign & method_n_nonzero_coef ?
        est = MiniBatchDictionaryLearning(
            n_components=n_components, alpha=alpha, n_iter=n_iter,
            n_jobs=n_jobs, fit_algorithm=method, batch_size=batch_size,
            shuffle=shuffle, dict_init=dict_init, random_state=random_state,
            transform_algorithm=transform_algorithm, transform_alpha=alpha,
            positive_code=positive_code, positive_dict=positive_dict,
            transform_max_iter=method_max_iter, verbose=verbose,
            callback=callback, max_no_improvement=max_no_improvement).fit(X)

        if not return_code:
            return est.components_, est.n_iter_, est.n_steps_
        else:
            code = est.transform(X)
            return code, est.components_, est.n_iter_, est.n_steps_

    # TODO remove the whole old behavior in 1.2
    # Fallback to old behavior

    if n_iter != "deprecated":
        warnings.warn(
            "'n_iter' is deprecated in version 1.0 and will be removed"
            " in version 1.2. Use 'max_iter' instead.", FutureWarning)
    else:
        n_iter = 100

    if n_components is None:
        n_components = X.shape[1]

    if method not in ('lars', 'cd'):
        raise ValueError('Coding method not supported as a fit algorithm.')

    _check_positive_coding(method, positive_code)

    method = 'lasso_' + method

    t0 = time.time()
    n_samples, n_features = X.shape
    # Avoid integer division problems
    alpha = float(alpha)
    random_state = check_random_state(random_state)

    # Init V with SVD of X
    if dict_init is not None:
        dictionary = dict_init
    else:
        _, S, dictionary = randomized_svd(X, n_components,
                                          random_state=random_state)
        dictionary = S[:, np.newaxis] * dictionary
    r = len(dictionary)
    if n_components <= r:
        dictionary = dictionary[:n_components, :]
    else:
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    if verbose == 1:
        print('[dict_learning]', end=' ')

    if shuffle:
        X_train = X.copy()
        random_state.shuffle(X_train)
    else:
        X_train = X

    dictionary = check_array(dictionary.T, order='F', dtype=np.float64,
                             copy=False)
    dictionary = np.require(dictionary, requirements='W')

    X_train = check_array(X_train, order='C', dtype=np.float64, copy=False)

    batches = gen_batches(n_samples, batch_size)
    batches = itertools.cycle(batches)

    # The covariance of the dictionary
    if inner_stats is None:
        A = np.zeros((n_components, n_components))
        # The data approximation
        B = np.zeros((n_features, n_components))
    else:
        A = inner_stats[0].copy()
        B = inner_stats[1].copy()

    # If n_iter is zero, we need to return zero.
    ii = iter_offset - 1

    for ii, batch in zip(range(iter_offset, iter_offset + n_iter), batches):
        this_X = X_train[batch]
        dt = (time.time() - t0)
        if verbose == 1:
            sys.stdout.write(".")
            sys.stdout.flush()
        elif verbose:
            if verbose > 10 or ii % ceil(100. / verbose) == 0:
                print("Iteration % 3i (elapsed time: % 3is, % 4.1fmn)"
                      % (ii, dt, dt / 60))

        this_code = sparse_encode(this_X, dictionary.T, algorithm=method,
                                  alpha=alpha, n_jobs=n_jobs,
                                  check_input=False,
                                  positive=positive_code,
                                  max_iter=method_max_iter, verbose=verbose).T

        # Update the auxiliary variables
        if ii < batch_size - 1:
            theta = float((ii + 1) * batch_size)
        else:
            theta = float(batch_size ** 2 + ii + 1 - batch_size)
        beta = (theta + 1 - batch_size) / (theta + 1)

        A *= beta
        A += np.dot(this_code, this_code.T)
        B *= beta
        B += np.dot(this_X.T, this_code.T)

        # Update dictionary
        dictionary = _update_dict(dictionary, B, A, verbose=verbose,
                                  random_state=random_state,
                                  positive=positive_dict)
        # XXX: Can the residuals be of any use?

        # Maybe we need a stopping criteria based on the amount of
        # modification in the dictionary
        if callback is not None:
            callback(locals())

    if return_inner_stats:
        if return_n_iter:
            return dictionary.T, (A, B), ii - iter_offset + 1
        else:
            return dictionary.T, (A, B)
    if return_code:
        if verbose > 1:
            print('Learning code...', end=' ')
        elif verbose == 1:
            print('|', end=' ')
        code = sparse_encode(X, dictionary.T, algorithm=method, alpha=alpha,
                             n_jobs=n_jobs, check_input=False,
                             positive=positive_code, max_iter=method_max_iter,
                             verbose=verbose)
        if verbose > 1:
            dt = (time.time() - t0)
            print('done (total time: % 3is, % 4.1fmn)' % (dt, dt / 60))
        if return_n_iter:
            return code, dictionary.T, ii - iter_offset + 1
        else:
            return code, dictionary.T

    if return_n_iter:
        return dictionary.T, ii - iter_offset + 1
    else:
        return dictionary.T


class _BaseSparseCoding(TransformerMixin):
    """Base class from SparseCoder and DictionaryLearning algorithms."""
    def __init__(self, transform_algorithm, transform_n_nonzero_coefs,
                 transform_alpha, split_sign, n_jobs, positive_code,
                 transform_max_iter):
        self.transform_algorithm = transform_algorithm
        self.transform_n_nonzero_coefs = transform_n_nonzero_coefs
        self.transform_alpha = transform_alpha
        self.transform_max_iter = transform_max_iter
        self.split_sign = split_sign
        self.n_jobs = n_jobs
        self.positive_code = positive_code

    def _transform(self, X, dictionary):
        """Private method allowing to accomodate both DictionaryLearning and
        SparseCoder."""
        X = self._validate_data(X, reset=False)

        code = sparse_encode(
            X, dictionary, algorithm=self.transform_algorithm,
            n_nonzero_coefs=self.transform_n_nonzero_coefs,
            alpha=self.transform_alpha, max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs, positive=self.positive_code)

        if self.split_sign:
            # feature vector is split into a positive and negative side
            n_samples, n_features = code.shape
            split_code = np.empty((n_samples, 2 * n_features))
            split_code[:, :n_features] = np.maximum(code, 0)
            split_code[:, n_features:] = -np.minimum(code, 0)
            code = split_code

        return code

    def transform(self, X):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self)
        return self._transform(X, self.components_)


class SparseCoder(_BaseSparseCoding, BaseEstimator):
    """Sparse coding

    Finds a sparse representation of data against a fixed, precomputed
    dictionary.

    Each row of the result is the solution to a sparse coding problem.
    The goal is to find a sparse array `code` such that::

        X ~= code * dictionary

    Read more in the :ref:`User Guide <SparseCoder>`.

    Parameters
    ----------
    dictionary : ndarray of shape (n_components, n_features)
        The dictionary atoms used for sparse coding. Lines are assumed to be
        normalized to unit norm.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution;
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (linear_model.Lasso). `'lasso_lars'` will be faster if
          the estimated components are sparse;
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution;
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `lasso_lars`.

        .. versionadded:: 0.22

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        The unchanged dictionary atoms.

        .. deprecated:: 0.24
           This attribute is deprecated in 0.24 and will be removed in
           1.1 (renaming of 0.26). Use `dictionary` instead.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.decomposition import SparseCoder
    >>> X = np.array([[-1, -1, -1], [0, 0, 3]])
    >>> dictionary = np.array(
    ...     [[0, 1, 0],
    ...      [-1, -1, 2],
    ...      [1, 1, 1],
    ...      [0, 1, 1],
    ...      [0, 2, 1]],
    ...    dtype=np.float64
    ... )
    >>> coder = SparseCoder(
    ...     dictionary=dictionary, transform_algorithm='lasso_lars',
    ...     transform_alpha=1e-10,
    ... )
    >>> coder.transform(X)
    array([[ 0.,  0., -1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.]])

    See Also
    --------
    DictionaryLearning
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    sparse_encode
    """
    _required_parameters = ["dictionary"]

    @_deprecate_positional_args
    def __init__(self, dictionary, *, transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 split_sign=False, n_jobs=None, positive_code=False,
                 transform_max_iter=1000):
        super().__init__(
            transform_algorithm, transform_n_nonzero_coefs,
            transform_alpha, split_sign, n_jobs, positive_code,
            transform_max_iter
        )
        self.dictionary = dictionary

    def fit(self, X, y=None):
        """Do nothing and return the estimator unchanged.

        This method is just there to implement the usual API and hence
        work in pipelines.

        Parameters
        ----------
        X : Ignored

        y : Ignored

        Returns
        -------
        self : object
        """
        return self

    @deprecated("The attribute 'components_' is deprecated "  # type: ignore
                "in 0.24 and will be removed in 1.1 (renaming of 0.26). Use "
                "the 'dictionary' instead.")
    @property
    def components_(self):
        return self.dictionary

    def transform(self, X, y=None):
        """Encode the data as a sparse combination of the dictionary atoms.

        Coding method is determined by the object parameter
        `transform_algorithm`.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test data to be transformed, must have the same number of
            features as the data used to train the model.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        return super()._transform(X, self.dictionary)

    def _more_tags(self):
        return {"requires_fit": False}

    @property
    def n_components_(self):
        return self.dictionary.shape[0]

    @property
    def n_features_in_(self):
        return self.dictionary.shape[1]


class DictionaryLearning(_BaseSparseCoding, BaseEstimator):
    """Dictionary learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Solves the optimization problem::

        (U^*,V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=n_features
        Number of dictionary elements to extract.

    alpha : float, default=1.0
        Sparsity controlling parameter.

    max_iter : int, default=1000
        Maximum number of iterations to perform.

    tol : float, default=1e-8
        Tolerance for numerical error.

    fit_algorithm : {'lars', 'cd'}, default='lars'
        * `'lars'`: uses the least angle regression method to solve the lasso
          problem (:func:`~sklearn.linear_model.lars_path`);
        * `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). Lars will be
          faster if the estimated components are sparse.

        .. versionadded:: 0.17
           *cd* coordinate descent method to improve speed.

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (:func:`~sklearn.linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (:class:`~sklearn.linear_model.Lasso`). `'lasso_lars'`
          will be faster if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

        .. versionadded:: 0.17
           *lasso_cd* coordinate descent method to improve speed.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.0

    n_jobs : int or None, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    code_init : ndarray of shape (n_samples, n_components), default=None
        Initial value for the code, for warm restart.

    dict_init : ndarray of shape (n_components, n_features), default=None
        Initial values for the dictionary, for warm restart.

    verbose : bool, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        dictionary atoms extracted from the data

    error_ : array
        vector of errors at each iteration

    n_iter_ : int
        Number of iterations run.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import DictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=100, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42,
    ... )
    >>> dict_learner = DictionaryLearning(
    ...     n_components=15, transform_algorithm='lasso_lars', random_state=42,
    ... )
    >>> X_transformed = dict_learner.fit_transform(X)

    We can check the level of sparsity of `X_transformed`:

    >>> np.mean(X_transformed == 0)
    0.88...

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.07...

    Notes
    -----
    **References:**

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    See Also
    --------
    SparseCoder
    MiniBatchDictionaryLearning
    SparsePCA
    MiniBatchSparsePCA
    """
    @_deprecate_positional_args
    def __init__(self, n_components=None, *, alpha=1, max_iter=1000, tol=1e-8,
                 fit_algorithm='lars', transform_algorithm='omp',
                 transform_n_nonzero_coefs=None, transform_alpha=None,
                 n_jobs=None, code_init=None, dict_init=None, verbose=False,
                 split_sign=False, random_state=None, positive_code=False,
                 positive_dict=False, transform_max_iter=1000):

        super().__init__(
            transform_algorithm, transform_n_nonzero_coefs,
            transform_alpha, split_sign, n_jobs, positive_code,
            transform_max_iter
        )
        self.n_components = n_components
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.fit_algorithm = fit_algorithm
        self.code_init = code_init
        self.dict_init = dict_init
        self.verbose = verbose
        self.random_state = random_state
        self.positive_dict = positive_dict

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` in the number of samples
            and `n_features` is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the object itself.
        """
        random_state = check_random_state(self.random_state)
        X = self._validate_data(X)
        if self.n_components is None:
            n_components = X.shape[1]
        else:
            n_components = self.n_components

        V, U, E, self.n_iter_ = dict_learning(
            X, n_components, alpha=self.alpha,
            tol=self.tol, max_iter=self.max_iter,
            method=self.fit_algorithm,
            method_max_iter=self.transform_max_iter,
            n_jobs=self.n_jobs,
            code_init=self.code_init,
            dict_init=self.dict_init,
            verbose=self.verbose,
            random_state=random_state,
            return_n_iter=True,
            positive_dict=self.positive_dict,
            positive_code=self.positive_code)
        self.components_ = U
        self.error_ = E
        return self


class MiniBatchDictionaryLearning(_BaseSparseCoding, BaseEstimator):
    """Mini-batch dictionary learning

    Finds a dictionary (a set of atoms) that can best be used to represent data
    using a sparse code.

    Solves the optimization problem::

       (U^*,V^*) = argmin 0.5 || X - U V ||_2^2 + alpha * || U ||_1
                    (U,V)
                    with || V_k ||_2 = 1 for all  0 <= k < n_components

    Read more in the :ref:`User Guide <DictionaryLearning>`.

    Parameters
    ----------
    n_components : int, default=None
        Number of dictionary elements to extract.

    alpha : float, default=1
        Sparsity controlling parameter.

    n_iter : int, default=1000
        Total number of iterations over data batches to perform.

        .. deprecated:: 1.0
           ``n_iter`` is deprecated in 1.0 and will be removed in 1.2. Use
           ``max_iter`` instead.

    max_iter : int, default=None
        Maximum number of iterations over the complete dataset before
        stopping independently of any early stopping criterion heuristics.
        If ``max_iter`` is not None, ``n_iter`` is ignored.

        .. versionadded:: 1.0

    fit_algorithm : {'lars', 'cd'}, default='lars'
        The algorithm used:

        - `'lars'`: uses the least angle regression method to solve the lasso
          problem (`linear_model.lars_path`)
        - `'cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). Lars will be faster if
          the estimated components are sparse.

    n_jobs : int, default=None
        Number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    batch_size : int, default=3
        Number of samples in each mini-batch.

    shuffle : bool, default=True
        Whether to shuffle the samples before forming batches.

    dict_init : ndarray of shape (n_components, n_features), default=None
        initial value of the dictionary for warm restart scenarios

    transform_algorithm : {'lasso_lars', 'lasso_cd', 'lars', 'omp', \
            'threshold'}, default='omp'
        Algorithm used to transform the data:

        - `'lars'`: uses the least angle regression method
          (`linear_model.lars_path`);
        - `'lasso_lars'`: uses Lars to compute the Lasso solution.
        - `'lasso_cd'`: uses the coordinate descent method to compute the
          Lasso solution (`linear_model.Lasso`). `'lasso_lars'` will be faster
          if the estimated components are sparse.
        - `'omp'`: uses orthogonal matching pursuit to estimate the sparse
          solution.
        - `'threshold'`: squashes to zero all coefficients less than alpha from
          the projection ``dictionary * X'``.

    transform_n_nonzero_coefs : int, default=None
        Number of nonzero coefficients to target in each column of the
        solution. This is only used by `algorithm='lars'` and `algorithm='omp'`
        and is overridden by `alpha` in the `omp` case. If `None`, then
        `transform_n_nonzero_coefs=int(n_features / 10)`.

    transform_alpha : float, default=None
        If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the
        penalty applied to the L1 norm.
        If `algorithm='threshold'`, `alpha` is the absolute value of the
        threshold below which coefficients will be squashed to zero.
        If `algorithm='omp'`, `alpha` is the tolerance parameter: the value of
        the reconstruction error targeted. In this case, it overrides
        `n_nonzero_coefs`.
        If `None`, default to 1.

    verbose : bool or int, default=False
        To control the verbosity of the procedure.

    split_sign : bool, default=False
        Whether to split the sparse feature vector into the concatenation of
        its negative part and its positive part. This can improve the
        performance of downstream classifiers.

    random_state : int, RandomState instance or None, default=None
        Used for initializing the dictionary when ``dict_init`` is not
        specified, randomly shuffling the data when ``shuffle`` is set to
        ``True``, and updating the dictionary. Pass an int for reproducible
        results across multiple function calls.
        See :term:`Glossary <random_state>`.

    positive_code : bool, default=False
        Whether to enforce positivity when finding the code.

        .. versionadded:: 0.20

    positive_dict : bool, default=False
        Whether to enforce positivity when finding the dictionary.

        .. versionadded:: 0.20

    transform_max_iter : int, default=1000
        Maximum number of iterations to perform if `algorithm='lasso_cd'` or
        `'lasso_lars'`.

        .. versionadded:: 0.22

    callback : callable, default=None
        callable that gets invoked at the end of each iteration.

        .. versionadded:: 1.0

    max_no_improvement : int, default=10
        Control early stopping based on the consecutive number of mini batches
        that does not yield an improvement on the smoothed cost function. To
        disable early stopping set `max_no_improvement` to None.

        .. versionadded:: 1.0

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Components extracted from the data.

    inner_stats_ : tuple of (A, B) ndarrays
        Internal sufficient statistics that are kept by the algorithm.
        Keeping them is useful in online settings, to avoid losing the
        history of the evolution, but they shouldn't have any use for the
        end user.
        `A` `(n_components, n_components)` is the dictionary covariance matrix.
        `B` `(n_features, n_components)` is the data approximation matrix.

        .. deprecated:: 1.0
           ``inner_stats_`` serves internal purpose only and will be removed
           in 1.2.

    n_iter_ : int
        Number of iterations run.

    iter_offset_ : int
        The number of iteration on data batches that has been
        performed before.

        .. deprecated:: 1.0
           ``iter_offset_`` has been renamed ``n_batches_seen_`` and will be
           removed in 1.2.

    random_state_ : RandomState instance
        RandomState instance that is generated either from a seed, the random
        number generattor or by `np.random`.

        .. deprecated:: 1.0
           ``random_state_`` serves internal purpose only and will be removed
           in 1.2.

    n_steps_ : int
        The number of iteration on data batches that has been
        performed before.

        .. versionadded:: 1.0

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.datasets import make_sparse_coded_signal
    >>> from sklearn.decomposition import MiniBatchDictionaryLearning
    >>> X, dictionary, code = make_sparse_coded_signal(
    ...     n_samples=100, n_components=15, n_features=20, n_nonzero_coefs=10,
    ...     random_state=42)
    >>> dict_learner = MiniBatchDictionaryLearning(
    ...     n_components=15, transform_algorithm='lasso_lars', random_state=42,
    ... )
    >>> X_transformed = dict_learner.fit_transform(X)

    We can check the level of sparsity of `X_transformed`:

    >>> np.mean(X_transformed == 0)
    0.85...

    We can compare the average squared euclidean norm of the reconstruction
    error of the sparse coded signal relative to the squared euclidean norm of
    the original signal:

    >>> X_hat = X_transformed @ dict_learner.components_
    >>> np.mean(np.sum((X_hat - X) ** 2, axis=1) / np.sum(X ** 2, axis=1))
    0.10...

    Notes
    -----
    **References:**

    J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009: Online dictionary learning
    for sparse coding (https://www.di.ens.fr/sierra/pdfs/icml09.pdf)

    See Also
    --------
    SparseCoder
    DictionaryLearning
    SparsePCA
    MiniBatchSparsePCA

    """
    @_deprecate_positional_args
    def __init__(self, n_components=None, *, alpha=1, n_iter="deprecated",
                 max_iter=None, fit_algorithm='lars', n_jobs=None,
                 batch_size=3, shuffle=True, dict_init=None,
                 transform_algorithm='omp', transform_n_nonzero_coefs=None,
                 transform_alpha=None, verbose=False, split_sign=False,
                 random_state=None, positive_code=False, positive_dict=False,
                 transform_max_iter=1000, callback=None,
                 max_no_improvement=10, tol=0):

        super().__init__(
            transform_algorithm, transform_n_nonzero_coefs, transform_alpha,
            split_sign, n_jobs, positive_code, transform_max_iter
        )
        self.n_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.max_iter = max_iter
        self.fit_algorithm = fit_algorithm
        self.dict_init = dict_init
        self.verbose = verbose
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.split_sign = split_sign
        self.random_state = random_state
        self.positive_dict = positive_dict
        self.callback = callback
        self.max_no_improvement = max_no_improvement
        self.tol = tol

    @deprecated("The attribute 'iter_offset_' is deprecated "  # type: ignore
                "in 1.0 and will be removed in 1.2.")
    @property
    def iter_offset_(self):
        return self.n_iter_

    @deprecated("The attribute 'random_state_' is deprecated "  # type: ignore
                "in 1.0 and will be removed in 1.2.")
    @property
    def random_state_(self):
        return self._random_state

    @deprecated("The attribute 'inner_stats_' is deprecated "  # type: ignore
                "in 1.0 and will be removed in 1.2.")
    @property
    def inner_stats_(self):
        return self._inner_stats

    def _check_params(self, X):
        # n_components
        if self.n_components is not None and self.n_components <= 0:
            raise ValueError(
                f"n_components should be > 0, got "
                f"{self.n_components} instead.")
        self._n_components = self.n_components
        if self._n_components is None:
            self._n_components = X.shape[1]

        # fit_algorithm
        if self.fit_algorithm not in ('lars', 'cd'):
            raise ValueError('Coding method not supported as a fit algorithm.')
        _check_positive_coding(self.fit_algorithm, self.positive_code)
        self._fit_algorithm = 'lasso_' + self.fit_algorithm

        # batch_size
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size should be > 0, got {self.batch_size} instead.")
        self._batch_size = min(self.batch_size, X.shape[0])

        # n_iter
        if self.n_iter != "deprecated" and self.n_iter < 0:
            raise ValueError(
                f"n_iter should be >= 0, got {self.n_iter} instead.")

        # max_iter
        if self.max_iter is not None and self.max_iter < 0:
            raise ValueError(
                f"max_iter should be >= 0, got {self.max_iter} instead.")

        # max_no_improvement
        if self.max_no_improvement is not None and self.max_no_improvement < 0:
            raise ValueError(
                f"max_no_improvement should be >= 0, got "
                f"{self.max_no_improvement} instead.")

        # tol
        self._tol = self.tol

        # TODO sparse coding checks

    def _initialize_dict(self, X, random_state):
        """Initialization of the dictionary"""
        if self.dict_init is not None:
            dictionary = self.dict_init
        else:
            # Init V with SVD of X
            _, S, dictionary = randomized_svd(X, self._n_components,
                                              random_state=random_state)
            dictionary = S[:, np.newaxis] * dictionary
        r = len(dictionary)
        if self._n_components <= r:
            dictionary = dictionary[:self._n_components, :]
        else:
            dictionary = np.r_[dictionary,
                               np.zeros((self._n_components - r,
                                         dictionary.shape[1]))]

        dictionary = check_array(dictionary, order='C', dtype=np.float64,
                                 copy=False)
        dictionary = np.require(dictionary, requirements='W')

        return dictionary

    def _minibatch_step(self, X, dictionary, random_state, step):
        """The guts of the algorithm"""
        batch_size = X.shape[0]

        # Compute code for this batch
        code = sparse_encode(
            X, dictionary, algorithm=self._fit_algorithm,
            alpha=self.alpha, n_jobs=self.n_jobs, check_input=False,
            positive=self.positive_code, max_iter=self.transform_max_iter,
            verbose=self.verbose).T

        # Update inner stats
        self._update_inner_stats(X, code, batch_size, step)

        # Update dictionary
        A, B = self._inner_stats

        _, self.used_atoms = _update_dict(dictionary.T, B, A, verbose=self.verbose,
                                          random_state=random_state, positive=self.positive_dict)

        print("D")
        print(dictionary)
        batch_cost = (0.5 * ((X - code.T.dot(dictionary))**2).sum()
                      + self.alpha * np.sum(np.abs(code))) / batch_size

        # from functools import partial
        # se = partial(sparse_encode, dictionary=dictionary, algorithm=self._fit_algorithm,
        #              alpha=self.alpha, n_jobs=self.n_jobs, check_input=False,
        #              positive=self.positive_code, max_iter=self.transform_max_iter,
        #              verbose=self.verbose)

        # if self.callback is not None:
        #     self.callback(cost1=cost1, cost2=cost2, se=se, dictionary=dictionary, alpha=self.alpha)
        #     #self.callback(locals())

        return batch_cost

    def _update_inner_stats(self, X, code, batch_size, step):
        """Update the inner stats inplace"""
        if step < batch_size - 1:
            theta = (step + 1) * batch_size
        else:
            theta = batch_size ** 2 + step + 1 - batch_size
        beta = (theta + 1 - batch_size) / (theta + 1)

        A, B = self._inner_stats
        A *= beta
        A += np.dot(code, code.T)
        B *= beta
        B += np.dot(X.T, code.T)

    def _minibatch_convergence(self, X, batch_cost, dictionary, dict_buffer,
                               n_samples, step):
        """Helper function to encapsulate the early stopping logic"""
        batch_size = X.shape[0]

        # Ignore first iteration
        if step == 0:
            if self.verbose:
                print(f"Minibatch iteration {step}: "
                      f"mean batch cost: {batch_cost}")
            return False

        # Compute an Exponentially Weighted Average of the cost function to
        # monitor the convergence while discarding minibatch-local stochastic
        # variability: https://en.wikipedia.org/wiki/Moving_average
        ewa_cost = self._ewa_cost
        if ewa_cost is None:
            ewa_cost = batch_cost
        else:
            alpha = batch_size / (n_samples + 1)
            alpha = min(alpha, 1)
            ewa_cost = ewa_cost * (1 - alpha) + batch_cost * alpha

        # Log progress to be able to monitor convergence
        if self.verbose:
            print(f"Minibatch iteration {step}: "
                  f"mean batch cost: {batch_cost}, ewa cost: {ewa_cost}")

        max_diff = ((dictionary - dict_buffer)**2).sum()
        max_diff /= (dict_buffer**2).sum()
        if self._tol > 0 and np.sqrt(max_diff) <= self._tol:
            if self.verbose:
                print(f"Converged (small dictionary change) at iteration "
                      f"{step}")
            # return True

        # Early stopping heuristic due to lack of improvement on smoothed
        # cost function
        # ewa_cost_min = self._ewa_cost_min
        # no_improvement = self._no_improvement
        # if ewa_cost_min is None or ewa_cost < ewa_cost_min:
        #     no_improvement = 0
        #     ewa_cost_min = ewa_cost
        # else:
        #     no_improvement += 1

        # if (self.max_no_improvement is not None
        #         and no_improvement >= self.max_no_improvement):
        #     if self.verbose:
        #         print(f"Converged (lack of improvement in cost function) at "
        #               f"iteration {step}")
        #     return True

        # update the convergence context to maintain state across successive
        # calls:
        self._ewa_cost = ewa_cost
        # self._ewa_cost_min = ewa_cost_min
        # self._no_improvement = no_improvement
        return False

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = self._validate_data(X, dtype=np.float64, order='C',
                                copy=self.shuffle)

        self._check_params(X)
        self._random_state = check_random_state(self.random_state)
        n_samples, n_features = X.shape

        dictionary = self._initialize_dict(X, self._random_state)
        dict_buffer = dictionary.copy()

        if self.shuffle:
            X_train = X.copy()
            self._random_state.shuffle(X_train)
        else:
            X_train = X

        if self.verbose:
            print("[dict_learning]")

        # Inner stats
        self._inner_stats = (A, B) = (
            np.zeros((self._n_components, self._n_components)),
            np.zeros((n_features, self._n_components)))

        import time

        if self.max_iter is not None:

            # Attributes to monitor the convergence
            self._ewa_cost = None
            self._ewa_cost_min = None
            self._no_improvement = 0
            self._cost_window = []
            self._n_iter_since_last_min = 0

            self.n_steps_ = 0

            batches = gen_batches(n_samples, self._batch_size)
            batches = itertools.cycle(batches)
            n_steps_per_epoch = int(np.ceil(n_samples / self._batch_size))
            n_iter = self.max_iter * n_steps_per_epoch

            from functools import partial
            se = partial(sparse_encode, algorithm=self._fit_algorithm,
                         alpha=self.alpha, n_jobs=self.n_jobs, check_input=False,
                         positive=self.positive_code, max_iter=self.transform_max_iter,
                         verbose=self.verbose)

            for i, batch in zip(range(n_iter), batches):
                if i % n_steps_per_epoch == 0:
                    self._random_state.shuffle(X_train)
                    print(i / n_steps_per_epoch)

                t = time.time()

                this_X = X_train[batch]

                batch_cost = self._minibatch_step(
                    this_X, dictionary, self._random_state, i)

                if self._minibatch_convergence(this_X, batch_cost, dictionary,
                                               dict_buffer, n_samples, i):
                    break

                max_diff = ((dictionary[self.used_atoms] - dict_buffer[self.used_atoms])**2).sum()
                max_diff /= (dict_buffer[self.used_atoms]**2).sum()
                max_diff = np.sqrt(max_diff)

                dict_buffer[:] = dictionary[:]

                delta_t = time.time() - t

                if self.callback is not None:
                    self.callback(cost=batch_cost, ewa=self._ewa_cost, max_diff=max_diff,
                    t=delta_t, se=se, dictionary=dictionary, alpha=self.alpha)
                    #self.callback(locals())

            self.n_steps_ = i + 1
            self.n_iter_ = self.n_steps_ // n_steps_per_epoch
        else:
            if self.n_iter != "deprecated":
                warnings.warn(
                    "'n_iter' is deprecated in version 1.0 and will be removed"
                    " in version 1.2. Use 'max_iter' instead.", FutureWarning)
                n_iter = self.n_iter
            else:
                n_iter = 1000

            batches = gen_batches(n_samples, self._batch_size)
            batches = itertools.cycle(batches)

            for i, batch in zip(range(n_iter), batches):
                self._minibatch_step(X_train[batch], dictionary,
                                     self._random_state, i)

                trigger_verbose = (self.verbose and
                                   i % ceil(100. / self.verbose) == 0)
                if self.verbose > 10 or trigger_verbose:
                    print(f"{i} batches processed.")

            self.n_iter_ = n_iter
            self.n_steps_ = n_iter

        self.components_ = dictionary

        return self

    def partial_fit(self, X, y=None, iter_offset="deprecated"):
        """Updates the model using the data in X as a mini-batch.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples in the number of samples
            and n_features is the number of features.

        y : Ignored

        iter_offset : int, default=None
            The number of iteration on data batches that has been
            performed before this call to partial_fit. This is optional:
            if no number is passed, the memory of the object is
            used.

            .. deprecated:: 1.0
               ``iter_offset`` will be removed in 1.2.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        has_components = hasattr(self, 'components_')

        X = self._validate_data(X, dtype=np.float64, order='C',
                                reset=not has_components)

        self._random_state = getattr(self, "_random_state",
                                     check_random_state(self.random_state))

        if iter_offset != "deprecated":
            warnings.warn("'iter_offset' is deprecated in version 1.0 and "
                          "will be removed in version 1.2", FutureWarning)
            self.n_steps_ = iter_offset
        else:
            self.n_steps_ = getattr(self, "n_steps_", 0)

        if not has_components:
            # this is the first call to partial_fit on this object
            self._check_params(X)

            dictionary = self._initialize_dict(X, self._random_state)

            self._inner_stats = (
                np.zeros((self._n_components, self._n_components)),
                np.zeros((X.shape[1], self._n_components)))
        else:
            dictionary = self.components_

        self._minibatch_step(
            X, dictionary, self._random_state, self.n_steps_)

        self.components_ = dictionary
        self.n_steps_ += 1

        return self
