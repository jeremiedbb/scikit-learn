""" Dictionary learning
"""
# Author: Vlad Niculae, Gael Varoquaux, Alexandre Gramfort
# License: BSD 3 clause

import time
import sys
import itertools

from math import ceil

import numpy as np
from scipy import linalg
from joblib import Parallel, delayed, effective_n_jobs

from ..base import BaseEstimator, TransformerMixin
from ..utils import (check_array, check_random_state, gen_even_slices,
                     gen_batches)
from ..utils._mask import _get_mask
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars

from sklearn.decomposition import sparse_encode
from sklearn.decomposition._dict_learning import get_loss


def sparse_encode_na(X, observed_mask, dictionary, alpha=1):
    # put 0 on nan and on corresponding column of D
    # then call sparse_encode
    
    if X.ndim == 1:
        X = X.reshape(1, -1)

    n_samples = X.shape[0]
    n_components = dictionary.shape[0]

    code_nan = np.zeros((n_samples, n_components))
    for i in range(n_samples):
        Do = np.multiply(dictionary, observed_mask[i])
        code_nan[i] = sparse_encode(X[[i]], Do, alpha=alpha, check_input=False)

    return code_nan


def update1(X, code, C, B, e, observed_mask, t, ro):
    """Update C, B, e inplace"""
    gamma = (1 - 1/t)**ro

    # update B
    # Bkj <- gamma * Bkj + x_obs_k . code_j^T
    B *= gamma
    B += np.outer(X, code)

    # update C
    # Cjk <- gamma * Cjk + mask_k * code_j²
    C *= gamma
    C += np.outer(code ** 2, observed_mask)

    # update e, part I
    # e_jk <- gamma * e_jk
    e *= gamma


def update_dict_na(C, B, e, D, code, observed_mask, Td = 5):
    """Update dictionary inplace"""
    e_temp = e.copy()
    for td in range(Td):        
        for j in range(D.shape[1]):
            D_code = D @ code
            e_temp[j] = e[j] + np.multiply(observed_mask, D_code) * code[j]

            # solve for uj: cj * uj = bj - ej + cj * dj
            # then dj <- uj
            np.divide(
                B[:,j] - e_temp[j] + np.multiply(C[j], D[:,j]), C[j],
                where=(C[j] != 0), out=D[:, j]
            )

            # Project uj on the constraint set
            D[:, j] /= linalg.norm(D[:, j])
            

def dict_learning_na(X, n_components=12, alpha=1, ro = .01,
                     T = 200):
    
    n_samples, n_features = X.shape

    # mask of the observed values of X
    observed_mask = np.logical_not(_get_mask(X, np.nan))

    # X observed, 0 where unobserved
    Xo = np.nan_to_num(X)

    # init code, dict
    code, S, dictionary = linalg.svd(Xo, full_matrices=False)
    dictionary = S[:, np.newaxis] * dictionary
    
    r = len(dictionary)
    if n_components <= r:  # True even if n_components=None
        code = code[:, :n_components]
        dictionary = dictionary[:n_components, :]
    else:
        code = np.c_[code, np.zeros((len(code), n_components - r))]
        dictionary = np.r_[dictionary,
                           np.zeros((n_components - r, dictionary.shape[1]))]

    D = dictionary.T
    
    # init stats
    C = np.zeros((n_components, n_features))
    B = np.zeros((n_features, n_components))
    e = np.zeros((n_components, n_features))
    
    loss = [] # record loss
    loss.append(get_loss(X, code, D.T, alpha))

    for t in range(1, T + 1):
        ii = t%n_samples

        # minibatch of X observed (1 sample for now)
        Xo_minibatch = Xo[ii]

        # observed mask for the minibatch
        observed_mask_minibatch = observed_mask[ii]
    
        # compute code for this minibatch
        this_code = sparse_encode_na(Xo_minibatch, observed_mask_minibatch,
                                     D.T, alpha)[0]

        # update stats
        update1(Xo_minibatch, this_code, C, B, e,
                observed_mask_minibatch, t, ro)

        # update dictionary
        update_dict_na(C, B, e, D, this_code, observed_mask_minibatch)

        # update e, part II
        # e_jk <- e_jk + mask_k * code_j * (D.code)_k
        D_code = D @ this_code
        e += np.outer(this_code, np.multiply(observed_mask_minibatch, D_code))

        # record loss
        code = sparse_encode_na(Xo, observed_mask, D.T, alpha)
        loss.append(get_loss(X, code, D.T, alpha))

    return code, D.T, loss
