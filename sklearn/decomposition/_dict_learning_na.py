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
from ..utils.extmath import randomized_svd, row_norms
from ..utils.validation import check_is_fitted
from ..linear_model import Lasso, orthogonal_mp_gram, LassoLars, Lars

from sklearn.decomposition import sparse_encode
from sklearn.decomposition._dict_learning import get_loss


def sparse_encode_na(X, dictionary, alpha=1):
    # put 0 on nan and on corresponding column of D
    # then call sparse_encode
    
    if len(X.shape) == 1:
        X = X.reshape((1,-1))

    code_nan = []
    for x in X:

        d = dictionary.copy()
        curr_x = x.reshape(1,-1)
        curr_nan = np.isnan(curr_x)
        curr_x[curr_nan] = 0
        d[:, np.where(curr_nan)] = 0

        curr_code_nan = sparse_encode(curr_x, d, alpha = alpha)

        code_nan.append(curr_code_nan)

    return np.vstack(code_nan)


def update1(x, D, code, C, B, e, Delta, t, ro):
    
    # batch size = 1
    assert len(x.shape) == 1
    assert len(code.shape) == 1

    gamma = (1 - 1/t)**ro
    
    # dot product between x and code
    x_code =  np.zeros((len(x), len(code)))
    for i in range(len(x)):
        for j in range(len(code)):
            x_code[i,j] = x[i] * code[j]

    B = gamma * B + np.dot(Delta, x_code)
    
    for j in range(len(C)):
        code_square = code[j]**2
        C[j] = gamma * C[j] + code_square * Delta
        e[j] = gamma * e[j]
    
    return C, B, e

def update_dict_na(C, B, e, D, code, Delta, Td = 5):

    assert len(code.shape) == 1

    e_temp = e.copy()
    for td in range(Td):        
        for j in range(len(e)):      
            D_code = np.dot(D, code)
            Delta_D_code = np.dot(Delta, D_code)
            e_temp[j] = e[j] + code[j] * Delta_D_code

            right_part = B[:,j] - e_temp[j] + np.dot(C[j], D[:,j])
            
            #FIXME LinalgError if rank(X) < n_components
            if np.max(np.abs(C[j])) == 0:
                # if C[j] is 0, assign random weight to u_j...
                # print(j, td, 'C_j ==0, use uj = randn')
                # u_j = np.random.randn(D[:, j].shape[0])
                u_j = D[:, j]
            else:
                try:
                    u_j = linalg.solve(C[j], right_part)  
                except np.linalg.LinAlgError:
                    # if C[j] is singular, assign random weight to u_j...
                    print(j, td, 'C_j Singular, use uj = 1')
                    u_j = np.ones(D[:, j].shape[0])
                    raise
            D[:, j] = u_j / linalg.norm(u_j, ord= 2)

    return D


def dict_learning_na(X, n_components=12, alpha=1, ro = .01,
                     T = 200):
    
    n_samples, n_feat = X.shape
    
    X_init = np.nan_to_num(X)
    code, S, dictionary = linalg.svd(X_init, full_matrices=False)
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
    assert D.shape == (n_feat, n_components)
    assert code.shape == (n_samples, n_components)
    
    
    # init:
    C = []
    e = []
    for j in range(n_components):
        C.append(np.zeros((n_feat, n_feat)))
        e.append(np.zeros(n_feat))
    B = np.zeros((n_feat, n_components))
    
    loss = [] # recod loss
    loss.append(get_loss(X, code, D.T, alpha))
    for t in range(1, T):
        ii = t%n_samples
        x = X[ii] 
        Delta = np.diag(~np.isnan(x))
        
        this_code = sparse_encode_na(x, D.T, alpha)
        this_code = this_code[0] # working with batch 1 (and vector)
        
        
        C, B, e = update1(x, D, this_code, C, B, e, Delta, t, ro)
        assert C[j].shape == (n_feat, n_feat)
        assert B.shape == (n_feat, n_components)

        # print(C)
        D = update_dict_na(C, B, e, D, this_code, Delta)
        assert D.shape == (n_feat, n_components)

        #update2
        D_code = np.dot(D, this_code)
        Delta_D_code = np.dot(Delta, D_code)
        for j in range(len(e)):    
            e[j] = e[j] + this_code[j] * Delta_D_code
        
        code = sparse_encode_na(X, D.T, alpha)
        loss.append(get_loss(X, code, D.T, alpha))

    return code, D.T, loss
