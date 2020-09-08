## test

import numpy as np
import time

from sklearn.decomposition._dict_learning_na import sparse_encode_na,\
                                            update1, dict_learning_na,\
                                            update_dict_na
from sklearn.decomposition import dict_learning_online
from sklearn.decomposition import sparse_encode

from sklearn.utils._testing import assert_array_almost_equal
from sklearn.utils._testing import assert_array_equal


def test_shape_encode():
    
    this_x = X[:5]
    this_code = sparse_encode_na(this_x, dict_, alpha=1)
    assert this_code.shape == (this_x.shape[0], n_components)
    
    this_x = X_na[:5]
    this_code = sparse_encode_na(this_x, dict_, alpha=1)
    assert this_code.shape == (this_x.shape[0], n_components)
        
def test_encode_reconstruction():
    
    code = sparse_encode_na(X, dict_, alpha=.0001)
    assert_array_almost_equal(np.dot(code, dict_), X, decimal=2)
    
    code = sparse_encode_na(X_na, dict_, alpha=.0001)
    X_new = np.dot(code, dict_)
    assert_array_almost_equal(X_new[Mask],
                              X[Mask], decimal=2)
       
def test_dict_learning_na():
    
    code, dict_, loss = dict_learning_na(X, n_components = 8, alpha = .0001)
    # assert_array_almost_equal(np.dot(code, dict_), X, decimal=2)

    code, dict_, loss = dict_learning_na(X_na, n_components = 8, alpha = .0001)
    # assert_array_almost_equal(np.dot(code, dict_), X, decimal=2)


rng_global = np.random.RandomState(0)
n_samples, n_features = 10, 8
n_components = 12
X = rng_global.randn(n_samples, n_features)
code, dict_ = dict_learning_online(X, n_components = n_components, alpha = .0001)

X_na = X.copy()
X_na[0,0] = np.nan
X_na[1,1] = np.nan
Mask = np.where(~np.isnan(X_na))

test_shape_encode()
test_encode_reconstruction()
test_dict_learning_na()