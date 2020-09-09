import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import linalg
import time

from sklearn.decomposition import dict_learning_online
from sklearn.decomposition import sparse_encode

from sklearn.decomposition._dict_learning_na import sparse_encode_na,\
                                            update1, dict_learning_na,\
                                            update_dict_na



n_samples, n_features = 50, 25
rank = 12
np.random.seed(42)
U = np.random.randn(n_samples, rank)
V = np.random.randn(rank, n_features)
X = np.dot(U, V)

X_na = X.copy()
X_na[0,0] = np.nan
X_na[1,1] = np.nan

n_components = 11
code, dict_ = dict_learning_online(X, n_components = n_components,
                                   alpha = .0001)

def bench_loss(n_samples = 50, n_features = 25,
               rank = 12, n_components = 11):


    U = np.random.randn(n_samples, rank)
    V = np.random.randn(rank, n_features)
    X = np.dot(U, V)

    X_na = X.copy()
    X_na[0,0] = np.nan
    X_na[1,1] = np.nan

    for ro in [0, .01, .1, .4]:
        code_na, D, loss = dict_learning_na(X_na, n_components = n_components,
                                            alpha=0.00001, ro = ro)
        plt.plot(loss, label = str(ro))

    code, D = dict_learning_online(X, n_components = n_components,
                                            alpha=0.00001)

    loss_sklearn = np.mean(np.abs(X - np.dot(code, D)))
    plt.plot([loss_sklearn]*210, label = 'sklearn')
    plt.legend()
    plt.title('loss for dict learning na')
    plt.ylabel('loss')

def bench_dict_learning():
    
    def get_time_dict(n_samples = 100, n_features=50, n_iter = 10):
        X = np.random.randn(n_samples, n_features)
        n_components = 10

        to = time.time()
        for _ in range(n_iter):
            code, D = dict_learning_online(X, n_components = n_components)
        t1 = time.time()
        print('dict_learning took:\t', (t1 - to)/n_iter, 'seconds')
        t = (t1 - to)/n_iter

        to = time.time()
        for _ in range(n_iter):
            code_na, D, loss = dict_learning_na(X, n_components = n_components)
        t1 = time.time()
        print('dict_learning_na took:\t', (t1 - to)/n_iter, 'seconds')
        t_na = (t1 - to)/n_iter

        return t, t_na

    ln = [50, 100, 1000, 10000]
    lt, lt_na = [], []
    for n_samples in ln:
        t, t_na = get_time_dict(n_samples = n_samples, n_features=200, n_iter=1)
        lt.append(t)
        lt_na.append(t_na)

    plt.plot(ln, lt, label = 'sklearn')
    plt.plot(ln, lt_na, label = 'na')
    plt.legend()
    plt.xscale('log')
    plt.title('dict_learning_online')
    plt.ylabel('time (s)')


def bench_sparse_encode(n_samples = 100, n_features=50, n_iter = 10):
    
    def get_time_sparse():
        X = np.random.randn(n_samples, n_features)
        code, dict_ = dict_learning_online(X, n_components = 12, alpha = 1)

        to = time.time()
        for _ in range(n_iter):
            sparse_encode(X, dict_, alpha=1)
        t1 = time.time()
        print('sparse encode took:\t', (t1 - to)/n_iter, 'seconds')
        t = (t1 - to)/n_iter

        to = time.time()
        for _ in range(n_iter):
            sparse_encode_na(X, dict_, alpha=1)
        t1 = time.time()
        print('sparse encode_na took:\t', (t1 - to)/n_iter, 'seconds')
        t_na = (t1 - to)/n_iter

        return t, t_na

    ln = [10, 100,1000, 10000]
    lt, lt_na = [], []
    for n_samples in ln:
        t, t_na = get_time_sparse()
        lt.append(t)
        lt_na.append(t_na)

    plt.plot(ln, lt, label = 'sklearn')
    plt.plot(ln, lt_na, label = 'na')
    plt.legend()
    plt.xscale('log')
    plt.title('sparse encode')
    plt.ylabel('time (s)')

if __name__ == '__main__':
    bench_sparse_encode()
    plt.figure()
    bench_dict_learning()
    plt.figure()
    bench_loss()