import numpy as np

from ._base import _BaseImputer
from ..utils.extmath import randomized_svd


F32PREC = np.finfo(np.float32).eps


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


class SoftImputer(_BaseImputer):
    """
    Implementation of the SoftImpute algorithm from:
    "Spectral Regularization Algorithms for Learning Large Incomplete Matrices"
    by Mazumder, Hastie, and Tibshirani.
    """
    def __init__(
            self,
            shrinkage_value=None,
            convergence_threshold=0.001,
            max_iters=100,
            max_rank=None,
            n_power_iterations=1,
            init_fill_method="zero",
            min_value=None,
            max_value=None,
            normalizer=None,
            transform_max_iter=100,
            add_indicator=False,
            verbose=True):
        """
        Parameters
        ----------
        shrinkage_value : float
            Value by which we shrink singular values on each iteration. If
            omitted then the default value will be the maximum singular
            value of the initialized matrix (zeros for missing values) divided
            by 50.

        convergence_threshold : float
            Minimum ration difference between iterations (as a fraction of
            the Frobenius norm of the current solution) before stopping.

        max_iters : int
            Maximum number of SVD iterations

        max_rank : int, optional
            Perform a truncated SVD on each iteration with this value as its
            rank.

        n_power_iterations : int
            Number of power iterations to perform with randomized SVD

        init_fill_method : str
            How to initialize missing values of data matrix, default is
            to fill them with zeros.

        min_value : float
            Smallest allowable value in the solution

        max_value : float
            Largest allowable value in the solution

        normalizer : object
            Any object (such as BiScaler) with fit() and transform() methods

        verbose : bool
            Print debugging info
        """
        super().__init__(add_indicator=add_indicator)
        self.init_fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.normalizer = normalizer
        self.shrinkage_value = shrinkage_value
        self.convergence_threshold = convergence_threshold
        self.max_iters = max_iters
        self.max_rank = max_rank
        self.transform_max_iter = transform_max_iter
        self.n_power_iterations = n_power_iterations
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm = np.sqrt((old_missing_values ** 2).sum())
        # edge cases
        if old_norm == 0 or (old_norm < F32PREC and np.sqrt(ssd) > F32PREC):
            return False
        else:
            return (np.sqrt(ssd) / old_norm) < self.convergence_threshold

    def _svd_step(self, X, shrinkage_value, max_rank=None):
        """
        Returns reconstructed X from low-rank thresholded SVD and
        the rank achieved.
        """
        if max_rank:
            # if we have a max rank then perform the faster randomized SVD
            (U, s, V) = randomized_svd(
                X,
                max_rank,
                n_iter=self.n_power_iterations,
                random_state=None)
        else:
            # perform a full rank SVD using ARPACK
            (U, s, V) = np.linalg.svd(
                X,
                full_matrices=False,
                compute_uv=True)

        s_thresh = np.maximum(s - shrinkage_value, 0)
        rank = (s_thresh > 0).sum()
        s_thresh = s_thresh[:rank]
        U_thresh = U[:, :rank]
        V_thresh = V[:rank, :]
        S_thresh = np.diag(s_thresh)
        #X_reconstruction = np.dot(U_thresh, np.dot(S_thresh, V_thresh))
        return U_thresh, S_thresh, V_thresh, rank

    def _max_singular_value(self, X_filled):
        # quick decomposition of X_filled into rank-1 SVD
        _, s, _ = randomized_svd(
            X_filled,
            1,
            n_iter=5,
            random_state=None)
        return s[0]

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], force_all_finite="allow-nan"
        )

        missing_mask = np.isnan(X)
        observed_mask = ~missing_mask

        super()._fit_indicator(missing_mask)

        X_filled = X.copy()
        X_filled[missing_mask] = 0

        max_singular_value = self._max_singular_value(X_filled)
        if self.verbose:
            print("[SoftImpute] Max Singular Value of X_init = %f" % (
                max_singular_value))

        if self.shrinkage_value is not None:
            shrinkage_value = self.shrinkage_value
        else:
            # totally hackish heuristic: keep only components
            # with at least 1/50th the max singular value
            shrinkage_value = max_singular_value / 50.0

        for i in range(self.max_iters):
            U, S, V, rank = self._svd_step(
                X_filled,
                shrinkage_value,
                max_rank=self.max_rank)
            X_reduced = U @ S
            self.components_ = V
            X_reconstructed = X_reduced @ V

            if self.min_value is not None or self.max_value is not None:
                np.clip(
                    X_reconstructed, self.min_value, self.max_value, out=X_reconstructed
                )

            # print error on observed data
            if self.verbose:
                mae = masked_mae(
                    X_true=X,
                    X_pred=X_reconstructed,
                    mask=observed_mask)
                print(
                    "[SoftImpute] Iter %d: observed MAE=%0.6f rank=%d" % (
                        i + 1,
                        mae,
                        rank))

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)
            X_filled[missing_mask] = X_reconstructed[missing_mask]

            self.rank_ = rank

            if converged:
                break

        if self.verbose:
            print("[SoftImpute] Stopped after iteration %d for lambda=%f" % (
                i + 1,
                shrinkage_value))

        X_indicator = super()._transform_indicator(missing_mask)

        return super()._concatenate_indicator(X_filled, X_indicator)

    def transform(self, X):
        # Repeat:
        #   Project onto the reduced space
        #   Project back onto the original space
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], force_all_finite="allow-nan", reset=False
        )

        # fill missing values
        missing_mask = np.isnan(X)
        X_filled = X.copy()
        X_filled[missing_mask] = 0

        for i in range(self.transform_max_iter):
            X_reduced = X_filled @ self.components_.T
            X_reconstructed = X_reduced @ self.components_

            if self.min_value is not None or self.max_value is not None:
                np.clip(
                    X_reconstructed, self.min_value, self.max_value, out=X_reconstructed
                )

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)

            X_filled[missing_mask] = X_reconstructed[missing_mask]

            if converged:
                break

        X_indicator = super()._transform_indicator(missing_mask)

        return super()._concatenate_indicator(X_filled, X_indicator)

    def score(self, X, y=None):
        # y = X_true
        Xt = self.transform(X)
        missing_mask = np.isnan(X)

        err = np.sum((Xt[missing_mask] - y[missing_mask])**2) / np.sum(y[missing_mask]**2)
        return 1 - err
