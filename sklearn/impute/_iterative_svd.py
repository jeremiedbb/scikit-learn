import numpy as np

from ._base import _BaseImputer
from ..decomposition import TruncatedSVD


F32PREC = np.finfo(np.float32).eps


def masked_mae(X_true, X_pred, mask):
    masked_diff = X_true[mask] - X_pred[mask]
    return np.mean(np.abs(masked_diff))


def generate_random_column_samples(column):
    col_mask = np.isnan(column)
    n_missing = np.sum(col_mask)
    if n_missing == len(column):
        return np.zeros_like(column)

    mean = np.nanmean(column)
    std = np.nanstd(column)

    if np.isclose(std, 0):
        return np.array([mean] * n_missing)
    else:
        return np.random.randn(n_missing) * std + mean


class IterativeSVDImputer(_BaseImputer):
    def __init__(
        self,
        rank=10,
        tol=1e-4,
        max_iter=200,
        transform_max_iter=200,
        gradual_rank_increase=True,
        svd_algorithm="arpack",
        init_fill_method="zero",
        min_value=None,
        max_value=None,
        add_indicator=False,
        verbose=False
    ):
        super().__init__(
            add_indicator=add_indicator,
        )
        self.rank = rank
        self.max_iter = max_iter
        self.transform_max_iter = transform_max_iter
        self.tol = tol
        self.gradual_rank_increase = gradual_rank_increase
        self.svd_algorithm = svd_algorithm
        self.init_fill_method = init_fill_method
        self.min_value = min_value
        self.max_value = max_value
        self.verbose = verbose

    def _converged(self, X_old, X_new, missing_mask):
        # check for convergence
        old_missing_values = X_old[missing_mask]
        new_missing_values = X_new[missing_mask]
        difference = old_missing_values - new_missing_values
        ssd = np.sum(difference ** 2)
        old_norm_squared = (old_missing_values ** 2).sum()
        # edge cases
        if old_norm_squared == 0 or \
                (old_norm_squared < F32PREC and ssd > F32PREC):
            return False

        return (ssd / old_norm_squared) < self.tol

    def _init_fill(
        self,
        X,
        missing_mask,
    ):
        if self.init_fill_method == "zero":
            X[missing_mask] = 0
        elif self.init_fill_method == "mean":
            self._fill_columns_with_fn(X, missing_mask, np.nanmean)
        elif self.init_fill_method == "median":
            self._fill_columns_with_fn(X, missing_mask, np.nanmedian)
        elif self.init_fill_method == "min":
            self._fill_columns_with_fn(X, missing_mask, np.nanmin)
        elif self.init_fill_method == "random":
            self._fill_columns_with_fn(
                X,
                missing_mask,
                col_fn=generate_random_column_samples)

    def _fill_columns_with_fn(self, X, missing_mask, col_fn):
        for col_idx in range(X.shape[1]):
            missing_col = missing_mask[:, col_idx]
            n_missing = missing_col.sum()
            if n_missing == 0:
                continue
            col_data = X[:, col_idx]
            fill_values = col_fn(col_data)
            if np.all(np.isnan(fill_values)):
                fill_values = 0
            X[missing_col, col_idx] = fill_values

    def fit(self, X, y=None):
        self.fit_transform(X, y)
        return self

    def fit_transform(self, X, y=None):
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], force_all_finite="allow-nan"
        )

        return self._fit_transform(X, update_components=True)

    def _fit_transform(self, X, *, update_components=True):
        missing_mask = np.isnan(X)
        observed_mask = ~missing_mask

        super()._fit_indicator(missing_mask)

        # fill missing values
        X_filled = X.copy()
        X_filled[missing_mask] = 0

        for i in range(1, self.max_iter + 1):
            # deviation from original svdImpute algorithm:
            # gradually increase the rank of the approximation
            if self.gradual_rank_increase:
                curr_rank = min(2 ** i, self.rank)
            else:
                curr_rank = self.rank

            if update_components:
                tsvd = TruncatedSVD(curr_rank, algorithm=self.svd_algorithm)
                X_reduced = tsvd.fit_transform(X_filled)
                self.components_ = tsvd.components_
            else:
                X_reduced = X_filled @ self.components_.T
            X_reconstructed = X_reduced @ self.components_

            if self.min_value is not None or self.max_value is not None:
                np.clip(
                    X_reconstructed, self.min_value, self.max_value, out=X_reconstructed
                )

            if self.verbose and update_components:
                # reconstruction quality of observed values
                mae = masked_mae(
                    X_true=X,
                    X_pred=X_reconstructed,
                    mask=observed_mask
                )
                print(f"[IterativeSVD] Iter {i}: observed MAE={mae}")

            converged = self._converged(
                X_old=X_filled,
                X_new=X_reconstructed,
                missing_mask=missing_mask)

            X_filled[missing_mask] = X_reconstructed[missing_mask]

            if converged:
                break

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
