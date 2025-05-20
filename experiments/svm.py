from pathlib import Path
import os
import gzip

import numpy as np
import torch
from torch import Tensor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from botorch.test_functions.synthetic import SyntheticTestFunction


class SVMReduced(SyntheticTestFunction):
    """
    A synthetic benchmark using an SVR model on a reduced CT slice dataset.
    Dimensionality reduction is done using XGBoost-based feature selection.
    The final three input dimensions are used as SVM hyperparameters.
    """
    _optimal_value = None
    _check_grad_at_opt: bool = False
    _bounds = None
    _optimizers = None

    def __init__(
        self,
        dim: int,
        bounds: Tensor,
        noise_std: float = 0.0,
        negate: bool = True,
    ):
        self.feature_dim = dim - 3
        self.dim = dim
        super().__init__(
            bounds=bounds,
            noise_std=noise_std,
            negate=negate,
        )
        self._load_data(n_features=self.feature_dim)

    def _load_data(self, n_features: int):
        data_folder = os.path.join(Path(__file__).parent.parent, "data", "svm")
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except FileNotFoundError:
            with gzip.open(os.path.join(data_folder, "CT_slice_X.npy.gz"), "rb") as fx, \
                 gzip.open(os.path.join(data_folder, "CT_slice_y.npy.gz"), "rb") as fy:
                X = np.load(fx)
                y = np.load(fy)

        # Normalize features
        X -= X.min(axis=0)
        X = X[:, X.max(axis=0) > 1e-6]  # Remove near-constant features
        X = X / (X.max(axis=0) - X.min(axis=0))
        X = 2 * X - 1
        assert X.min() == -1 and X.max() == 1

        # Standardize target
        y = (y - y.mean()) / y.std()

        # Subsample 10,000 examples
        rng = np.random.RandomState(0)
        indices = rng.permutation(X.shape[0])[:10_000]
        X, y = X[indices], y[indices]

        # Feature selection via XGBoost
        xgb = XGBRegressor(max_depth=8).fit(X, y)
        top_indices = (-xgb.feature_importances_).argsort()[:n_features]
        X = X[:, top_indices]

        # Train-test split
        split = X.shape[0] // 2
        self._X_train, self._y_train = X[:split], y[:split]
        self._X_test, self._y_test = X[split:], y[split:]

    def evaluate_true(self, X: Tensor) -> Tensor:
        return torch.cat([self._evaluate_true(x) for x in X])

    def _evaluate_true(self, X: Tensor) -> Tensor:
        x = X.numpy()
        lengthscales = x[: self.feature_dim]
        epsilon = 0.01 * 10 ** (2 * x[-3])  # Default ~0.1
        C = 0.01 * 10 ** (4 * x[-2])        # Default ~1.0
        gamma = (1 / self.feature_dim) * 0.1 * 10 ** (2 * x[-1])  # Default ~1.0 / dim

        model = SVR(
            C=C, epsilon=epsilon, gamma=gamma,
            tol=0.001, cache_size=1000, verbose=False
        )
        model.fit(self._X_train * lengthscales, self._y_train.copy())
        pred = model.predict(self._X_test * lengthscales)
        mse = ((pred - self._y_test) ** 2).mean()
        return torch.tensor([np.sqrt(mse)], dtype=X.dtype)