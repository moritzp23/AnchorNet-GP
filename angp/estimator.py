"""scikit-learn style interface to ANGP."""
import numpy as np
import torch

from . import train
from .predict import posterior_dense, posterior_keops, posterior_mean_lr, posterior_var_dense

try:
    from sklearn.base import BaseEstimator, RegressorMixin
except ImportError:                       # scikit-learn is optional
    BaseEstimator = RegressorMixin = object


class ANGPRegressor(RegressorMixin, BaseEstimator):
    """Gaussian process regression with an ARD squared-exponential kernel, trained on a
    FITC low-rank approximation with AnchorNet landmarks.

    Parameters
    ----------
    rank : int
        Number of landmarks r. Larger ranks are more accurate and more expensive; the
        cost of a fit grows roughly with r^2.
    mean : {'lowrank', 'exact'}
        Predictive mean returned by predict(). 'lowrank' is the mean of the FITC model,
        which needs only the low-rank factors. 'exact' is the mean of the full-kernel GP,
        from a dense solve for n <= max_dense_n and from preconditioned KeOps CG in fp32
        otherwise (see the paper for when fp32 CG does not converge).
    n_iter, lr, grad_clip : Adam iterations, initial learning rate and gradient clipping.
    noise_floor : lower bound on the noise standard deviation (standardized units).
    gradients : {'adaptive', 'manual', 'blocked'}
        'manual' keeps all r x n intermediates in memory, 'blocked' processes the data in
        blocks of 20000 rows, 'adaptive' times both and picks one.
    normalize : bool
        Standardize every input column and the targets with the mean and standard
        deviation of the training data (computed in float64, with 1e-6 added to the
        standard deviation as in bayesian_benchmarks). Predictions and the fitted
        hyperparameters below are in the original units. bayesian_benchmarks instead uses
        the statistics of the full dataset, including the test points; to reproduce the
        paper, pass its already standardized data with normalize=False.
    precond_rank, cg_max_iter, cg_tol, max_dense_n : settings of the 'exact' mean.
    device : torch device; a CUDA GPU is needed for the KeOps solve and is much faster.
    random_state : seed for the subsample of the median heuristic.

    Attributes
    ----------
    lengthscale_ : (d,) ndarray, ARD length scales
    signal_std_ : float, sigma_f
    noise_std_ : float, sigma_y
    landmarks_ : (r,) ndarray, indices of the landmarks in the training data
    fit_time_ : float, seconds
    """

    def __init__(self, rank=1000, mean='lowrank', n_iter=100, lr=0.2, grad_clip=10.0,
                 noise_floor=1e-4, gradients='adaptive', normalize=True, precond_rank=200,
                 cg_max_iter=200, cg_tol=1e-5, max_dense_n=60000, device='cuda',
                 random_state=42, verbose=False):
        self.rank = rank
        self.mean = mean
        self.n_iter = n_iter
        self.lr = lr
        self.grad_clip = grad_clip
        self.noise_floor = noise_floor
        self.gradients = gradients
        self.normalize = normalize
        self.precond_rank = precond_rank
        self.cg_max_iter = cg_max_iter
        self.cg_tol = cg_tol
        self.max_dense_n = max_dense_n
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _tensor(self, A):
        return torch.as_tensor(np.asarray(A, dtype=np.float32), device=self.device)

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.ndim != 2 or X.shape[0] != y.shape[0]:
            raise ValueError(f'X must be (n, d) and y (n,), got {X.shape} and {y.shape}')
        if self.mean not in ('lowrank', 'exact'):
            raise ValueError(f"mean must be 'lowrank' or 'exact', not {self.mean!r}")
        self.n_features_in_ = X.shape[1]
        if self.normalize:
            self.x_mean_, self.x_std_ = X.mean(0), 1e-6 + X.std(0)
            self.y_mean_, self.y_std_ = float(y.mean()), 1e-6 + float(y.std())
        else:
            self.x_mean_, self.x_std_ = np.zeros(X.shape[1]), np.ones(X.shape[1])
            self.y_mean_, self.y_std_ = 0.0, 1.0

        self._X = self._tensor((X - self.x_mean_) / self.x_std_)
        self._Y = self._tensor((y - self.y_mean_) / self.y_std_).reshape(-1, 1)
        res = train.fit(self._X, self._Y, min(self.rank, X.shape[0]), n_iter=self.n_iter,
                        lr=self.lr, grad_clip=self.grad_clip, noise_floor=self.noise_floor,
                        grads=self.gradients, seed=self.random_state, verbose=self.verbose)
        self._ell, self._sf, self._sy = res['ell'], res['sigma_f'], res['sigma_y']
        self._nyst = res['nyst_idx']
        self.lengthscale_ = res['ell'].detach().cpu().numpy() * self.x_std_
        self.signal_std_ = float(res['sigma_f']) * self.y_std_
        self.noise_std_ = float(res['sigma_y']) * self.y_std_
        self.landmarks_ = res['nyst_idx'].cpu().numpy()
        self.fit_time_ = res['fit_time']
        return self

    def predict(self, X, return_std=False, noise=True):
        """Predictive mean at X, and with return_std=True also the standard deviation.

        The standard deviation is that of a new observation (noise=True) or of the latent
        function (noise=False). With mean='lowrank' it is the FITC predictive variance.
        With mean='exact' it is computed with a dense factorization and is therefore only
        available for at most max_dense_n training points.
        """
        if not hasattr(self, '_X'):
            raise RuntimeError('call fit before predict')
        dense = self._X.shape[0] <= self.max_dense_n
        if return_std and self.mean == 'exact' and not dense:
            raise NotImplementedError(
                f'the full-kernel variance needs a dense factorization, which is limited to '
                f'max_dense_n = {self.max_dense_n} training points; use mean="lowrank" for '
                f'the FITC variance')
        Xs = self._tensor((np.asarray(X, dtype=np.float64) - self.x_mean_) / self.x_std_)
        var = None
        with torch.no_grad():
            if self.mean == 'lowrank':
                out = posterior_mean_lr(Xs, self._X, self._Y, self._nyst, self._ell,
                                        self._sf, self._sy, return_var=return_std)
                mu, var = out if return_std else (out, None)
            elif dense:
                mu = posterior_dense(Xs, self._X, self._Y, self._ell, self._sf, self._sy)
                if return_std:
                    var = posterior_var_dense(Xs, self._X, self._ell, self._sf, self._sy)
            else:
                mu, _, _, _ = posterior_keops(Xs, self._X, self._Y, self._ell, self._sf,
                                              self._sy, self.cg_tol, self.cg_max_iter,
                                              self._nyst, self.precond_rank)
        mean = mu.reshape(-1).double().cpu().numpy() * self.y_std_ + self.y_mean_
        if not return_std:
            return mean
        if noise:
            var = var + self._sy ** 2
        return mean, np.sqrt(var.double().cpu().numpy()) * self.y_std_
