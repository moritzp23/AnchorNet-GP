"""FITC negative log marginal likelihood, written as one differentiable forward.

This is the reference implementation: the gradient comes from autograd. The fits in the
paper use the analytic gradients in fitc_manual.py (unblocked) and fitc_blocked.py
(blocked over the data), which agree with this module.

With W = L_11^{-1} K_12, so that K_hat = W^T W, and the FITC diagonal
D = diag(K - K_hat) + sigma^2 I,

    logdet(K_tilde) = sum_i log D_i + logdet(M),        M = I_r + W D^{-1} W^T
    y^T K_tilde^{-1} y = y^T D^{-1} y - (W D^{-1} y)^T M^{-1} (W D^{-1} y)

M is symmetric positive definite and is factored by Cholesky.
"""
import math

import numpy as np
import torch

LOG_2PI = np.log(2 * np.pi)


def safe_cholesky(K, jitter=1e-4, max_tries=7):
    """Cholesky with escalating jitter, returning (L, jitter_used).

    jitter=0 first tries the exact factorization. A factor is accepted only if cuSOLVER
    reports success and the factor is finite: on numerically rank-deficient matrices
    cholesky_ex has been seen to report success while returning NaN on the diagonal.
    """
    eye = torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
    j = jitter
    if not math.isfinite(float(j)) or float(j) < 0.0:
        j = 0.0
    for _ in range(max_tries):
        L, info = torch.linalg.cholesky_ex(K + j * eye if j > 0 else K)
        if int(info) == 0 and bool(torch.isfinite(L).all()):
            return L, j
        j = j * 10.0 if j > 0 else 1e-10 * float(K.diagonal().mean().detach())
    raise torch.linalg.LinAlgError(
        f"not positive definite even with jitter {j:.1e}; "
        f"the rank is too high for this lengthscale in {K.dtype}.")


def _nll(logtheta, X, y, nyst_idx, rank, fitc, jitter=1e-4):
    # logtheta = [log l_1 .. log l_d, log sigma_f, log sigma_y] (ARD), or a single
    # shared log l in front of the last two entries.
    n_l = logtheta.numel() - 2
    ell = torch.exp(logtheta[:n_l])
    sigma_f = torch.exp(logtheta[n_l])
    noise = torch.exp(logtheta[n_l + 1])

    N = X.shape[0]
    y = y.reshape(-1)

    Xs = X / ell
    S_12 = torch.pow(torch.cdist(Xs[nyst_idx, :], Xs, p=2), 2)        # r x N
    K_12 = sigma_f ** 2 * torch.exp(-S_12 / 2)                        # r x N
    K_11 = K_12[:, nyst_idx]                                          # r x r

    L_11, _ = safe_cholesky(K_11, jitter * sigma_f ** 2)
    W = torch.linalg.solve_triangular(
        L_11.mT, K_12.mT, upper=True, left=False).mT                 # r x N

    if fitc:
        d_hat = (W * W).sum(0)                                        # diag(K_hat)
        D = torch.clamp(sigma_f ** 2 - d_hat, min=0.0) + noise ** 2
    else:
        D = (noise ** 2).expand(N)

    # M = I + V V^T with V = W / sqrt(D), which keeps M symmetric under rounding.
    rootD = torch.sqrt(D)
    V = W / rootD
    M = torch.eye(rank, device=X.device, dtype=X.dtype) + V @ V.T
    L_M, _ = safe_cholesky(M, jitter=0.0)

    logdet = torch.log(D).sum() + 2.0 * torch.log(torch.diagonal(L_M)).sum()

    Wdy = V @ (y / rootD)                                             # W D^-1 y
    v = torch.cholesky_solve(Wdy.unsqueeze(1), L_M).squeeze(1)
    quad = (y * y / D).sum() - (Wdy * v).sum()

    return 0.5 * (N * LOG_2PI + logdet + quad)


class NegativeLogLikelihood:
    """FITC objective with autograd gradients."""

    @staticmethod
    def apply(logtheta, X, y, nyst_idx, rank):
        return _nll(logtheta, X, y, nyst_idx, rank, fitc=True)
