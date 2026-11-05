"""FITC objective with analytic gradients, unblocked (all r x N intermediates in memory).

The derivation follows Section 3 of the paper. With A = K_11^{-1} K_12, the adjoint of
the kernel block is formed once,

    Psi = df/dK_12 = 2 A Phi,   Psi[:, nyst] -= A Phi A^T,
    Phi = 0.5 (K_tilde^{-1} - alpha alpha^T) - diag(p * active),
    p_i = 0.5 ([K_tilde^{-1}]_ii - alpha_i^2),

where the diagonal term is the contribution of the FITC correction. Each ARD length scale
is then a contraction df/dlog l_k = <Psi * K_12, S_k>, so the total cost is
O(r^2 n + d r n). The amplitude and the noise need no further O(r^2 n) work, because
K_hat is exactly quadratic in sigma_f and the noise enters only through D.
"""
import numpy as np
import torch

from .fitc import safe_cholesky

LOG_2PI = np.log(2 * np.pi)


def _tri_rn(T, B, upper):
    """Solve T X = B for X, with B of size r x N, N large and r small.

    Same result as torch.linalg.solve_triangular(T, B, upper=upper), but cuBLAS is asked
    for the transposed problem. The left-sided form is very slow for small r and very
    wide B (rank 50 on Buzz: 38.6 s against 0.22 s).
    """
    return torch.linalg.solve_triangular(
        T.mT, B.mT, upper=not upper, left=False).mT


def _pieces(logtheta, X, y, nyst_idx, rank, fitc, jitter):
    """Factorization shared by the value and the gradient."""
    n_l = logtheta.numel() - 2
    ell = torch.exp(logtheta[:n_l])
    sigma_f = torch.exp(logtheta[n_l])
    noise = torch.exp(logtheta[n_l + 1])

    N = X.shape[0]
    Xs = X / ell
    Xm = Xs[nyst_idx, :]
    K_12 = sigma_f ** 2 * torch.exp(-0.5 * torch.pow(torch.cdist(Xm, Xs, p=2), 2))
    K_11 = K_12[:, nyst_idx]

    L_11, _ = safe_cholesky(K_11, jitter * sigma_f ** 2)
    W = _tri_rn(L_11, K_12, upper=False)                           # r x N
    A = _tri_rn(L_11.T, W, upper=True)                             # K_11^-1 K_12

    if fitc:
        slack = sigma_f ** 2 - (W * W).sum(0)                       # sigma_f^2 - diag(K_hat)
        active = (slack > 0).to(X.dtype)                            # clamp is inactive here
        D = torch.clamp(slack, min=0.0) + noise ** 2
    else:
        active = torch.zeros(N, device=X.device, dtype=X.dtype)
        D = (noise ** 2).expand(N).clone()

    rootD = torch.sqrt(D)
    V = W / rootD
    M = torch.eye(rank, device=X.device, dtype=X.dtype) + V @ V.T
    L_M, _ = safe_cholesky(M, 0.0)

    yD = y.reshape(-1) / D
    v = torch.cholesky_solve((W @ yD).unsqueeze(1), L_M).squeeze(1)
    alpha = yD - (W.T @ v) / D                                      # K_tilde^-1 y
    quad = (y.reshape(-1) * alpha).sum()

    return dict(quad=quad, n_l=n_l, sigma_f=sigma_f, noise=noise, N=N, Xm=Xm, Xs=Xs,
                K_12=K_12, W=W, A=A, D=D, L_M=L_M, alpha=alpha, active=active,
                fitc=fitc, nyst_idx=nyst_idx)


def _value(p, y):
    logdet = torch.log(p['D']).sum() + 2.0 * torch.log(torch.diagonal(p['L_M'])).sum()
    return 0.5 * (p['N'] * LOG_2PI + logdet + (y.reshape(-1) * p['alpha']).sum())


def _gradient(p):
    """Analytic dNLL/dlogtheta, length n_l + 2."""
    n_l = p['n_l']
    W, A, D, L_M, alpha = p['W'], p['A'], p['D'], p['L_M'], p['alpha']
    K_12, Xm, Xs = p['K_12'], p['Xm'], p['Xs']
    nyst_idx, active = p['nyst_idx'], p['active']
    noise, fitc, N = p['noise'], p['fitc'], p['N']

    # diag(K_tilde^-1) = 1/D - colsum((L_M^-1 W)^2)/D^2
    Z = _tri_rn(L_M, W, upper=False)
    diag_Kinv = 1.0 / D - (Z * Z).sum(0) / D ** 2
    del Z

    AD = A / D
    C = AD @ W.T
    Cm = torch.cholesky_solve(C.T, L_M).T
    WD = W / D
    AKi = AD - Cm @ WD                                # A K_tilde^-1, r x n
    del AD, WD, C

    a = A @ alpha
    alpha2 = alpha ** 2

    q = 0.5 * (diag_Kinv - alpha2) * active if fitc else torch.zeros_like(D)
    APhi = 0.5 * (AKi - torch.outer(a, alpha)) - A * q
    del AKi
    APhiAt = APhi @ A.T

    Psi = 2.0 * APhi
    Psi[:, nyst_idx] -= APhiAt
    del APhi

    Psi *= K_12
    grads = torch.zeros(n_l + 2, device=W.device, dtype=W.dtype)

    # column k of Xs is strided by d; a transposed copy keeps the reads below coalesced
    XsT = Xs.T.contiguous() if n_l > 1 else None
    for k in range(n_l):
        if n_l == 1:
            S_k = torch.pow(torch.cdist(Xm, Xs, p=2), 2)
        else:
            S_k = (Xm[:, k:k + 1] - XsT[k].unsqueeze(0)) ** 2
        grads[k] = (Psi * S_k).sum()
        del S_k
    del Psi

    # log sigma_f: dK_hat/dlog sigma_f = 2 K_hat, and K_tilde alpha = y.
    tr_sf = 2.0 * (N - (diag_Kinv * D).sum())
    q_sf = 2.0 * (p['quad'] - (D * alpha2).sum())
    dD = active * 2.0 * (D - noise ** 2) if fitc else torch.zeros_like(D)
    grads[n_l] = 0.5 * ((tr_sf + (diag_Kinv * dD).sum()) - (q_sf + (dD * alpha2).sum()))

    # log sigma_y enters only through D, with dD/dlog sigma_y = 2 sigma^2
    two_s2 = 2.0 * noise ** 2
    grads[n_l + 1] = 0.5 * two_s2 * (diag_Kinv.sum() - alpha2.sum())

    return grads


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logtheta, X, y, nyst_idx, rank, fitc, jitter):
        with torch.no_grad():
            p = _pieces(logtheta, X, y, nyst_idx, rank, fitc, jitter)
            val = _value(p, y)
            ctx.grad_theta = _gradient(p)
        return val

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.grad_theta, None, None, None, None, None, None


class NegativeLogLikelihood:
    """FITC objective with analytic gradients."""

    @staticmethod
    def apply(logtheta, X, y, nyst_idx, rank, jitter=1e-4):
        return _Fn.apply(logtheta, X, y, nyst_idx, rank, True, jitter)
