"""FITC objective and analytic gradients, blocked over the data: O(rB + r^2) memory.

Everything the objective and its gradient need is a scalar, an n-vector or an r x r
accumulator, so only one r x B block of the kernel is held at a time. Three passes:

  pass 1   M = I + sum_i w_i w_i^T / d_i  and  u = sum_i w_i y_i / d_i; factor M.
  pass 2   alpha_i = (y_i - w_i^T M^{-1} u) / d_i, diag(K_tilde^{-1}) and the r x r
           pieces of the gradient.
  pass 3   the adjoint of K_12 and the length-scale contractions.

K_hat is exactly quadratic in sigma_f, so the sigma_f gradient needs no further
O(r^2 n) work:

    tr(K_tilde^{-1} K_hat) = n - sum_i [K_tilde^{-1}]_ii d_i
    alpha^T K_hat alpha    = y^T alpha - sum_i d_i alpha_i^2
"""
import numpy as np
import torch

from .fitc import safe_cholesky

LOG_2PI = np.log(2 * np.pi)


def _kern(Xa, Xb, sigma_f):
    """Kernel block; inputs are pre-scaled by the length scales."""
    return sigma_f ** 2 * torch.exp(-0.5 * torch.pow(torch.cdist(Xa, Xb, p=2), 2))


def nll_and_grad(logtheta, X, y, nyst_idx, rank, fitc=True, jitter=1e-4,
                 chunk=20000, need_grad=True):
    dev, dt = X.device, X.dtype
    n_l = logtheta.numel() - 2
    ell = torch.exp(logtheta[:n_l])
    sigma_f = torch.exp(logtheta[n_l])
    noise = torch.exp(logtheta[n_l + 1])
    N = X.shape[0]
    y = y.reshape(-1)

    Xs = X / ell
    Xm = Xs[nyst_idx, :].contiguous()
    K_11 = _kern(Xm, Xm, sigma_f)
    # Jitter relative to sigma_f^2 keeps K_hat exactly quadratic in sigma_f.
    L_11, _ = safe_cholesky(K_11, jitter * sigma_f ** 2)

    def _S(Xa, Xb, k):
        """S_k for one block. A single shared length scale takes the full distance."""
        if n_l == 1:
            return torch.pow(torch.cdist(Xa, Xb, p=2), 2)
        return (Xa[:, k:k + 1] - Xb[:, k].unsqueeze(0)) ** 2

    # ---------------- pass 1: M, u, D ----------------
    M = torch.eye(rank, device=dev, dtype=dt)
    u = torch.zeros(rank, device=dev, dtype=dt)
    D = torch.empty(N, device=dev, dtype=dt)
    logdet_D = torch.zeros((), device=dev, dtype=dt)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Kb = _kern(Xm, Xs[i:j], sigma_f)
        Wb = torch.linalg.solve_triangular(L_11, Kb, upper=False)
        Db = (torch.clamp(sigma_f ** 2 - (Wb * Wb).sum(0), min=0.0) + noise ** 2
              if fitc else (noise ** 2).expand(j - i).clone())
        D[i:j] = Db
        logdet_D = logdet_D + torch.log(Db).sum()
        Vb = Wb / torch.sqrt(Db)
        M += Vb @ Vb.T
        u += Wb @ (y[i:j] / Db)
        del Kb, Wb, Vb, Db

    L_M, _ = safe_cholesky(M, 0.0)
    v = torch.cholesky_solve(u.unsqueeze(1), L_M).squeeze(1)
    logdet = logdet_D + 2.0 * torch.log(torch.diagonal(L_M)).sum()

    # ---------------- pass 2: alpha, C, a, diag(K^-1) ----------------
    alpha = torch.empty(N, device=dev, dtype=dt)
    diag_Kinv = torch.empty(N, device=dev, dtype=dt) if need_grad else None
    if need_grad:
        C = torch.zeros(rank, rank, device=dev, dtype=dt)   # A D^-1 W^T
        a = torch.zeros(rank, device=dev, dtype=dt)         # A alpha

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Kb = _kern(Xm, Xs[i:j], sigma_f)
        Wb = torch.linalg.solve_triangular(L_11, Kb, upper=False)
        Db = D[i:j]
        ab = (y[i:j] - Wb.T @ v) / Db
        alpha[i:j] = ab
        if need_grad:
            Ab = torch.linalg.solve_triangular(L_11.T, Wb, upper=True)
            C += (Ab / Db) @ Wb.T
            a += Ab @ ab
            Zb = torch.linalg.solve_triangular(L_M, Wb, upper=False)
            diag_Kinv[i:j] = 1.0 / Db - (Zb * Zb).sum(0) / Db ** 2
            del Ab, Zb
        del Kb, Wb

    quad = (y * alpha).sum()
    value = 0.5 * (N * LOG_2PI + logdet + quad)
    if not need_grad:
        return value, None

    Cm = torch.cholesky_solve(C.T, L_M).T
    del C
    alpha2 = alpha ** 2
    active = ((D - noise ** 2) > 0).to(dt) if fitc else torch.zeros(N, device=dev, dtype=dt)
    q = 0.5 * (diag_Kinv - alpha2) * active

    # ---------------- pass 3: adjoint and length-scale contractions ----------------
    # Psi = df/dK_12 = 2 A Phi with Psi[:, nyst] -= A Phi A^T. The landmark-column
    # correction is folded in as -<A Phi A^T, H_k>, H_k = dK_11/dlog l_k, so only an
    # r x r accumulator is needed.
    APhiAt = torch.zeros(rank, rank, device=dev, dtype=dt)
    acc = torch.zeros(n_l, device=dev, dtype=dt)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xb = Xs[i:j]
        Kb = _kern(Xm, Xb, sigma_f)
        Wb = torch.linalg.solve_triangular(L_11, Kb, upper=False)
        Ab = torch.linalg.solve_triangular(L_11.T, Wb, upper=True)
        Db, ab = D[i:j], alpha[i:j]
        AKib = Ab / Db - Cm @ (Wb / Db)                    # A K_tilde^-1, block
        del Wb
        APhib = 0.5 * (AKib - torch.outer(a, ab)) - Ab * q[i:j]
        del AKib
        APhiAt += APhib @ Ab.T
        del Ab
        APhib *= Kb                                        # Psi_b * K_12 (without the 2)
        for k in range(n_l):
            Sb = _S(Xm, Xb, k)
            acc[k] += (APhib * Sb).sum()
            del Sb
        del Kb, APhib

    grads = torch.zeros(n_l + 2, device=dev, dtype=dt)
    for k in range(n_l):
        H_k = _S(Xm, Xm, k) * K_11                         # dK_11/dlog l_k
        grads[k] = 2.0 * acc[k] - (APhiAt * H_k).sum()
        del H_k

    tr_sf = 2.0 * (N - (diag_Kinv * D).sum())
    q_sf = 2.0 * (quad - (D * alpha2).sum())
    dD = active * 2.0 * (D - noise ** 2)
    grads[n_l] = 0.5 * ((tr_sf + (diag_Kinv * dD).sum()) - (q_sf + (dD * alpha2).sum()))

    two_s2 = 2.0 * noise ** 2
    grads[n_l + 1] = 0.5 * two_s2 * (diag_Kinv.sum() - alpha2.sum())

    return value, grads


class _Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logtheta, X, y, nyst_idx, rank, fitc, jitter, chunk):
        with torch.no_grad():
            val, grad = nll_and_grad(logtheta, X, y, nyst_idx, rank, fitc, jitter, chunk)
        ctx.grad_theta = grad
        return val

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output * ctx.grad_theta,) + (None,) * 7


class NegativeLogLikelihood:
    """FITC objective, blocked, analytic gradients."""

    @staticmethod
    def apply(logtheta, X, y, nyst_idx, rank, jitter=1e-4, chunk=20000):
        return _Fn.apply(logtheta, X, y, nyst_idx, rank, True, jitter, chunk)
