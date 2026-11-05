"""Predictive means: the full-kernel mean (dense or KeOps CG) and the low-rank FITC mean."""
import torch

from .fitc import safe_cholesky


def kernel(X1, X2, l, sigma_f):
    """Squared-exponential kernel; l is a scalar or a length-d vector (ARD)."""
    return sigma_f ** 2 * torch.exp(-0.5 * torch.pow(torch.cdist(X1 / l, X2 / l, p=2), 2))


def kernel_keops(X1, X2, l, sigma_f):
    """The same kernel as a KeOps LazyTensor, for matrix-free products."""
    from pykeops.torch import LazyTensor
    x1_i = LazyTensor((X1 / l).contiguous()[:, None, :])
    x2_j = LazyTensor((X2 / l).contiguous()[None, :, :])
    D_ij = ((x1_i - x2_j) ** 2).sum(-1)
    return sigma_f ** 2 * (-D_ij / 2).exp()


def nystrom_preconditioner(X, nyst_idx, l, sigma_f, sigma_y, rank, chunk=50000):
    """P^{-1} for P = W^T W + sigma_y^2 I, built from the landmarks nyst_idx.

    Applied through a thin QR of [W^T; sigma_y I_r] = Q R with Q1 = Q[:N], which gives
    P^{-1} v = (v - Q1 Q1^T v) / sigma_y^2 without forming the Gram matrix W W^T.
    """
    Xs = X / l
    Xm = Xs[nyst_idx].contiguous()
    K_11 = sigma_f ** 2 * torch.exp(-0.5 * torch.cdist(Xm, Xm) ** 2)
    L_11, _ = safe_cholesky(K_11, 1e-4 * float(sigma_f) ** 2)
    N = Xs.shape[0]
    Wt = torch.empty(N, rank, device=X.device, dtype=X.dtype)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        K_12 = sigma_f ** 2 * torch.exp(-0.5 * torch.cdist(Xm, Xs[i:j]) ** 2)
        Wt[i:j, :] = torch.linalg.solve_triangular(
            L_11.mT, K_12.mT, upper=True, left=False)
    sig = sigma_y * torch.eye(rank, device=X.device, dtype=X.dtype)
    Q, _ = torch.linalg.qr(torch.cat([Wt, sig], dim=0))
    del Wt, sig
    Q1 = Q[:N, :].contiguous()
    del Q

    def apply(v):
        return (v - Q1 @ (Q1.T @ v)) / sigma_y ** 2
    return apply


def _dot64(u, v):
    """u . v accumulated in float64, returned as a Python float."""
    return float((u * v).sum(dtype=torch.float64))


def keops_cg(K, b, alpha, tol=1e-5, max_iter=200, precond=None, true_every=10):
    """Preconditioned CG for (K + alpha I) x = b with an iteration cap.

    Every `true_every` iterations the true residual ||b - A x_k|| / ||b|| is recomputed,
    and the iterate with the smallest true residual is returned. In fp32 the recurrence
    residual and the preconditioned residual keep decreasing after the true residual has
    stagnated (Appendix B of the paper), so neither is used for stopping or selection.

    Returns (x, iterations, best true residual, preconditioned residual at that iterate).
    """
    x = torch.zeros_like(b)
    r = b.clone()
    z = precond(r) if precond else r
    rz = _dot64(r, z)
    p = z.clone()
    bnorm = float(torch.sqrt((b * b).sum(dtype=torch.float64)))
    bz = _dot64(b, precond(b)) if precond else float((b * b).sum(dtype=torch.float64))
    best_true, best_x, best_it, best_rel = float('inf'), x.clone(), 0, float('inf')
    rel = float('inf')
    true_rel = float('inf')
    used = 0
    for used in range(1, max_iter + 1):
        Ap = K @ p + alpha * p
        denom = _dot64(p, Ap)
        if denom <= 0:
            break
        a = rz / denom
        x = x + a * p
        r = r - a * Ap
        z = precond(r) if precond else r
        rz_new = _dot64(r, z)
        rel = (rz_new / bz) ** 0.5 if rz_new > 0 else float('inf')

        checkpoint = (used % true_every == 0) or (used == max_iter) or (rz_new <= 0)
        if checkpoint:
            tr = b - (K @ x + alpha * x)
            true_rel = float(torch.sqrt((tr * tr).sum(dtype=torch.float64))) / bnorm
            del tr
            if true_rel < best_true:
                best_true, best_it, best_rel = true_rel, used, rel
                best_x.copy_(x)
        if (checkpoint and true_rel < tol) or rz_new <= 0:
            break
        p = z + (rz_new / rz) * p
        rz = rz_new
    if best_it < used:
        print(f'--> predictive CG stopped improving after iter {best_it} '
              f'(true residual {best_true:.2e} -> {true_rel:.2e}); returning the '
              f'iterate at {best_it}')
    return best_x, used, best_true, best_rel


def posterior_keops(X_s, X_train, Y_train, l, sigma_f, sigma_y, tol=1e-5, max_iter=200,
                    nyst_idx=None, precond_rank=0):
    """Full-kernel predictive mean by preconditioned KeOps CG.

    The preconditioner uses the first min(precond_rank, r) landmarks of the model.
    """
    K = kernel_keops(X_train, X_train, l, sigma_f)
    P = None
    if precond_rank and nyst_idx is not None:
        P = nystrom_preconditioner(X_train, nyst_idx[:precond_rank], l, sigma_f,
                                   sigma_y, min(precond_rank, len(nyst_idx)))
    mu, iters, relres, relres_P = keops_cg(K, Y_train, sigma_y ** 2, tol, max_iter, P)
    K_s = kernel_keops(X_s, X_train, l, sigma_f)
    return K_s @ mu, iters, relres, relres_P


def posterior_dense(X_s, X_train, Y_train, l, sigma_f, sigma_y):
    """Full-kernel predictive mean with a dense solve (small datasets)."""
    n = len(X_train)
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * torch.eye(
        n, device=X_train.device, dtype=X_train.dtype)
    return kernel(X_train, X_s, l, sigma_f).T @ torch.linalg.solve(K, Y_train)


def posterior_var_dense(X_s, X_train, l, sigma_f, sigma_y, chunk=4096):
    """Variance of the latent full-kernel GP at X_s, k(x,x) - k_x^T (K + sigma^2 I)^-1 k_x.

    Uses a dense Cholesky factor of the n x n matrix, so it is meant for the same sizes
    as posterior_dense.
    """
    n = len(X_train)
    K = kernel(X_train, X_train, l, sigma_f) + sigma_y ** 2 * torch.eye(
        n, device=X_train.device, dtype=X_train.dtype)
    L, _ = safe_cholesky(K, 0.0)
    del K
    out = []
    for i in range(0, len(X_s), chunk):
        V = torch.linalg.solve_triangular(L, kernel(X_train, X_s[i:i + chunk], l, sigma_f),
                                          upper=False)
        out.append(sigma_f ** 2 - (V * V).sum(0))
    return torch.clamp(torch.cat(out), min=0.0)


def _tri_solve_rn(L, B):
    """Solve L W = B for a wide B by asking cuBLAS for the transposed problem."""
    return torch.linalg.solve_triangular(L.mT, B.mT, upper=True, left=False).mT


def posterior_mean_lr(X_s, X_train, Y_train, nyst_idx, l, sigma_f, sigma_y,
                      return_var=False):
    """Predictive mean (and optionally variance) of the low-rank FITC model.

    With L = chol(K_mm), W = L^{-1} K_mn, D the FITC diagonal and A = I + W D^{-1} W^T,
    the mean is K_sm L^{-T} A^{-1} W D^{-1} y. The r x r matrix A and the right-hand side
    are accumulated over blocks of training points, so W is never held at r x N.

    The variance of the latent function is the FITC predictive variance
    sigma_f^2 - ||V_s||^2 + ||L_A^{-1} V_s||^2 per test point, with V_s = L^{-1} K_ms;
    the first two terms are the part of the prior variance the landmarks cannot explain.
    With return_var the function returns (mean, variance).
    """
    dev, dt = X_train.device, X_train.dtype
    X_m = X_train[nyst_idx]
    r = X_m.shape[0]

    K_mm = kernel(X_m, X_m, l, sigma_f)
    K_sm = kernel(X_s, X_m, l, sigma_f)
    L, _ = safe_cholesky(K_mm, 1e-6)

    N = X_train.shape[0]
    A = torch.eye(r, device=dev, dtype=dt)
    rhs = torch.zeros(r, 1, device=dev, dtype=dt)
    chunk = max(1, min(N, int(2e8 // max(r, 1))))
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        W_b = _tri_solve_rn(L, kernel(X_m, X_train[i:j], l, sigma_f))
        rootd_b = torch.sqrt(
            torch.clamp(sigma_f ** 2 - (W_b * W_b).sum(0), min=0.0) + sigma_y ** 2)
        V_b = W_b / rootd_b
        A += V_b @ V_b.T
        rhs += V_b @ (Y_train[i:j] / rootd_b.unsqueeze(-1))
        del W_b, V_b

    L_A, _ = safe_cholesky(A, 0.0)
    t = torch.cholesky_solve(rhs, L_A)
    mean = K_sm @ torch.linalg.solve_triangular(L.T, t, upper=True)
    if not return_var:
        return mean
    V = _tri_solve_rn(L, K_sm.T)                               # r x n_s
    Z = _tri_solve_rn(L_A, V)
    var = sigma_f ** 2 - (V * V).sum(0) + (Z * Z).sum(0)
    return mean, torch.clamp(var, min=0.0)
