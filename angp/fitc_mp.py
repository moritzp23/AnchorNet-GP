"""Blocked FITC objective with selected GEMMs in reduced precision (Section 7 of the paper).

The same computation as fitc_blocked.py. The large GEMMs take fp16 inputs and accumulate
in fp32; kernel blocks, triangular solves and both Cholesky factorizations stay in fp32.
The variant is chosen with the environment variable LRGP_MP_MODE:

    none    everything in fp32, identical to fitc_blocked.py
    gemm2   the GEMM that forms M (pass 1) and the gradient GEMM C += (A/D) W^T (pass 2)
    gemm4   as gemm2, and also Cm (W/D) and (A Phi) A^T in pass 3. All four GEMMs scale
            their inputs by powers of two first, so the fp16 output cannot overflow.

If the reduced-precision M does not factor without jitter, pass 1 is repeated in fp32
for that call (a fallback). FP16_CALLS counts the calls, FALLBACK_CALLS lists the calls
that fell back and FALLBACK_REASONS says whether M overflowed or was indefinite. With
LRGP_MP_DIAG=<file.csv> each call also forms M in fp32 and writes the spectra of both
matrices to that file.

LRGP_MP_REPAIR replaces the fallback for an indefinite (but finite) M by a modified
Cholesky factorization, to test whether repairing the factorization is enough:

    fallback  (default) repeat pass 1 in fp32
    shift     factor M + tau I, tau = 1e-3, 1e-2, ... until the factorization succeeds
    clamp     factor Q max(Lambda, 1) Q^T, which uses lambda_min(M) >= 1 of the exact M
              (eigendecomposition in fp32)

An overflowed M still falls back. REPAIR_CALLS lists the repaired calls and REPAIR_TAUS
the shift (shift) or the largest eigenvalue change (clamp) of each.
"""
import csv
import os

import numpy as np
import torch

from .fitc import safe_cholesky

LOG_2PI = np.log(2 * np.pi)

MODE = os.environ.get('LRGP_MP_MODE', 'gemm2')
if MODE not in ('none', 'gemm2', 'gemm4'):
    raise ValueError(f'LRGP_MP_MODE={MODE!r}; expected none, gemm2 or gemm4')
LOW = torch.float16
DIAG = os.environ.get('LRGP_MP_DIAG', '')

FP16_CALLS = 0
FALLBACK_CALLS = []
FALLBACK_REASONS = []
REPAIR = os.environ.get('LRGP_MP_REPAIR', 'fallback')
if REPAIR not in ('fallback', 'shift', 'clamp'):
    raise ValueError(f'LRGP_MP_REPAIR={REPAIR!r}; expected fallback, shift or clamp')
REPAIR_CALLS = []
REPAIR_TAUS = []


def _repair(M):
    """Modified Cholesky factor of an indefinite M (see LRGP_MP_REPAIR)."""
    I = torch.eye(M.shape[0], device=M.device, dtype=M.dtype)
    if REPAIR == 'shift':
        tau = 1e-3
        while True:
            L, info = torch.linalg.cholesky_ex(M + tau * I)
            if int(info) == 0 and bool(torch.isfinite(L).all()):
                return L, tau
            tau *= 10.0
    lam, Q = torch.linalg.eigh(M)
    lam_c = torch.clamp(lam, min=1.0)
    L = torch.linalg.cholesky(((Q * lam_c) @ Q.T).to(M.dtype))
    return L, float((lam_c - lam).max())


def _kern(Xa, Xb, sigma_f):
    return sigma_f ** 2 * torch.exp(-0.5 * torch.pow(torch.cdist(Xa, Xb, p=2), 2))


def _diag_row(logtheta, M16, M32, ok):
    e32 = torch.linalg.eigvalsh(M32.double())
    e16 = torch.linalg.eigvalsh(M16.double())
    th = torch.exp(logtheta.detach()).tolist()
    row = dict(call=FP16_CALLS, ok16=int(ok), lam_min32=float(e32[0]), lam_max32=float(e32[-1]),
               kappa32=float(e32[-1] / e32[0]), lam_min16=float(e16[0]),
               err_rel=float((M16 - M32).norm() / M32.norm()),
               sigma_f=th[-2], sigma_y=th[-1], ell=' '.join('%.5g' % v for v in th[:-2]))
    new = not os.path.exists(DIAG)
    os.makedirs(os.path.dirname(os.path.abspath(DIAG)), exist_ok=True)
    with open(DIAG, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def _pow2(A):
    """Power of two that brings max|A| to at most 1. Kept on the device to avoid a sync."""
    m = A.abs().amax().clamp_min(torch.finfo(A.dtype).tiny)
    return torch.exp2(-torch.ceil(torch.log2(m)))


def _gemm(A, B):
    """A @ B with fp16 inputs, accumulated and returned in fp32.

    In gemm4 the inputs are first scaled to max|.| <= 1, so every entry of the product is
    at most the inner dimension (20000 < 65504) and cannot overflow.
    """
    if MODE != 'gemm4':
        return (A.to(LOW) @ B.to(LOW)).to(A.dtype)
    sa, sb = _pow2(A), _pow2(B)
    return (((A * sa).to(LOW) @ (B * sb).to(LOW)).to(A.dtype)) / (sa * sb)


def nll_and_grad(logtheta, X, y, nyst_idx, rank, fitc=True, jitter=1e-4,
                 chunk=20000, need_grad=True):
    global FP16_CALLS
    dev, dt = X.device, X.dtype
    reduced_on = MODE in ('gemm2', 'gemm4')
    n_l = logtheta.numel() - 2
    ell = torch.exp(logtheta[:n_l])
    sigma_f = torch.exp(logtheta[n_l])
    noise = torch.exp(logtheta[n_l + 1])
    N = X.shape[0]
    y = y.reshape(-1)

    Xs = X / ell
    Xm = Xs[nyst_idx, :].contiguous()
    K_11 = _kern(Xm, Xm, sigma_f)
    L_11, _ = safe_cholesky(K_11, jitter * sigma_f ** 2)

    def _S(Xa, Xb, k):
        if n_l == 1:
            return torch.pow(torch.cdist(Xa, Xb, p=2), 2)
        return (Xa[:, k:k + 1] - Xb[:, k].unsqueeze(0)) ** 2

    # ---------------- pass 1: M, u, D ----------------
    # Jitter is not used on a reduced-precision M that lost definiteness to rounding:
    # it would hide the error. That call repeats pass 1 in fp32 instead.
    for reduced in ((True, False) if reduced_on else (False,)):
        M = torch.eye(rank, device=dev, dtype=dt)
        if reduced and DIAG:
            M_ref = torch.eye(rank, device=dev, dtype=dt)
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
            M += _gemm(Vb, Vb.T) if reduced else Vb @ Vb.T
            if reduced and DIAG:
                M_ref += Vb @ Vb.T
            u += Wb @ (y[i:j] / Db)
            del Kb, Wb, Vb, Db

        if not reduced:
            L_M, _ = safe_cholesky(M, 0.0)
            break
        FP16_CALLS += 1
        L_M, info = torch.linalg.cholesky_ex(M)
        ok = int(info) == 0 and bool(torch.isfinite(L_M).all())
        if DIAG:
            _diag_row(logtheta, M, M_ref, ok)
            del M_ref
        if ok:
            break
        if REPAIR != 'fallback' and bool(torch.isfinite(M).all()):
            L_M, tau = _repair(M)
            REPAIR_CALLS.append(FP16_CALLS)
            REPAIR_TAUS.append(tau)
            print(f'    [mp] reduced-precision M indefinite, repaired ({REPAIR}, tau {tau:.3g}, '
                  f'call {FP16_CALLS}, {len(REPAIR_TAUS)} repairs so far)', flush=True)
            break
        FALLBACK_CALLS.append(FP16_CALLS)
        FALLBACK_REASONS.append('indefinite' if bool(torch.isfinite(M).all()) else 'overflow')
        print(f'    [mp] reduced-precision M {FALLBACK_REASONS[-1]}, pass 1 redone in {dt} '
              f'(call {FP16_CALLS}, {len(FALLBACK_CALLS)} fallbacks so far)', flush=True)
    v = torch.cholesky_solve(u.unsqueeze(1), L_M).squeeze(1)
    logdet = logdet_D + 2.0 * torch.log(torch.diagonal(L_M)).sum()

    # ---------------- pass 2: alpha, C, a, diag(K^-1) ----------------
    alpha = torch.empty(N, device=dev, dtype=dt)
    diag_Kinv = torch.empty(N, device=dev, dtype=dt) if need_grad else None
    if need_grad:
        C = torch.zeros(rank, rank, device=dev, dtype=dt)
        a = torch.zeros(rank, device=dev, dtype=dt)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Kb = _kern(Xm, Xs[i:j], sigma_f)
        Wb = torch.linalg.solve_triangular(L_11, Kb, upper=False)
        Db = D[i:j]
        ab = (y[i:j] - Wb.T @ v) / Db
        alpha[i:j] = ab
        if need_grad:
            Ab = torch.linalg.solve_triangular(L_11.T, Wb, upper=True)
            C += _gemm(Ab / Db, Wb.T) if reduced_on else (Ab / Db) @ Wb.T
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

    # ---------------- pass 3 ----------------
    APhiAt = torch.zeros(rank, rank, device=dev, dtype=dt)
    acc = torch.zeros(n_l, device=dev, dtype=dt)

    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        Xb = Xs[i:j]
        Kb = _kern(Xm, Xb, sigma_f)
        Wb = torch.linalg.solve_triangular(L_11, Kb, upper=False)
        Ab = torch.linalg.solve_triangular(L_11.T, Wb, upper=True)
        Db, ab = D[i:j], alpha[i:j]
        AKib = Ab / Db - (_gemm(Cm, Wb / Db) if MODE == 'gemm4' else Cm @ (Wb / Db))
        del Wb
        APhib = 0.5 * (AKib - torch.outer(a, ab)) - Ab * q[i:j]
        del AKib
        APhiAt += _gemm(APhib, Ab.T) if MODE == 'gemm4' else APhib @ Ab.T
        del Ab
        APhib *= Kb
        for k in range(n_l):
            Sb = _S(Xm, Xb, k)
            acc[k] += (APhib * Sb).sum()
            del Sb
        del Kb, APhib

    grads = torch.zeros(n_l + 2, device=dev, dtype=dt)
    for k in range(n_l):
        H_k = _S(Xm, Xm, k) * K_11
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
    """FITC objective, blocked, with the GEMMs selected by LRGP_MP_MODE in fp16."""

    @staticmethod
    def apply(logtheta, X, y, nyst_idx, rank, jitter=1e-4, chunk=20000):
        return _Fn.apply(logtheta, X, y, nyst_idx, rank, True, jitter, chunk)
