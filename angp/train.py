"""Hyperparameter fit of ANGP (Section 5 and Appendix A of the paper).

The objective is the FITC negative log marginal likelihood minus the log priors of
Maddox et al. (2022) on l, sigma_f^2 and sigma_y^2. The length scales start at the
median pairwise distance of 2000 random inputs, sigma_f at std(y) and sigma_y at the
mode of its prior, sqrt(2). Adam runs for a fixed number of iterations with the learning
rate lr * (0.02 + 0.98 * (1 + cos(pi t / (T - 1))) / 2) and gradients clipped to norm 10.
"""
import math
import time

import torch
from torch.distributions import Gamma
from torch.nn.functional import softplus

from . import fitc_blocked, fitc_manual
from .anchor_net import get_anchor_net_indices


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()


def _inv_softplus(val):
    val = torch.clamp(val, min=1e-6)
    return val + torch.log(-torch.expm1(-val))


def median_heuristic_lengthscale(X, sample_size=2000):
    """Median pairwise distance of a random subsample of the inputs."""
    idx = torch.randperm(X.shape[0], device=X.device)[:min(sample_size, X.shape[0])]
    return torch.median(torch.pdist(X[idx], p=2))


def fit(X, Y, rank, r_fps=None, n_iter=100, lr=0.2, grad_clip=10.0, noise_floor=1e-4,
        grads='adaptive', seed=None, empty_cache=True, verbose=True):
    """Fit the ARD length scales, sigma_f and sigma_y of the FITC model.

    X is an (n, d) and Y an (n, 1) float32 tensor on the GPU. grads is 'adaptive',
    'manual' (unblocked), 'blocked' or 'mp' (blocked with fp16 GEMMs, see fitc_mp.py).
    'adaptive' times one evaluation of the unblocked and the blocked implementation and
    uses the unblocked one only if it is at least 20% faster.

    Returns a dict with the fitted 'ell' (d,), 'sigma_f', 'sigma_y' (0-d tensors), the
    landmark indices 'nyst_idx', the gradient path 'grads', 'final_nll' (per point) and
    the times 'anchornet_time', 'train_time' and 'fit_time' in seconds.
    """
    say = print if verbose else (lambda *a, **k: None)
    if seed is not None:
        torch.manual_seed(seed)
    dev, dt = X.device, X.dtype
    N, d = X.shape
    r_fps = max(rank // 10, 1) if r_fps is None else r_fps

    t0 = _sync()
    nyst_idx = get_anchor_net_indices(X, rank, r_fps=r_fps)
    anchornet_time = _sync() - t0
    say(f'--> AnchorNet: {rank} landmarks, {r_fps} anchors, {anchornet_time:.2f}s')

    def gamma(a, b):
        return Gamma(torch.tensor(a, device=dev, dtype=dt), torch.tensor(b, device=dev, dtype=dt))
    prior_l, prior_sf2, prior_sy2 = gamma(3.0, 6.0), gamma(2.0, 0.15), gamma(1.1, 0.05)

    # optimizer variables: d raw length scales, raw sigma_f, raw sigma_y (softplus)
    def unpack(raw):
        return softplus(raw[:d]), softplus(raw[d]), softplus(raw[d + 1]) + noise_floor

    def log_theta_of(raw):
        ell, sf, sy = unpack(raw)
        return torch.log(torch.cat([ell, sf.reshape(1), sy.reshape(1)]))

    def make_loss(mod):
        def loss_fn(raw):
            ell, sf, sy = unpack(raw)
            val = mod.NegativeLogLikelihood.apply(log_theta_of(raw), X, Y, nyst_idx, rank)
            return val - (prior_l.log_prob(ell).sum() + prior_sf2.log_prob(sf ** 2)
                          + prior_sy2.log_prob(sy ** 2))
        return loss_fn

    start = _sync()
    l0 = median_heuristic_lengthscale(X)
    sf0 = torch.clamp(Y.std(), min=1e-3)
    sy0 = torch.sqrt(torch.tensor((1.1 - 1) / 0.05, device=dev, dtype=dt))
    x_init = torch.cat([_inv_softplus(l0).reshape(1).expand(d), _inv_softplus(sf0).reshape(1),
                        _inv_softplus(torch.clamp(sy0 - noise_floor, min=1e-6)).reshape(1)])
    say(f'--> init: l0 = {l0.item():.4f}, sigma_f0 = {sf0.item():.4f}, '
        f'sigma_y0 = {sy0.item():.4f}')

    if grads == 'adaptive':
        # One warm-up and one timed evaluation of each implementation. The probe is
        # evaluated with the optimizer vector x_init passed directly as log(theta), as in
        # the runs reported in the paper; only its run time is used.
        def probe(mod):
            try:
                for _ in range(2):
                    th = x_init.clone().detach().requires_grad_(True)
                    t = _sync()
                    mod.NegativeLogLikelihood.apply(th, X, Y, nyst_idx, rank).backward()
                    t = _sync() - t
                return t
            except torch.OutOfMemoryError:
                return float('inf')
            finally:
                if empty_cache:
                    torch.cuda.empty_cache()
        t_m, t_b = probe(fitc_manual), probe(fitc_blocked)
        grads = 'manual' if t_m < 0.8 * t_b else 'blocked'
        if empty_cache:
            torch.cuda.empty_cache()
        if X.is_cuda:
            torch.cuda.reset_peak_memory_stats()
        say(f'--> adaptive gradients: unblocked {t_m:.3f}s, blocked {t_b:.3f}s -> {grads}')
    if grads == 'mp':
        from . import fitc_mp                 # reads LRGP_MP_MODE on import
        mod = fitc_mp
    else:
        mod = {'manual': fitc_manual, 'blocked': fitc_blocked}[grads]
    loss_fn = make_loss(mod)

    xk = x_init.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([xk], lr=lr)
    for i in range(n_iter):
        opt.zero_grad()
        loss = loss_fn(xk) / N
        loss.backward()
        for grp in opt.param_groups:
            grp['lr'] = lr * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * i / max(n_iter - 1, 1))))
        gnorm = torch.nn.utils.clip_grad_norm_([xk], grad_clip).item()
        opt.step()
        if i % 10 == 0 or i == n_iter - 1:
            ell, sf, sy = unpack(xk.detach())
            say(f'    Adam iter {i}: NLL/n = {loss.item():.4f} | grad norm = {gnorm:.6f}'
                f' | median l = {float(ell.median()):.4g}, sf = {float(sf):.4g}, '
                f'sy = {float(sy):.4g}')
    theta = xk.detach()

    # The final iterate is the result, unless the objective ended above its starting
    # value; then the starting point is kept. This did not happen in the paper's runs.
    final_nll = float(loss_fn(theta))
    if not math.isfinite(final_nll) or final_nll > float(loss_fn(x_init)) + 1e-3:
        say('--> objective increased during optimization; keeping the initial point')
        theta = x_init.clone()

    train_time = _sync() - start
    ell, sf, sy = unpack(theta)
    say(f'--> fit time {anchornet_time + train_time:.2f}s')
    return dict(ell=ell, sigma_f=sf, sigma_y=sy, nyst_idx=nyst_idx, grads=grads,
                final_nll=final_nll / N, anchornet_time=anchornet_time,
                train_time=train_time, fit_time=anchornet_time + train_time)
