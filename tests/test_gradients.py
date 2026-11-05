"""Check that the three FITC implementations agree (value and gradient) on random data.

    python tests/test_gradients.py

Compares the autograd reference (angp/fitc.py) with the analytic gradients, unblocked
(fitc_manual.py) and blocked with a small block size (fitc_blocked.py), in float64 on
the GPU, for an ARD kernel. Also checks the reference gradient against central finite
differences.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angp import fitc, fitc_blocked, fitc_manual  # noqa: E402
from angp.anchor_net import get_anchor_net_indices  # noqa: E402


def value_and_grad(nll, theta, X, y, idx, rank):
    th = theta.clone().requires_grad_(True)
    v = nll(th, X, y, idx, rank)
    v.backward()
    return float(v), th.grad.clone()


def main():
    torch.manual_seed(0)
    dev, dt = torch.device('cuda'), torch.float64
    n, d, rank = 3000, 4, 60
    X = torch.randn(n, d, device=dev, dtype=dt)
    y = torch.sin(X @ torch.randn(d, device=dev, dtype=dt)) + 0.1 * torch.randn(n, device=dev, dtype=dt)
    idx = get_anchor_net_indices(X, rank, r_fps=rank // 10)
    theta = torch.log(torch.tensor([1.5, 0.8, 2.0, 1.1, 0.9, 0.3], device=dev, dtype=dt))

    ref_v, ref_g = value_and_grad(fitc.NegativeLogLikelihood.apply, theta, X, y, idx, rank)
    impls = {
        'unblocked': fitc_manual.NegativeLogLikelihood.apply,
        'blocked': lambda *a: fitc_blocked.NegativeLogLikelihood.apply(*a, chunk=700),
    }
    ok = True
    for name, nll in impls.items():
        v, g = value_and_grad(nll, theta, X, y, idx, rank)
        ev, eg = abs(v - ref_v) / abs(ref_v), float((g - ref_g).norm() / ref_g.norm())
        ok &= ev < 1e-10 and eg < 1e-8
        print(f'{name:9s} value rel. err {ev:.1e}, gradient rel. err {eg:.1e}')

    h, fd = 1e-5, torch.empty_like(theta)
    for k in range(theta.numel()):
        e = torch.zeros_like(theta)
        e[k] = h
        with torch.no_grad():
            fd[k] = (fitc._nll(theta + e, X, y, idx, rank, True)
                     - fitc._nll(theta - e, X, y, idx, rank, True)) / (2 * h)
    efd = float((fd - ref_g).norm() / ref_g.norm())
    ok &= efd < 1e-6
    print(f'finite differences vs autograd: rel. err {efd:.1e}')
    print('OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
