"""Check the predictive variances against a direct float64 computation.

    python tests/test_variance.py

Runs the variance code in float64 on random data and compares it with the textbook
formulas, k(x,x) - k_x^T (K + sigma^2 I)^{-1} k_x for the full-kernel GP and
k(x,x) - Q(x,x) + k_xm Sigma k_mx, Sigma = (K_mm + K_mn D^{-1} K_nm)^{-1}, for FITC.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from angp.anchor_net import get_anchor_net_indices  # noqa: E402
from angp.predict import posterior_mean_lr, posterior_var_dense  # noqa: E402


def main():
    torch.manual_seed(0)
    dev, dt = torch.device('cuda'), torch.float64
    n, d, rank = 1500, 3, 120
    X = torch.rand(n, d, device=dev, dtype=dt) * 4 - 2
    Y = (torch.sin(2 * X[:, :1]) * X[:, 1:2]) + 0.1 * torch.randn(n, 1, device=dev, dtype=dt)
    Xs = torch.rand(200, d, device=dev, dtype=dt) * 4 - 2
    ell = torch.tensor([0.8, 1.2, 3.0], device=dev, dtype=dt)
    sf, sy = torch.tensor(1.3, device=dev, dtype=dt), torch.tensor(0.1, device=dev, dtype=dt)
    idx = get_anchor_net_indices(X, rank, r_fps=rank // 10)

    def k(A, B):
        return sf ** 2 * torch.exp(-0.5 * torch.cdist(A / ell, B / ell) ** 2)

    eye = torch.eye(n, device=dev, dtype=dt)
    ref_exact = sf ** 2 - (k(X, Xs) * torch.linalg.solve(k(X, X) + sy ** 2 * eye, k(X, Xs))).sum(0)
    Xm = X[idx]
    Kmm = k(Xm, Xm) + 1e-6 * torch.eye(rank, device=dev, dtype=dt)
    Kmn, Kms = k(Xm, X), k(Xm, Xs)
    D = sf ** 2 - (Kmn * torch.linalg.solve(Kmm, Kmn)).sum(0) + sy ** 2
    Sig = torch.linalg.inv(Kmm + Kmn @ (Kmn.T / D[:, None]))
    ref_fitc = sf ** 2 - (Kms * torch.linalg.solve(Kmm, Kms)).sum(0) + (Kms * (Sig @ Kms)).sum(0)

    v_exact = posterior_var_dense(Xs, X, ell, sf, sy)
    v_fitc = posterior_mean_lr(Xs, X, Y, idx, ell, sf, sy, return_var=True)[1]
    e1 = float((v_exact - ref_exact).abs().max() / ref_exact.abs().max())
    e2 = float((v_fitc - ref_fitc).abs().max() / ref_fitc.abs().max())
    print(f'full-kernel variance rel. err {e1:.1e}')
    print(f'FITC variance rel. err        {e2:.1e}')
    ok = e1 < 1e-8 and e2 < 1e-6
    print('OK' if ok else 'FAILED')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
