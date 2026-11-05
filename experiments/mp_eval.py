"""Single objective evaluations in reduced precision at fitted hyperparameters (Section 7).

scan   For each theta JSON: kappa(M) of the fp32 matrix M = I + W D^{-1} W^T (eigenvalues
       in fp64), then one objective-and-gradient evaluation with all GEMMs in fp32 and
       with the gemm2 and gemm4 variants. Records whether the reduced-precision M
       factored, the relative NLL and gradient errors against fp32, and the median time
       of three evaluations. This gives Table 9 of the paper.

nll    The fp32 FITC negative log likelihood per training point at each theta JSON (the
       'final train NLL' row of Table 8).

repair One evaluation per theta JSON with the gemm2 and gemm4 variants, each with the fp32
       fallback and with the two modified Cholesky repairs of an indefinite M (shift and
       clamp, see LRGP_MP_REPAIR in angp/fitc_mp.py). Records the outcome (factored,
       repaired or fallback), the shift or eigenvalue change, and the relative NLL and
       gradient errors against fp32, also split into length scales, sigma_f and sigma.

    python experiments/mp_eval.py scan --out results/mp_scan.csv results/theta/3droad_r3000_s73.json
"""
import argparse
import csv
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('task', choices=['scan', 'nll', 'repair'])
ap.add_argument('theta', nargs='+')
ap.add_argument('--out', default='', help='CSV to append to (scan and repair)')
ap.add_argument('--gpu', default='0')
args = ap.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

import numpy as np  # noqa: E402
import torch  # noqa: E402
from bayesian_benchmarks.data import get_regression_data  # noqa: E402

from angp import fitc_mp  # noqa: E402
from angp.anchor_net import get_anchor_net_indices  # noqa: E402
from angp.fitc import safe_cholesky  # noqa: E402
from uci import DATASETS  # noqa: E402

DEV, DT = torch.device('cuda'), torch.float32
U16 = 2.0 ** -11


def timed(fn):
    torch.cuda.synchronize()
    t = time.time()
    out = fn()
    torch.cuda.synchronize()
    return out, time.time() - t


def kappa_M(X, idx, rank, ell, sf, sy):
    """Extreme eigenvalues of M = I + W D^{-1} W^T, formed in fp32 in blocks."""
    Xs = X / ell
    Xm = Xs[idx].contiguous()
    L11, _ = safe_cholesky(fitc_mp._kern(Xm, Xm, sf), 1e-4 * sf ** 2)
    M = torch.eye(rank, device=DEV, dtype=DT)
    for i in range(0, X.shape[0], 20000):
        Wb = torch.linalg.solve_triangular(L11, fitc_mp._kern(Xm, Xs[i:i + 20000], sf),
                                           upper=False)
        Db = torch.clamp(sf ** 2 - (Wb * Wb).sum(0), min=0.0) + sy ** 2
        Vb = Wb / torch.sqrt(Db)
        M += Vb @ Vb.T
    ev = torch.linalg.eigvalsh(M.double())
    return float(ev[0]), float(ev[-1])


fields = ['dataset', 'rank', 'split', 'lam_min', 'lam_max', 'kappa', 'u16_kappa',
          'nll32', 't32', 'variant', 'factorized', 'failure', 'nll_rel', 'grad_rel',
          't_mixed', 'speedup']
if args.task == 'repair':
    fields = ['dataset', 'rank', 'variant', 'repair', 'outcome', 'tau', 'nll_rel', 'grad_rel',
              'grad_rel_ell', 'grad_rel_sf', 'grad_rel_sy']
wr = None
if args.task in ('scan', 'repair') and args.out:
    new = not os.path.exists(args.out)
    fh = open(args.out, 'a', newline='')
    wr = csv.DictWriter(fh, fieldnames=fields)
    if new:
        wr.writeheader()

cache = {}
for path in args.theta:
    th = json.load(open(path))
    ds, rk, sp = th['dataset'], int(th['rank']), int(th['split'])
    if (ds, sp) not in cache:
        cache.clear()
        torch.cuda.empty_cache()
        data = get_regression_data(DATASETS[ds], split=sp)
        cache[(ds, sp)] = (torch.tensor(data.X_train, device=DEV, dtype=DT),
                           torch.tensor(data.Y_train, device=DEV, dtype=DT).reshape(-1))
    X, y = cache[(ds, sp)]
    idx = get_anchor_net_indices(X, rk, r_fps=max(rk // 10, 1))
    ell = torch.tensor(th['ell'], device=DEV, dtype=DT)
    sf, sy = float(th['sigma_f']), float(th['sigma_y'])
    lt = torch.cat([torch.log(ell), torch.tensor([np.log(sf), np.log(sy)], device=DEV, dtype=DT)])

    fitc_mp.MODE, fitc_mp.REPAIR = 'none', 'fallback'
    v32, g32 = fitc_mp.nll_and_grad(lt, X, y, idx, rk)
    if args.task == 'nll':
        print(f'{path}: fp32 NLL per point {float(v32) / X.shape[0]:.5f}', flush=True)
        continue
    if args.task == 'repair':
        def rel(a, b):
            return float((a - b).norm() / b.norm())
        nl = len(th['ell'])
        for variant in ('gemm2', 'gemm4'):
            for repair in ('fallback', 'shift', 'clamp'):
                fitc_mp.MODE, fitc_mp.REPAIR = variant, repair
                fb0, rp0 = len(fitc_mp.FALLBACK_CALLS), len(fitc_mp.REPAIR_TAUS)
                row = dict(dataset=ds, rank=rk, variant=variant, repair=repair)
                v, g = fitc_mp.nll_and_grad(lt, X, y, idx, rk)
                fb = len(fitc_mp.FALLBACK_CALLS) - fb0
                rp = len(fitc_mp.REPAIR_TAUS) - rp0
                row['outcome'] = (fitc_mp.FALLBACK_REASONS[-1] + '->fp32' if fb
                                  else 'repaired' if rp else 'factored')
                row['tau'] = fitc_mp.REPAIR_TAUS[-1] if rp else ''
                if not fb and bool(torch.isfinite(g).all()):
                    row.update(nll_rel=abs(float(v - v32)) / abs(float(v32)), grad_rel=rel(g, g32),
                               grad_rel_ell=rel(g[:nl], g32[:nl]),
                               grad_rel_sf=rel(g[nl:nl + 1], g32[nl:nl + 1]),
                               grad_rel_sy=rel(g[nl + 1:], g32[nl + 1:]))
                print({k: (f'{x:.3g}' if isinstance(x, float) else x) for k, x in row.items()},
                      flush=True)
                if wr:
                    wr.writerow(row)
                    fh.flush()
        fitc_mp.REPAIR = 'fallback'
        continue

    with torch.no_grad():
        lmin, lmax = kappa_M(X, idx, rk, ell, sf, sy)
    t32 = float(np.median([timed(lambda: fitc_mp.nll_and_grad(lt, X, y, idx, rk))[1]
                           for _ in range(3)]))
    base = dict(dataset=ds, rank=rk, split=sp, lam_min=lmin, lam_max=lmax,
                kappa=lmax / lmin, u16_kappa=U16 * lmax / lmin, nll32=float(v32), t32=t32)
    for variant in ('gemm2', 'gemm4'):
        fitc_mp.MODE = variant
        row = dict(base, variant=variant)
        try:
            fitc_mp.nll_and_grad(lt, X, y, idx, rk)               # warm-up
            fb0, ts = len(fitc_mp.FALLBACK_CALLS), []
            for _ in range(3):
                (v, g), t = timed(lambda: fitc_mp.nll_and_grad(lt, X, y, idx, rk))
                ts.append(t)
            fb = len(fitc_mp.FALLBACK_CALLS) - fb0
            ok = fb == 0 and bool(torch.isfinite(v)) and bool(torch.isfinite(g).all())
            row.update(factorized=int(fb == 0),
                       failure=fitc_mp.FALLBACK_REASONS[-1] if fb else '',
                       nll_rel=abs(float(v - v32)) / abs(float(v32)) if ok else '',
                       grad_rel=float((g - g32).norm() / g32.norm()) if ok else '',
                       t_mixed=float(np.median(ts)) if ok else '',
                       speedup=t32 / float(np.median(ts)) if ok else '')
        except Exception as e:                       # any other numerical failure
            row.update(factorized=0, nll_rel=repr(e)[:60])
        print({k: (f'{x:.4g}' if isinstance(x, float) else x) for k, x in row.items()},
              flush=True)
        if wr:
            wr.writerow(row)
            fh.flush()
