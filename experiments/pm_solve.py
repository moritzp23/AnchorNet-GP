"""Full-kernel predictive solve in fp32 or fp64 with a per-iteration trace (Appendix B).

Solves (K + sigma_y^2 I) alpha = y by preconditioned CG with KeOps products, at the
hyperparameters written by run.py --dump_theta, and appends one row per iteration to
--out. Every --true-every iterations the true residual ||y - A alpha_k|| / ||y|| and the
test RMSE are recomputed. The preconditioner uses its own AnchorNet landmarks
(--precond-rank of them). With --checkpoint the CG state is saved every --ckpt-every
iterations and a later call with the same path resumes from it; the clock continues
from the checkpoint.

    python experiments/pm_solve.py --theta results/theta/buzz_r250_s73.json \
        --dtype fp64 --precond-rank 1000 --max-iter 100 --out results/my_traces.csv
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
ap.add_argument('--theta', required=True, help='JSON written by run.py --dump_theta')
ap.add_argument('--dtype', default='fp64', choices=['fp32', 'fp64'])
ap.add_argument('--precond-rank', type=int, default=1000, help='0 disables it')
ap.add_argument('--max-iter', type=int, default=100)
ap.add_argument('--true-every', type=int, default=5)
ap.add_argument('--checkpoint', default='')
ap.add_argument('--ckpt-every', type=int, default=10)
ap.add_argument('--prealloc-frac', type=float, default=0.9,
                help='reserve this fraction of the GPU so that no other job can start '
                     'on it during a long solve (0 disables)')
ap.add_argument('--gpu', default='0')
ap.add_argument('--out', required=True)
args = ap.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
_cache = os.path.join(ROOT, '.keops_cache')
os.makedirs(_cache, exist_ok=True)
for _v in ('PYKEOPS_BUILD_DIR', 'KEOPS_CACHE_DIR', 'KEOPS_BUILD_DIR', 'XDG_CACHE_HOME'):
    os.environ[_v] = _cache

import numpy as np  # noqa: E402
import torch  # noqa: E402
from pykeops.torch import LazyTensor  # noqa: E402
from bayesian_benchmarks.data import get_regression_data  # noqa: E402

from angp.anchor_net import get_anchor_net_indices  # noqa: E402
from uci import DATASETS  # noqa: E402

DEV = torch.device('cuda')
dt = {'fp32': torch.float32, 'fp64': torch.float64}[args.dtype]

th = json.load(open(args.theta))
DS, RANK, SPLIT = th['dataset'], int(th['rank']), int(th['split'])
SF, SY = float(th['sigma_f']), float(th['sigma_y'])
sy2 = SY ** 2

data = get_regression_data(DATASETS[DS], split=SPLIT)
Y_test = data.Y_test.ravel()
print(f'{DS} rank {RANK} split {SPLIT}: N={data.X_train.shape[0]}, '
      f'sigma_f={SF:.4f}, sigma_y={SY:.5f}, {args.dtype}', flush=True)

X = torch.tensor(data.X_train, device=DEV, dtype=dt)
Y = torch.tensor(data.Y_train, device=DEV, dtype=dt).reshape(-1, 1)
ell = torch.tensor(th['ell'], device=DEV, dtype=dt)
Z = (X / ell).contiguous()
Zt = (torch.tensor(data.X_test, device=DEV, dtype=dt) / ell).contiguous()
N = Z.shape[0]


def K_lazy(A, B):
    ai, bj = LazyTensor(A[:, None, :]), LazyTensor(B[None, :, :])
    return SF ** 2 * (-((ai - bj) ** 2).sum(-1) / 2).exp()


K, Kt = K_lazy(Z, Z), K_lazy(Zt, Z)


def A_mul(v):
    return K @ v + sy2 * v


# one warm-up product (KeOps compiles here), then one timed product
p0 = torch.randn(N, 1, device=DEV, dtype=dt)
A_mul(p0)
torch.cuda.synchronize()
t = time.time()
A_mul(p0)
torch.cuda.synchronize()
mvm = time.time() - t
del p0

if args.prealloc_frac > 0:
    free, total = torch.cuda.mem_get_info()
    want = int(min(free - 2 ** 30, total * args.prealloc_frac))
    if want > 0:
        buf = torch.empty(want, dtype=torch.uint8, device=DEV)
        del buf                       # stays in torch's caching allocator

PR = args.precond_rank
P = None
if PR:
    idx = get_anchor_net_indices(torch.tensor(data.X_train, device=DEV, dtype=torch.float32),
                                 PR, r_fps=max(PR // 10, 1))
    Zm = Z[idx].contiguous()
    K11 = SF ** 2 * torch.exp(-0.5 * torch.cdist(Zm, Zm) ** 2)
    L11 = torch.linalg.cholesky(K11 + 1e-4 * SF ** 2 * torch.eye(PR, device=DEV, dtype=dt))
    # thin QR of [W^T; sigma I], filled in place to keep the memory at one N x PR copy
    Aug = torch.empty(N + PR, PR, device=DEV, dtype=dt)
    for i in range(0, N, 50000):
        j = min(i + 50000, N)
        Aug[i:j, :] = torch.linalg.solve_triangular(
            L11, SF ** 2 * torch.exp(-0.5 * torch.cdist(Zm, Z[i:j]) ** 2), upper=False).T
    Aug[N:, :] = SY * torch.eye(PR, device=DEV, dtype=dt)
    del K11, L11
    Q, _ = torch.linalg.qr(Aug)
    del Aug
    Q1 = Q[:N, :]

    def P(v):
        return (v - Q1 @ (Q1.T @ v)) / sy2

bn = float(torch.sqrt((Y * Y).sum(dtype=torch.float64)))
bz = float((Y * (P(Y) if P else Y)).sum(dtype=torch.float64))

ckpt = args.checkpoint or None
if ckpt:
    os.makedirs(os.path.dirname(os.path.abspath(ckpt)), exist_ok=True)
start, t_prev = 1, 0.0
if ckpt and os.path.exists(ckpt):
    st = torch.load(ckpt, map_location=DEV, weights_only=True)
    x, r, p, rz = st['x'].to(dt), st['r'].to(dt), st['p'].to(dt), st['rz']
    start, t_prev = st['it'] + 1, st.get('elapsed', 0.0)
    print(f'resuming from iteration {start}', flush=True)
else:
    x = torch.zeros_like(Y)
    r = Y.clone()
    z = P(r) if P else r
    p = z.clone()
    rz = (r * z).sum()

fields = ['dataset', 'rank', 'split', 'dtype', 'precond_rank', 'iter', 'relres_true',
          'relres_rec', 'relres_prec', 'rmse', 'elapsed_s', 'mvm_s']
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
new = not os.path.exists(args.out) or os.path.getsize(args.out) == 0
fh = open(args.out, 'a', newline='')
wr = csv.DictWriter(fh, fieldnames=fields)
if new:
    wr.writeheader()

t0 = time.time() - t_prev
for it in range(start, args.max_iter + 1):
    Ap = A_mul(p)
    a = rz / (p * Ap).sum()
    x = x + a * p
    r = r - a * Ap
    rec = float(torch.sqrt((r * r).sum(dtype=torch.float64))) / bn
    z = P(r) if P else r
    rzn64 = float((r * z).sum(dtype=torch.float64))
    prec = (rzn64 / bz) ** 0.5 if rzn64 > 0 else float('inf')

    true = rmse = ''
    if it % args.true_every == 0 or it == 1 or it == args.max_iter:
        tr = Y - A_mul(x)
        true = float(torch.sqrt((tr * tr).sum(dtype=torch.float64))) / bn
        rmse = float(np.sqrt(np.mean(
            ((Kt @ x).cpu().numpy().ravel().astype(np.float64) - Y_test) ** 2)))
        del tr
        print(f'  {it:4d}  true {true:.4e}  rec {rec:.4e}  P^-1 {prec:.4e}  '
              f'RMSE {rmse:.4f}  {time.time() - t0:7.0f}s', flush=True)

    wr.writerow(dict(dataset=DS, rank=RANK, split=SPLIT, dtype=args.dtype, precond_rank=PR,
                     iter=it, relres_rec=f'{rec:.6e}', relres_prec=f'{prec:.6e}',
                     relres_true=(f'{true:.6e}' if true != '' else ''),
                     rmse=(f'{rmse:.6f}' if rmse != '' else ''),
                     elapsed_s=f'{time.time() - t0:.1f}', mvm_s=f'{mvm:.2f}'))
    fh.flush()

    rzn = (r * z).sum()
    p = z + (rzn / rz) * p
    rz = rzn

    if ckpt and (it % args.ckpt_every == 0 or it == args.max_iter):
        torch.save({'x': x, 'r': r, 'p': p, 'rz': rz, 'it': it,
                    'elapsed': time.time() - t0}, ckpt + '.tmp')
        os.replace(ckpt + '.tmp', ckpt)

fh.close()
