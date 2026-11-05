"""Fit ANGP on one dataset, rank and split, then evaluate both predictive means.

The defaults are the configuration of the paper: ARD squared-exponential kernel, FITC
objective, median-heuristic initialization, 100 Adam iterations with a cosine-decayed
learning rate (eta_0 = 0.2), gradient clipping at 10, no validation split, and r/10
AnchorNet anchors. A run appends one row to --csv; failed runs are recorded as well.

    python experiments/run.py --dataset bike --rank 100 --split 1 --csv results/my_runs.csv

Gradient paths (--grads): 'adaptive' times one objective-and-gradient evaluation of the
unblocked and the blocked implementation and uses the unblocked one only if it is at
least 20% faster; 'manual' and 'blocked' force one of them; 'mp' uses the blocked
implementation with reduced-precision GEMMs, selected by LRGP_MP_MODE (see
angp/fitc_mp.py).
"""
import argparse
import csv
import fcntl
import json
import os
import sys
import time
import traceback
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from uci import DATASETS  # noqa: E402

CSV_FIELDS = [
    'timestamp', 'tag', 'status', 'error', 'dataset', 'split', 'rank', 'r_fps',
    'N_train', 'N_test', 'd', 'grads', 'mp_mode', 'anchornet_time', 'train_time',
    'fit_time', 'eff_rank', 'l_opt', 'l_opt_min', 'l_opt_max', 'sigma_f_opt',
    'sigma_y_opt', 'final_nll', 'test_rmse_lr', 'predict_time_lr', 'test_rmse_exact',
    'predict_time_exact', 'cg_iters', 'cg_resid', 'mp_calls', 'mp_fallback_calls',
    'mp_repair', 'mp_repair_calls',
    'peak_mem_gb', 'gpu_name', 'argv',
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--dataset', required=True, choices=list(DATASETS))
    p.add_argument('--rank', type=int, required=True, help='number of landmarks r')
    p.add_argument('--r_fps', type=int, default=None,
                   help='AnchorNet anchors chosen by farthest point sampling (default r/10)')
    p.add_argument('--split', type=int, default=73,
                   help='bayesian_benchmarks split; the paper uses 1, 2, 3, 4 and 73')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--gpu', default='0', help='sets CUDA_VISIBLE_DEVICES')
    p.add_argument('--grads', default='adaptive',
                   choices=['adaptive', 'manual', 'blocked', 'mp'])
    p.add_argument('--adam_epochs', type=int, default=100)
    p.add_argument('--adam_lr', type=float, default=0.2)
    p.add_argument('--grad_clip', type=float, default=10.0)
    p.add_argument('--noise_floor', type=float, default=1e-4,
                   help='lower bound on sigma_y')
    p.add_argument('--max_dense_n', type=int, default=60000,
                   help='above this training size the full-kernel mean uses KeOps CG')
    p.add_argument('--precond_rank', type=int, default=200,
                   help='landmarks used for the CG preconditioner (0 disables it)')
    p.add_argument('--cg_max_iter', type=int, default=200)
    p.add_argument('--cg_tol', type=float, default=1e-5)
    p.add_argument('--skip_exact', action='store_true',
                   help='only evaluate the low-rank predictive mean')
    p.add_argument('--prealloc_frac', type=float, default=0.0,
                   help='reserve this fraction of free GPU memory at start-up, so that '
                        'other jobs cannot share the card during a timing run')
    p.add_argument('--dump_theta', default='',
                   help='write the fitted hyperparameters (full ARD vector) as JSON')
    p.add_argument('--csv', default=None, help='append the result row to this CSV')
    p.add_argument('--tag', default='')
    a = p.parse_args()
    if a.r_fps is None:
        a.r_fps = max(a.rank // 10, 1)
    return a


args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
_cache = os.path.join(ROOT, '.keops_cache')
os.makedirs(_cache, exist_ok=True)
for _v in ('PYKEOPS_BUILD_DIR', 'KEOPS_CACHE_DIR', 'KEOPS_BUILD_DIR', 'XDG_CACHE_HOME'):
    os.environ[_v] = _cache

import torch  # noqa: E402
from bayesian_benchmarks.data import get_regression_data  # noqa: E402

from angp import train  # noqa: E402
from angp.predict import posterior_dense, posterior_keops, posterior_mean_lr  # noqa: E402

DEVICE = torch.device('cuda')
DTYPE = torch.float32


def sync():
    torch.cuda.synchronize()
    return time.time()


def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))


def write_row(path, row):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction='ignore')
        if f.tell() == 0:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in CSV_FIELDS})
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def run(record):
    data = get_regression_data(DATASETS[args.dataset], split=args.split)
    torch.manual_seed(args.seed)
    X_train = torch.tensor(data.X_train, device=DEVICE, dtype=DTYPE)
    Y_train = torch.tensor(data.Y_train, device=DEVICE, dtype=DTYPE)
    X_test = torch.tensor(data.X_test, device=DEVICE, dtype=DTYPE)
    N, d = X_train.shape
    record.update(N_train=N, N_test=X_test.shape[0], d=d)
    print(f'--> {args.dataset}: train {tuple(X_train.shape)}, test {tuple(X_test.shape)}')

    res = train.fit(X_train, Y_train, args.rank, r_fps=args.r_fps, n_iter=args.adam_epochs,
                    lr=args.adam_lr, grad_clip=args.grad_clip, noise_floor=args.noise_floor,
                    grads=args.grads, empty_cache=args.prealloc_frac <= 0)
    for k in ('grads', 'final_nll', 'anchornet_time', 'train_time', 'fit_time'):
        record[k] = res[k]
    nyst_idx = res['nyst_idx']
    l_opt, sf_opt, sy_opt = res['ell'], res['sigma_f'], res['sigma_y']
    record.update(l_opt=float(l_opt.median()), l_opt_min=float(l_opt.min()),
                  l_opt_max=float(l_opt.max()), sigma_f_opt=float(sf_opt),
                  sigma_y_opt=float(sy_opt))
    print(f'--> sigma_f = {float(sf_opt):.4f}, sigma_y = {float(sy_opt):.4f}, '
          f'length scales {[round(float(v), 4) for v in l_opt]}')
    if args.grads == 'mp':
        from angp import fitc_mp
        record['mp_calls'] = fitc_mp.FP16_CALLS
        record['mp_fallback_calls'] = ' '.join(
            f'{c}:{r}' for c, r in zip(fitc_mp.FALLBACK_CALLS, fitc_mp.FALLBACK_REASONS))
        record['mp_repair'] = fitc_mp.REPAIR
        record['mp_repair_calls'] = ' '.join(
            f'{c}:{t:.3g}' for c, t in zip(fitc_mp.REPAIR_CALLS, fitc_mp.REPAIR_TAUS))
        print(f'--> reduced-precision M kept in {fitc_mp.FP16_CALLS - len(fitc_mp.FALLBACK_CALLS)}'
              f' of {fitc_mp.FP16_CALLS} objective calls'
              + (f', {len(fitc_mp.REPAIR_CALLS)} of them repaired ({fitc_mp.REPAIR})'
                 if fitc_mp.REPAIR_CALLS else ''))
    if args.dump_theta:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_theta)), exist_ok=True)
        with open(args.dump_theta, 'w') as f:
            json.dump({'dataset': args.dataset, 'rank': args.rank, 'split': args.split,
                       'ell': [float(v) for v in l_opt], 'sigma_f': float(sf_opt),
                       'sigma_y': float(sy_opt)}, f, indent=1)

    with torch.no_grad():
        # numerical rank of K_11 at the fitted hyperparameters
        Xm = (X_train / l_opt)[nyst_idx]
        ev = torch.linalg.eigvalsh((sf_opt ** 2 * torch.exp(-0.5 * torch.cdist(Xm, Xm) ** 2)).double())
        record['eff_rank'] = int((ev > 1e-6 * ev[-1]).sum())
        del Xm, ev

        t0 = sync()
        if args.skip_exact:
            pass
        elif N > args.max_dense_n:
            pm, it, res, _ = posterior_keops(X_test, X_train, Y_train, l_opt, sf_opt, sy_opt,
                                             args.cg_tol, args.cg_max_iter, nyst_idx,
                                             args.precond_rank)
            record.update(test_rmse_exact=rmse(pm.cpu().numpy(), data.Y_test),
                          cg_iters=it, cg_resid=res)
        else:
            pm = posterior_dense(X_test, X_train, Y_train, l_opt, sf_opt, sy_opt)
            record['test_rmse_exact'] = rmse(pm.cpu().numpy(), data.Y_test)
        if not args.skip_exact:
            record['predict_time_exact'] = sync() - t0

        t0 = sync()
        pm = posterior_mean_lr(X_test, X_train, Y_train, nyst_idx, l_opt, sf_opt, sy_opt)
        record['test_rmse_lr'] = rmse(pm.cpu().numpy(), data.Y_test)
        record['predict_time_lr'] = sync() - t0

    print(f'--> test RMSE: full-kernel mean {record.get("test_rmse_exact", float("nan")):.4f}'
          f' [{record.get("predict_time_exact", 0):.2f}s], low-rank mean '
          f'{record["test_rmse_lr"]:.4f} [{record["predict_time_lr"]:.2f}s]')


if __name__ == '__main__':
    if args.prealloc_frac > 0:
        free, _ = torch.cuda.mem_get_info()
        buf = torch.empty(int(free * args.prealloc_frac), dtype=torch.uint8, device=DEVICE)
        del buf                      # the block stays in torch's caching allocator
    record = dict(timestamp=datetime.now().isoformat(timespec='seconds'), tag=args.tag,
                  dataset=args.dataset, split=args.split, rank=args.rank, r_fps=args.r_fps,
                  mp_mode=os.environ.get('LRGP_MP_MODE', 'gemm2') if args.grads == 'mp' else '',
                  gpu_name=torch.cuda.get_device_name(0), argv=' '.join(sys.argv))
    try:
        run(record)
        record['status'] = 'ok'
    except torch.cuda.OutOfMemoryError as e:
        record.update(status='oom', error=str(e).split('\n')[0][:300])
        print(f'[!] out of memory: {record["error"]}')
    except Exception as e:
        record.update(status=type(e).__name__, error=str(e).split('\n')[0][:300])
        traceback.print_exc()
    finally:
        record['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 2 ** 30
        if args.csv:
            write_row(args.csv, record)
    if record['status'] != 'ok':
        raise SystemExit(1)
