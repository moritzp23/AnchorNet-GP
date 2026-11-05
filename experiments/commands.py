"""Print the command for every run behind the paper's tables and figures, one per line.

    python experiments/commands.py fits   > fits.txt     # Tables 1-6: 1 fit per (dataset, rank, split)
    python experiments/commands.py fp64   > fp64.txt     # Tables 2, 4-6, Figure 4: fp64 CG solves
    python experiments/commands.py figs   > figs.txt     # Figures 2 and 3: fp32 CG traces
    python experiments/commands.py mp     > mp.txt       # Section 7: Tables 7-9, modified Cholesky

Each line is independent and runs on GPU 0 (add --gpu to choose another); distribute
them as convenient, e.g. `bash fits.txt` or a job scheduler. Results go to results/new/.
The fp64, figure and scan commands read fitted hyperparameters from results/theta/ (the
ones from the paper); `commands.py fp64 results/new/theta` uses those of your own fits.

Approximate cost on one A40: fits about 100 GPU-hours (44 h fitting, the rest in the fp32
CG solves), fp64 solves about 1,200 GPU-hours (one HouseElectric rank-3000 solve alone
takes 36 h for 200 iterations), figure traces 8 hours, MP runs about 1 hour.
"""
import collections
import csv
import os
import sys

from uci import SPLITS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RANKS = {
    'poletele': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000],
    'elevator': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000],
    'bike': [50, 100, 200, 250, 400, 500, 800, 1000, 1600],
    'kin40k': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000, 4000, 6000, 8000],
    'protein': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000, 4000, 6000, 8000],
    '3droad': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000, 4000, 6000],
    'song': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000, 4000],
    'buzz': [50, 100, 200, 250, 400, 500, 800, 1000],
    'houseelectric': [50, 100, 200, 250, 400, 500, 800, 1000, 1600, 2000, 3000],
}
FP64 = ('3droad', 'buzz', 'houseelectric')


def fp64_budget(ds, rk):
    return {'3droad': 200, 'buzz': 100}.get(ds, 200 if rk >= 3000 else 100)


def recorded_grads():
    """Gradient path of the timing runs in the paper, per (dataset, rank).

    run.py chooses between the unblocked and the blocked gradients by timing both. The
    unblocked code became faster during the experiments (a transposed triangular
    solve), so on cells timed before that change the choice could now differ. Fixing
    the path reproduces the configuration that was timed.
    """
    votes = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(os.path.join(ROOT, 'results', 'runs.csv'))):
        if r['use_fit_time'] == '1' and r['grads']:
            votes[(r['dataset'], int(r['rank']))][r['grads']] += 1
    return {k: c.most_common(1)[0][0] for k, c in votes.items()}


def fits():
    grads = recorded_grads()
    for ds, ranks in RANKS.items():
        for rk in ranks:
            for sp in SPLITS:
                th = (f' --dump_theta results/new/theta/{ds}_r{rk}_s{sp}.json'
                      if ds in FP64 else '')
                print(f'python experiments/run.py --dataset {ds} --rank {rk} --split {sp} '
                      f'--grads {grads.get((ds, rk), "adaptive")} '
                      f'--prealloc_frac 0.9 --csv results/new/runs.csv{th}')


def fp64(theta='results/theta'):
    for ds in FP64:
        for rk in RANKS[ds]:
            for sp in SPLITS:
                print(f'python experiments/pm_solve.py --theta {theta}/{ds}_r{rk}_s{sp}.json '
                      f'--dtype fp64 --precond-rank 1000 --max-iter {fp64_budget(ds, rk)} '
                      f'--checkpoint results/new/ckpt/{ds}_r{rk}_s{sp}.pt '
                      f'--out results/new/fp64_traces.csv')


def figs():
    print('python experiments/pm_solve.py --theta results/theta/buzz_r250_s73.json --dtype fp32 '
          '--precond-rank 200 --max-iter 500 --out results/new/figures/buzz250_trace.csv')
    for rk in (500, 3000):
        for pr in (0, 5, 50, 200, 500, 1000):
            if pr < rk:
                print(f'python experiments/pm_solve.py --theta results/theta/houseelectric_r{rk}_s73.json '
                      f'--dtype fp32 --precond-rank {pr} --max-iter 200 --out results/new/figures/he_pr_sweep.csv')


def mp():
    base = ('--dataset 3droad --rank 3000 --split 73 --prealloc_frac 0.9 --skip_exact '
            '--csv results/new/mp_fits.csv')
    print(f'python experiments/run.py {base} --grads blocked '
          f'--dump_theta results/new/theta_3droad_r3000_fp32.json')
    print(f'LRGP_MP_MODE=gemm2 LRGP_MP_DIAG=results/new/mp_trajectory_3droad_r3000.csv '
          f'python experiments/run.py {base} --grads mp '
          f'--dump_theta results/new/theta_3droad_r3000_gemm2.json')
    print(f'LRGP_MP_MODE=gemm4 python experiments/run.py {base} --grads mp '
          f'--dump_theta results/new/theta_3droad_r3000_gemm4.json')
    print('python experiments/mp_eval.py nll results/new/theta_3droad_r3000_fp32.json '
          'results/new/theta_3droad_r3000_gemm2.json results/new/theta_3droad_r3000_gemm4.json')
    thetas = ' '.join(f'results/theta/{ds}_r{rk}_s73.json' for ds in FP64 for rk in RANKS[ds])
    print(f'python experiments/mp_eval.py scan --out results/new/mp_scan.csv {thetas}')
    # modified Cholesky in place of the fp32 fallback (LRGP_MP_REPAIR, angp/fitc_mp.py)
    rep = [f'results/new/theta_3droad_r3000_{v}_{rp}.json'
           for v in ('gemm2', 'gemm4') for rp in ('shift', 'clamp')]
    for th in rep:
        v, rp = th[:-5].split('_')[-2:]
        print(f'LRGP_MP_MODE={v} LRGP_MP_REPAIR={rp} python experiments/run.py {base} '
              f'--csv results/new/mp_repair_fits.csv --grads mp --dump_theta {th}'.replace(
                  ' --csv results/new/mp_fits.csv', ''))
    print('python experiments/mp_eval.py nll ' + ' '.join(rep))
    rthetas = ' '.join(f'results/theta/{ds}_r{rk}_s73.json' for ds, ranks in
                       (('3droad', (250, 1000, 3000, 6000)), ('buzz', RANKS['buzz']),
                        ('houseelectric', RANKS['houseelectric'])) for rk in ranks)
    print(f'python experiments/mp_eval.py repair --out results/new/mp_repair_scan.csv {rthetas}')


if __name__ == '__main__':
    groups = {'fits': fits, 'fp64': fp64, 'figs': figs, 'mp': mp}
    if len(sys.argv) not in (2, 3) or sys.argv[1] not in groups:
        raise SystemExit(__doc__)
    groups[sys.argv[1]](*sys.argv[2:])
