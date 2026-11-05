"""Print the values of Tables 7-9 of the paper (Section 7) from results/mp/.

  Table 7  mp_fits.csv: the three fits of 3DRoad, rank 3000, split 73. The fp32 NLL per
           point at the fitted hyperparameters is printed by `mp_eval.py nll`.
  Table 8  mp_scan.csv (`mp_eval.py scan`): one evaluation per rank at the fp32 fit.
  Table 9  mp_trajectory_3droad_r3000.csv: spectra of the fp32 and the fp16 matrix M at
           every objective call of the gemm2 fit (call k is Adam iteration k - 1).
  Modified Cholesky  mp_repair_fits.csv: the fits of Table 7 with an indefinite M repaired
           (shift or clamp) instead of recomputed in fp32; mp_repair_scan.csv
           (`mp_eval.py repair`): one evaluation per rank at the fp32 fit.

    python experiments/mp_summary.py [--dir results/mp]
"""
import argparse
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
ap.add_argument('--dir', default=os.path.join(ROOT, 'results', 'mp'))
ap.add_argument('--iters', type=int, default=100, help='Adam iterations of the fits')
a = ap.parse_args()


def rows(name):
    p = os.path.join(a.dir, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


print('Table 7: 3DRoad, r = 3000, split 73')
for r in rows('mp_fits.csv'):
    fb = [c for c in r['mp_fallback_calls'].split() if int(c.split(':')[0]) <= a.iters]
    used = f'{a.iters - len(fb)} of {a.iters}' if r['mp_calls'] else '--'
    print(f"  {r['variant']:6s} fit {float(r['fit_time']):6.1f} s | median l {float(r['l_opt']):.5f} | "
          f"sigma_f {float(r['sigma_f_opt']):.4f} | sigma {float(r['sigma_y_opt']):.4f} | "
          f"eff. rank {r['eff_rank']} | low-rank RMSE {float(r['test_rmse_lr']):.4f} | "
          f"reduced M used {used}")

print('\nTable 8: one evaluation at the fp32 fit (split 73)')
for r in rows('mp_scan.csv'):
    ok = r['factorized'] == '1'
    print(f"  {r['dataset']:13s} r={int(r['rank']):5d} {r['variant']} kappa {float(r['kappa']):.2g} "
          f"{'factorizes' if ok else 'fails (' + (r['failure'] or 'error') + ')':22s}"
          + (f" grad err {float(r['grad_rel']):.3f} time {float(r['t32']):.2f} / "
             f"{float(r['t_mixed']):.2f} s" if ok else ''))

print('\nTable 9: gemm2 fit, iteration = call - 1')
for r in rows('mp_trajectory_3droad_r3000.csv'):
    print(f"  iter {int(r['call']) - 1:3d} sigma_f {float(r['sigma_f']):.3f} sigma {float(r['sigma_y']):.3f} "
          f"lam_max {float(r['lam_max32']):.2g} lam_min {float(r['lam_min32']):.3f} "
          f"lam_min(fp16) {float(r['lam_min16']):.3f} "
          f"{'succeeds' if r['ok16'] == '1' else 'overflow' if r['err_rel'] == 'inf' else 'fails'}")

print('\nModified Cholesky: 3DRoad, r = 3000, split 73')
for r in rows('mp_repair_fits.csv'):
    rep = r['mp_repair_calls'].split()
    taus = [float(c.split(':')[1]) for c in rep]
    print(f"  {r['mp_mode']:6s} {r['mp_repair']:5s} fit {float(r['fit_time']):6.1f} s | "
          f"median l {float(r['l_opt']):.5f} | sigma_f {float(r['sigma_f_opt']):.4f} | "
          f"sigma {float(r['sigma_y_opt']):.4f} | low-rank RMSE {float(r['test_rmse_lr']):.4f} | "
          f"{len(rep)} repaired (tau {min(taus):.3g} to {max(taus):.3g}), "
          f"{len(r['mp_fallback_calls'].split())} fp32 fallbacks")

print('\nModified Cholesky: one evaluation at the fp32 fit (split 73), relative gradient error')
groups = {}
for r in rows('mp_repair_scan.csv'):
    k = (r['dataset'], r['variant'], r['repair'], r['outcome'])
    groups.setdefault(k, []).append(float(r['grad_rel']) if r['grad_rel'] else None)
for (ds, v, rp, out), g in groups.items():
    e = [x for x in g if x is not None]
    print(f"  {ds:13s} {v} {rp:8s} {out:16s} {len(g):2d} ranks"
          + (f" | {min(e):.3g} to {max(e):.3g}" if e else ''))
