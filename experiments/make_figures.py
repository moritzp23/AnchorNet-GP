"""Figures 2-4 of the paper (Appendix B) from the recorded CG traces.

  buzz250_residuals.png  results/figures/buzz250_trace.csv (fp32, preconditioner rank 200)
                         and the fp64 run of Buzz rank 250, split 73
  he_precond_sweep.png   results/figures/he_pr_sweep.csv (fp32, HouseElectric r = 500, 3000)
  fp64_both.png          results/fp64_traces.csv, split 73, with the low-rank mean of
                         results/runs.csv as dotted lines

    python experiments/make_figures.py --out figures/

With --results DIR the four input files are read from DIR instead (figures/ subfolder
included), for example results/new/ after rerunning the experiments.
"""
import argparse
import csv
import os
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'results')

plt.rcParams.update({
    'font.size': 11, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 9.5, 'xtick.labelsize': 10, 'ytick.labelsize': 10,
    'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.5,
    'figure.dpi': 110, 'savefig.bbox': 'tight', 'lines.linewidth': 1.8,
    'axes.axisbelow': True,
})

TRUE_RES = r'$\|y-\widetilde{K}\alpha_k\|/\|y\|$'
MADDOX = {'buzz': 0.300, 'houseelectric': 0.052, '3droad': 0.194}


def load(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def trace(rows, **sel):
    """iteration -> row, for the rows matching every key in sel."""
    out = {}
    for r in rows:
        if all(str(r.get(k)) == str(v) for k, v in sel.items()):
            out[int(float(r['iter']))] = r
    return out


def series(tr, col):
    xs = sorted(k for k in tr if (tr[k].get(col) or '').strip())
    return xs, [float(tr[k][col]) for k in xs]


def lowrank_rmse(runs, ds, rk):
    v = [float(r['test_rmse_lr']) for r in runs
         if r['dataset'] == ds and int(r['rank']) == rk and r.get('use_rmse', '1') == '1']
    return statistics.mean(v) if v else None


def fig_buzz_residuals(out):
    tr = trace(load(os.path.join(RES, 'figures', 'buzz250_trace.csv')),
               dataset='buzz', rank=250, dtype='fp32')
    fig, (a, b) = plt.subplots(2, 1, figsize=(8.2, 7.0), sharex=True,
                               gridspec_kw={'height_ratios': [1.75, 1]})
    a.semilogy(*series(tr, 'relres_rec'), color='#c0392b', lw=1.2,
               label=r'recurrence $\|r_k\|/\|y\|$')
    a.semilogy(*series(tr, 'relres_prec'), color='#e0a030', lw=1.2, ls='--',
               label=r'preconditioned $\sqrt{r_k^\top P^{-1}r_k}\,/\,\sqrt{y^\top P^{-1}y}$')
    a.semilogy(*series(tr, 'relres_true'), color='#1f3a73', lw=2.4,
               label='true ' + TRUE_RES)
    xs, _ = series(tr, 'relres_true')
    a.set_xlim(0, max(xs) if xs else 500)
    a.set_ylabel('relative residual')
    a.grid(which='minor', alpha=0.12)
    a.legend(loc='lower left', framealpha=0.95)

    x, y = series(tr, 'rmse')
    b.plot(x, y, color='#1f3a73', lw=2.0, label='float32 CG')
    b.axhline(MADDOX['buzz'], color='#c0392b', ls='--', lw=1.4, label='Maddox et al.')
    f64 = trace(load(os.path.join(RES, 'fp64_traces.csv')), dataset='buzz', rank=250, split=73)
    _, fy = series(f64, 'rmse')
    if fy:
        b.axhline(fy[-1], color='#2a8f6f', ls=':', lw=1.4, label='float64 CG')
    b.set_ylim(0.25, 0.45)
    b.set_xlim(0, max(x) if x else 500)
    b.set_xlabel('CG iteration')
    b.set_ylabel('test RMSE')
    b.legend(loc='upper right', framealpha=0.95)
    save(fig, out, 'buzz250_residuals')


def fig_he_precond(out):
    rows = load(os.path.join(RES, 'figures', 'he_pr_sweep.csv'))
    prs = sorted({int(float(r['precond_rank'])) for r in rows})
    ranks = sorted({int(float(r['rank'])) for r in rows})
    fig, axes = plt.subplots(1, len(ranks), figsize=(11.0, 4.6), sharey=True, squeeze=False)
    cmap = plt.get_cmap('viridis')
    for ax, rk in zip(axes[0], ranks):
        # the preconditioner is built from the model's landmarks, so its rank is below r
        for p in [p for p in prs if p < rk]:
            x, y = series(trace(rows, dataset='houseelectric', rank=rk, precond_rank=p),
                          'relres_true')
            if x:
                ax.semilogy(x, y, color=cmap(prs.index(p) / max(len(prs) - 1, 1)),
                            label='$P$ rank %d' % p)
        ax.set_xlabel('CG iteration')
        ax.grid(which='minor', alpha=0.12)
        ax.set_xlim(0, 200)
    axes[0][0].set_ylabel('true residual ' + TRUE_RES)
    h, lab = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, lab, loc='lower center', ncol=len(lab), bbox_to_anchor=(0.5, 0.885),
               frameon=True, framealpha=1.0, edgecolor='#999999')
    save(fig, out, 'he_precond_sweep')


def fig_fp64(out):
    rows = load(os.path.join(RES, 'fp64_traces.csv'))
    runs = load(os.path.join(RES, 'runs.csv'))
    cols = [('houseelectric', [500, 1000, 2000, 3000], (0.04, 0.09)),
            ('buzz', [250, 500, 1000], (0.25, 0.42))]
    fig, axes = plt.subplots(2, len(cols), figsize=(11.0, 7.6), squeeze=False,
                             gridspec_kw={'height_ratios': [1, 1]})
    palette = ['#2d1a73', '#c0309a', '#d9534f', '#f0a030']
    for j, (ds, ranks, ylim) in enumerate(cols):
        top, bot = axes[0][j], axes[1][j]
        for i, rk in enumerate(ranks):
            c = palette[i % len(palette)]
            tr = trace(rows, dataset=ds, rank=rk, split=73)
            x, y = series(tr, 'relres_true')
            if x:
                top.semilogy(x, y, color=c, label='rank %d' % rk)
            x, y = series(tr, 'rmse')
            if x:
                bot.plot(x, y, color=c)
            lr = lowrank_rmse(runs, ds, rk)
            if lr is not None:
                bot.axhline(lr, color=c, ls=':', lw=1.4)
        top.legend(loc='best', framealpha=0.95)
        top.grid(which='minor', alpha=0.12)
        top.set_xlim(0, 100)
        bot.axhline(MADDOX[ds], color='#c0392b', ls='--', lw=1.4, label='Maddox et al.')
        bot.plot([], [], color='#777777', ls=':', lw=1.4, label='low-rank mean')
        bot.legend(loc='upper right', framealpha=0.95)
        bot.set_ylim(*ylim)
        bot.set_xlim(0, 100)
        bot.set_xlabel('CG iteration')
        if j == 0:
            top.set_ylabel('true residual ' + TRUE_RES)
            bot.set_ylabel('test RMSE')
    save(fig, out, 'fp64_both')


def save(fig, out, stem):
    fig.savefig(os.path.join(out, stem + '.png'), dpi=160)
    plt.close(fig)
    print(os.path.join(out, stem + '.png'))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default=os.path.join(ROOT, 'figures'))
    ap.add_argument('--results', default=RES)
    a = ap.parse_args()
    RES = a.results
    os.makedirs(a.out, exist_ok=True)
    fig_buzz_residuals(a.out)
    fig_he_precond(a.out)
    fig_fp64(a.out)
