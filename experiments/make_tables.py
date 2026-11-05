"""Build Tables 1-6 of the paper from the result files.

Inputs (defaults under results/):
  runs.csv          one row per fit (run.py output). Optional 0/1 columns select which rows
                    enter which column: use_rmse (RMSE of the low-rank and the fp32
                    full-kernel mean), use_rmse_fp32 (overrides use_rmse for the fp32
                    column), use_eff_rank, use_fit_time, use_predict_lr and
                    use_predict_exact. A missing column means every 'ok' row is used, so
                    the CSV written by run.py can be passed directly.
  fp64_traces.csv   fp64 CG traces written by pm_solve.py, one run per (dataset, rank, split).

Rules:
  RMSE              mean and sample standard deviation over the splits.
  fit time          median over the timing runs of a cell.
  predict times     median over the runs of a cell.
  fp64 RMSE         final iterate of each run, mean and sd over the splits.
  fp64 time         mean over the splits of the time at the first recorded iteration
                    after which the RMSE stays within 1e-4 of its final value.
  Table 2           fp64 column for 3DRoad, Buzz and HouseElectric, fp32 otherwise.
  Table 1           the cheapest rank (by fit time) whose Table 2 entry is at or below the
                    reference of Maddox et al.

    python experiments/make_tables.py --out tables/
"""
import argparse
import csv
import math
import os
import statistics as st
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Maddox et al. (2022), Table 1, single precision: RMSE, sd, fit time on a V100 [s].
# Their Protein entry 0.055 is read as 0.55 (Section 6 of the paper).
MADDOX = {
    'poletele': (0.117, 0.004, 57.6),
    'elevator': (0.364, 0.004, 58.4),
    'bike': (0.074, 0.003, 62.3),
    'kin40k': (0.099, 0.001, 65.5),
    'protein': (0.55, 0.006, 70.9),
    '3droad': (0.194, 0.010, 1260.0),
    'song': (0.761, 0.004, 24357.0),
    'buzz': (0.300, 0.010, 25436.0),
    'houseelectric': (0.052, 0.002, 42751.0),
}
NAME = {'poletele': 'PoleTele', 'elevator': 'Elevators', 'bike': 'Bike', 'kin40k': 'Kin40K',
        'protein': 'Protein', '3droad': '3DRoad', 'song': 'Song', 'buzz': 'Buzz',
        'houseelectric': 'HouseElectric'}
FP64 = ('3droad', 'buzz', 'houseelectric')     # fp32 CG does not reach the needed accuracy
PM = r'\,$\pm$\,'


def flt(r, k):
    try:
        return float(r[k])
    except (KeyError, TypeError, ValueError):
        return float('nan')


def load_runs(path):
    cells = defaultdict(lambda: defaultdict(list))   # (ds, rank) -> quantity -> values
    size = {}
    for r in csv.DictReader(open(path)):
        if r.get('status', 'ok') != 'ok':
            continue
        use = lambda k: (r.get(k) or '1') not in ('0', 'False')   # noqa: E731
        c = cells[(r['dataset'], int(float(r['rank'])))]
        size[r['dataset']] = (int(float(r['N_train'])), int(float(r['d'])))
        if use('use_rmse'):
            c['lr'].append(flt(r, 'test_rmse_lr'))
        if use('use_eff_rank' if 'use_eff_rank' in r else 'use_rmse'):
            c['eff'].append(flt(r, 'eff_rank'))
        fp32_flag = 'use_rmse_fp32' if 'use_rmse_fp32' in r else 'use_rmse'
        if use(fp32_flag) and math.isfinite(flt(r, 'test_rmse_exact')):
            c['fp32'].append(flt(r, 'test_rmse_exact'))
        for flag, key, q in (('use_fit_time', 'fit_time', 'fit'),
                             ('use_predict_lr', 'predict_time_lr', 't_lr'),
                             ('use_predict_exact', 'predict_time_exact', 't_fp32')):
            if use(flag) and math.isfinite(flt(r, key)):
                c[q].append(flt(r, key))
    return cells, size


def load_fp64(path, cells):
    runs = defaultdict(dict)                          # (ds, rank, split) -> iter -> row
    if os.path.exists(path):
        for r in csv.DictReader(open(path)):
            runs[(r['dataset'], int(r['rank']), r['split'])][int(r['iter'])] = r
    for (ds, rk, _), tr in runs.items():
        # CG can reach an exactly zero residual and then produce NaN; skip those rows
        rm = {i: float(r['rmse']) for i, r in tr.items()
              if r['rmse'] and math.isfinite(float(r['rmse']))}
        if not rm:
            continue
        final = rm[max(rm)]
        settle = min(i for i in rm if all(abs(rm[j] - final) < 1e-4 for j in rm if j >= i))
        cells[(ds, rk)]['fp64'].append(final)
        cells[(ds, rk)]['t_fp64'].append(float(tr[settle]['elapsed_s']))


def cell(cells, ds, rk):
    """The results of one (dataset, rank), without inserting empty cells."""
    return cells[(ds, rk)] if (ds, rk) in cells else defaultdict(list)


def mean(v):
    return st.mean(v) if v else None


def ms(v, ref=None):
    """'mean +- sd', bold if the mean is at or below ref."""
    if not v:
        return ''
    s = f'{st.mean(v):.4f}' + (f'{PM}{st.stdev(v):.4f}' if len(v) > 1 else '')
    return rf'\textbf{{{s}}}' if ref is not None and st.mean(v) <= ref else s


def t_short(v):
    return f'{v:,.2f}' if v is not None else '--'


def t_round(v):
    if v is None:
        return '--'
    return f'{v:.1f}' if v < 100 else f'{v:,.0f}'


def headline(cells, ds, rk):
    """The Table 2 entry: fp64 for the three large datasets, fp32 otherwise."""
    return cell(cells, ds, rk)['fp64' if ds in FP64 else 'fp32']


def selected_rank(cells, ds):
    ok = [rk for (d, rk) in cells if d == ds and headline(cells, ds, rk)
          and st.mean(headline(cells, ds, rk)) <= MADDOX[ds][0]]
    return min(ok, key=lambda rk: st.median(cells[(ds, rk)]['fit'])) if ok else None


def fit_time(cells, ds, rk):
    v = cell(cells, ds, rk)['fit']
    return st.median(v) if v else None


def table_summary(cells, size):
    L = [r'\begin{table}[h]', r'  \centering',
         r'  \caption{Fit time comparison with \texttt{Maddox} \cite[Table 1]{maddox2022low}.}',
         r'  \label{tab:summary}', r'  \resizebox{\textwidth}{!}{%',
         r'  \begin{tabular}{lrrrrrrr}', r'    \toprule',
         r'    & & & \multicolumn{2}{c}{RMSE} & \multicolumn{3}{c}{time [s]} \\',
         r'    \cmidrule(lr){4-5}\cmidrule(lr){6-8}',
         r'    dataset & $(n, \, d)$ & $r$ & ANGP & \texttt{Maddox} & ANGP & \texttt{Maddox} & speed-up \\',
         r'    \midrule']
    for ds, (m_r, m_sd, m_t) in MADDOX.items():
        rk = selected_rank(cells, ds)
        if rk is None:
            continue
        n, d = size[ds]
        k = f'{n / 1000:.1f}'.rstrip('0').rstrip('.')
        val = ms(headline(cells, ds, rk), m_r) + (r'$^{\ast}$' if ds in FP64 else '')
        ft = fit_time(cells, ds, rk)
        L.append(rf'    {NAME[ds]} & ({k}K, {d}) & {rk} & {val} & {m_r:.3f}{PM}{m_sd:.3f} '
                 rf'& {t_round(ft)} & {m_t:,.0f} & ${m_t / ft:,.1f}\times$ \\')
    L += [r'    \bottomrule', r'  \end{tabular}}', r'\end{table}']
    return '\n'.join(L)


def table_rank(cells, key, label, caption, speedup_row):
    ranks = sorted({rk for (_, rk) in cells})
    star = (lambda ds: r'$^{\ast}$' if ds in FP64 else '') if key == 'headline' else (lambda ds: '')
    L = [r'\begin{table}[t]' if key != 'headline' else r'\begin{table}[h]', r'  \centering',
         r'  \small', rf'  \caption{{{caption}}}', rf'  \label{{{label}}}']
    if key == 'headline':
        L += [r'  \resizebox{\textwidth}{!}{%']
    else:
        L += [r'  \begin{adjustbox}{max width=\textwidth}']
    L += [r'  \begin{tabular}{rrrrrrrrrr}', r'    \toprule',
          '     & ' + ' & '.join(NAME[ds] + star(ds) for ds in MADDOX) + r' \\',
          '    $r$ & ' + ' & '.join(f'({MADDOX[ds][0]:.3f})' for ds in MADDOX) + r' \\',
          r'    \midrule']
    for rk in ranks:
        row = []
        for ds, (m_r, _, _) in MADDOX.items():
            v = headline(cells, ds, rk) if key == 'headline' else cell(cells, ds, rk)[key]
            s = f'{st.mean(v):.4f}' if v else ''
            row.append(rf'\textbf{{{s}}}' if v and st.mean(v) <= m_r else s)
        if any(row):
            L.append(f'    {rk} & ' + ' & '.join(row) + r' \\')
    if speedup_row:
        sp = []
        for ds, (_, _, m_t) in MADDOX.items():
            rk = selected_rank(cells, ds)
            sp.append(rf'${m_t / fit_time(cells, ds, rk):.1f}\times$' if rk else '--')
        L += [r'    \midrule', r'    \textit{speed-up} & ' + ' & '.join(sp) + r' \\']
    L += [r'    \bottomrule', r'  \end{tabular}}' if key == 'headline' else r'  \end{tabular}']
    if key != 'headline':
        L += [r'  \end{adjustbox}']
    L += [r'\end{table}']
    return '\n'.join(L)


def table_cost(cells, ds):
    m_r, _, m_t = MADDOX[ds]
    L = [r'\begin{table}[t]', r'  \centering',
         rf'  \caption{{Test RMSE and prediction time of the three predictive means on {NAME[ds]}. '
         rf'The reference RMSE is ${m_r:.3f}$.}}',
         rf'  \label{{tab:cost-{ds}}}', r'  \begin{adjustbox}{max width=\textwidth}',
         r'  \begin{tabular}{rrrrrrrrrr}', r'    \toprule',
         r'    & & & & \multicolumn{2}{c}{low-rank mean} & \multicolumn{2}{c}{float32 CG} & \multicolumn{2}{c}{float64 CG} \\',
         r'    \cmidrule(lr){5-6}\cmidrule(lr){7-8}\cmidrule(lr){9-10}',
         r'    $r$ & eff.\ & fit [s] & $\times$ref & RMSE & time [s] & RMSE & time [s] & RMSE & time [s] \\',
         r'    \midrule']
    for rk in sorted(rk for (d, rk) in cells if d == ds):
        c = cells[(ds, rk)]
        ft = fit_time(cells, ds, rk)
        eff = str(int(st.median(c['eff']))) if c['eff'] else '--'
        tl = st.median(c['t_lr']) if c['t_lr'] else None
        te = st.median(c['t_fp32']) if c['t_fp32'] else None
        L.append(rf'    {rk} & {eff} & {t_short(ft)} & ${m_t / ft:.1f}\times$ '
                 rf'& {ms(c["lr"], m_r)} & {t_short(tl)} '
                 rf'& {ms(c["fp32"], m_r)} & {f"{te:,.0f}" if te is not None else "--"} '
                 rf'& {ms(c["fp64"], m_r) or "--"} & {t_round(mean(c["t_fp64"]))} \\')
    L += [r'    \bottomrule', r'  \end{tabular}', r'  \end{adjustbox}', r'\end{table}']
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--runs', default=os.path.join(ROOT, 'results', 'runs.csv'))
    ap.add_argument('--fp64', default=os.path.join(ROOT, 'results', 'fp64_traces.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'tables'))
    a = ap.parse_args()
    cells, size = load_runs(a.runs)
    load_fp64(a.fp64, cells)
    tables = {
        'summary.tex': table_summary(cells, size),
        'rmse_vs_rank.tex': table_rank(
            cells, 'headline', 'tab:rmse-rank',
            r'Test RMSE from the posterior mean solve against rank. The last row reports '
            r'the speedup for hyperparameter fitting at the smallest rank that achieves at '
            r'least the same accuracy as \texttt{Maddox}.', True),
        'rmse_vs_rank_lr.tex': table_rank(
            cells, 'lr', 'tab:rmse-rank-lr',
            r'Test RMSE from the low-rank predictive mean for the fits of '
            r'Table~\ref{tab:rmse-rank}.', False),
    }
    for ds in FP64:
        tables[f'cost_{ds}.tex'] = table_cost(cells, ds)
    os.makedirs(a.out, exist_ok=True)
    for name, text in tables.items():
        with open(os.path.join(a.out, name), 'w') as f:
            f.write(text + '\n')
        print(os.path.join(a.out, name))


if __name__ == '__main__':
    main()
