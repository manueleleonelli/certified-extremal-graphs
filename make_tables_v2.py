"""
Tables and summary numbers of Section 6 from results/v2 (25 replications).
Usage: python make_tables_v2.py [recovery|recovery_w2|recovery_w4|exact|exact_signed|oa|certify|certify_dense|mio|all]
"""
import os, sys, json, glob
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, 'results', 'v2')
TAB = os.path.join(HERE, 'results', 'v2', 'tables')
os.makedirs(TAB, exist_ok=True)


def load(exp):
    rows = []
    for fn in sorted(glob.glob(os.path.join(V2, exp, '*.json'))):
        rows.append(json.load(open(fn)))
    return rows


def cp_upper(x, n, alpha=0.05):
    """one-sided Clopper-Pearson upper bound for a binomial proportion"""
    from scipy.stats import beta
    return 1.0 if x >= n else float(beta.ppf(1 - alpha, x + 1, n - x))


def q(v, p):
    return float(np.percentile(v, p)) if len(v) else float('nan')


def fmt_t(t):
    if t != t:
        return '--'
    if t < 10:
        return f'{t:.1f}'
    return f'{t:.0f}'


# ---------------------------------------------------------------------------
METHODS = [('mst', 'MST'), ('emtp2', 'EMTP$_2$'), ('l1', '$\\ell_1$'), ('gl', 'EGL'), ('eg', 'EGlearn'),
           ('l0', '$\\ell_0$'), ('l0g', '$\\ell_0^\\pm$')]


def recovery(exp='recovery'):
    rows = load(exp)
    suf = '' if exp == 'recovery' else exp[len('recovery'):]
    cfg = {}
    for r in rows:
        cfg.setdefault((r['d'], r['gtype'], r['n']), []).append(r)
    order = [(d, g, n) for d in [10, 20] for g in ['sparse', 'dense', 'signed'] for n in [100, 250, 1000]]
    lines = []; summ = dict(ses=[], best=[], l0_sig_better=0, l0_sig_worse=0, l0_tie=0, configs=0)
    stats = {}
    for key in order:
        R = cfg.get(key, [])
        if not R:
            continue
        R = sorted(R, key=lambda r: r['rep'])
        F = {m: np.array([r.get(m + '_f1', np.nan) for r in R], float) for m, _ in METHODS}
        means = {m: np.nanmean(F[m]) for m, _ in METHODS}
        ses = {m: np.nanstd(F[m], ddof=1) / np.sqrt(np.sum(~np.isnan(F[m]))) for m, _ in METHODS}
        best = max(means, key=means.get)
        bold = {}
        pvals = {}
        for m, _ in METHODS:
            if m == best:
                bold[m] = True; pvals[m] = 1.0; continue
            dlt = F[best] - F[m]
            dlt = dlt[~np.isnan(dlt)]
            if np.all(dlt == 0):
                p = 1.0
            else:
                p = float(wilcoxon(dlt).pvalue)
            pvals[m] = p
            bold[m] = p >= 0.05
        stats[key] = dict(means=means, ses=ses, best=best, p=pvals, n=len(R))
        summ['ses'] += [ses[m] for m, _ in METHODS]
        d, g, n = key
        cells = []
        for m, _ in METHODS:
            s = f'{means[m]:.3f}'
            cells.append('\\textbf{' + s + '}' if bold[m] else s)
        lines.append(f"{d} & {g} & {n} & {R[0]['ktrue']} & " + ' & '.join(cells) + ' \\\\')
        # l0 against the best competitor (excluding l0g)
        comp = [m for m, _ in METHODS if m not in ('l0', 'l0g')]
        bc = max(comp, key=lambda m: means[m])
        dl = F['l0'] - F[bc]
        p = 1.0 if np.all(dl == 0) else float(wilcoxon(dl).pvalue)
        stats[key].update(best_comp=bc, l0_minus_bc=float(np.mean(dl)), l0_vs_bc_p=p)
        summ['configs'] += 1
        if p < 0.05 and np.mean(dl) > 0:
            summ['l0_sig_better'] += 1
        elif p < 0.05:
            summ['l0_sig_worse'] += 1
        else:
            summ['l0_tie'] += 1
    open(os.path.join(TAB, f'recovery{suf}_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('reps per config:', {k: v['n'] for k, v in stats.items()})
    print('SE range: %.3f-%.3f' % (min(summ['ses']), max(summ['ses'])))
    for key, s in stats.items():
        print(key, 'best', s['best'], 'best comp', s['best_comp'], 'l0-bc %+.3f p=%.3g' % (s['l0_minus_bc'], s['l0_vs_bc_p']))
    print({k: v for k, v in summ.items() if k != 'ses'})
    allm = {m: np.mean([s['means'][m] for s in stats.values()]) for m, _ in METHODS}
    print('mean over configs:', {m: round(v, 3) for m, v in allm.items()})
    # supplementary table: oracle, sizes, times
    ec = []
    for key in order:
        R = cfg.get(key, [])
        if not R:
            continue
        d, g, n = key
        three = ('eg', 'l0', 'l0g')
        orc = [np.mean([r.get(m + '_oracle_f1', np.nan) for r in R]) for m in three]
        sz = [np.mean([r.get(m + '_k', np.nan) for r in R]) for m in three]
        tm = [np.median([r.get(m + '_time', np.nan) for r in R]) for m in three]
        ec.append(f"{d} & {g} & {n} & {R[0]['ktrue']} & " + ' & '.join(f'{x:.3f}' for x in orc) + ' & '
                  + ' & '.join(f'{x:.1f}' for x in sz) + ' & ' + ' & '.join(f'{x:.1f}' for x in tm) + ' \\\\')
    open(os.path.join(TAB, f'recovery{suf}_ec_rows.tex'), 'w').write('\n'.join(ec) + '\n')
    print('\n'.join(ec))
    return stats


# ---------------------------------------------------------------------------
def exact(signed=False):
    exp = 'exact_signed' if signed else 'exact'
    rows = load(exp)
    cfg = {}
    for r in rows:
        cfg.setdefault((r['d'], r['ktrue'], r['n']), []).append(r)
    lines = []; tot = dict(ds=0, pr=0, heur=0, cert=0, full=0, ds_heur_fail=0, ds_cert_fail=0, gaps=[])
    for key in sorted(cfg):
        R = cfg[key]
        P = [p for r in R for p in r['problems']]
        nh = sum(p['heur_optimal'] for p in P); nc = sum(p['cert_optimal'] for p in P); nf = sum(p['exact'] for p in P)
        dsh = sum(not all(p['heur_optimal'] for p in r['problems']) for r in R)
        dsc = sum(not all(p['cert_optimal'] for p in r['problems']) for r in R)
        tot['ds'] += len(R); tot['pr'] += len(P); tot['heur'] += nh; tot['cert'] += nc; tot['full'] += nf
        tot['ds_heur_fail'] += dsh; tot['ds_cert_fail'] += dsc
        tot['gaps'] += [p['gap_heur'] for p in P if not p['heur_optimal']]
        sup = np.mean([p['n_supports'] for p in P]); ref = np.mean([p['refits'] for p in P])
        tc = [p['t_cert'] for p in P]
        d, kt, n = key
        lines.append(f"{d} & {kt} & {n} & {len(R)} & {len(P)} & {nh} & {nc} & {nf} & {sup:,.0f} & {ref:,.0f} & "
                     f"{fmt_t(np.median(tc))} & {fmt_t(np.max(tc))} \\\\".replace(',', '{,}'))
        print(key, 'datasets', len(R), 'heur-fail datasets', dsh, 'cert-fail datasets', dsc,
              'fallback', sum(p['n_fallback'] for p in P), 't_brute med %.1f' % np.median([p['t_brute'] for p in P]))
    open(os.path.join(TAB, exp + '_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print({k: v for k, v in tot.items() if k != 'gaps'}, 'max gap', max(tot['gaps']) if tot['gaps'] else 0)
    print('95%% upper bound, data sets with a path miss: %.3f' % cp_upper(tot['ds_heur_fail'], tot['ds']))
    return tot


# ---------------------------------------------------------------------------
def oa():
    rows = load('oa')
    for d in sorted(set(r['d'] for r in rows)):
        R = [r for r in rows if r['d'] == d]
        conv = [r for r in R if r['status'] == 'optimal']
        err = [r for r in R if str(r['status']).startswith('solver error')]
        stuck = [r for r in R if r['LB'] == r['LB'] and abs(r['LB'] - r['F_full']) < 1e-9]
        rad120 = [max([e['r'] for e in r['rlog'] if e['event'] == 'certified' and e['time'] <= 120] or [0]) for r in R]
        print(f"d={d}: {len(R)} data sets; converged {len(conv)} (iters {sorted(r['iters'] for r in conv)}, time {sorted(round(r['time']) for r in conv)});"
              f" solver errors {len(err)}; LB at f(E) at the limit {len(stuck)} of {len(R) - len(conv) - len(err)} not converged;"
              f" iters median {np.median([r['iters'] for r in R]):.0f}, cuts median {np.median([r['ncuts'] for r in R]):.0f};"
              f" UB = path value {sum(abs(r['UB'] - r['heur']) < 1e-7 for r in R)};"
              f" radius certified within 120 s: {dict(zip(*np.unique(rad120, return_counts=True)))}; k={R[0]['k']}")


# ---------------------------------------------------------------------------
def radius_time(res, r):
    """time at which radius r was certified, or None"""
    t = None
    for ev in res['rlog']:
        if ev['event'] == 'certified' and ev['r'] == r:
            t = ev['time']
    return t


def certify():
    rows = load('certify')
    cfg = {}
    for r in rows:
        cfg.setdefault((r['d'], r['n']), []).append(r)
    lines = []
    for key in sorted(cfg):
        R = cfg[key]
        d, n = key
        B = R[0]['budget']
        cells = []
        for r in (1, 2, 3):
            tc = [radius_time(x['comp'], r) for x in R]
            tn = [radius_time(x['nocomp'], r) if x['nocomp']['r_max'] >= r else 'na' for x in R]
            nc = sum(t is not None for t in tc)
            # median with time-outs counted as exceeding the budget
            def med(ts):
                ts = [t for t in ts if t != 'na']
                if not ts:
                    return '--'
                v = sorted([t if t is not None else np.inf for t in ts])
                m = np.median(v)
                return fmt_t(m) if np.isfinite(m) else f'$>{B}$'
            cells += [f'{nc}/{len(R)}', med(tc), med(tn)]
        ks = [x['k'] for x in R]
        lines.append(f"{d} & {n} & {np.median(ks):.0f} & " + ' & '.join(cells) + ' \\\\')
        imp = sum(x['comp']['improved'] for x in R)
        df1 = [x['comp']['f1'] - x['f1_heur'] for x in R if x['comp']['improved']]
        r4 = [radius_time(x['comp'], 4) for x in R]
        prog = [ev['done'] / ev['total'] for x in R for ev in x['comp']['rlog'] if ev['event'] == 'timeout']
        ratios = {}
        for r in (1, 2, 3):
            rr = [radius_time(x['nocomp'], r) / radius_time(x['comp'], r) for x in R
                  if radius_time(x['comp'], r) and radius_time(x['nocomp'], r)]
            ratios[r] = (np.median(rr), len(rr)) if rr else None
        refr = [x['nocomp']['refits'] / max(1, x['comp']['refits']) for x in R]
        print(key, 'k med', np.median(ks), 'improved', imp, 'dF1', np.round(df1, 3), 'r4 cert', sum(t is not None for t in r4),
              'r4 med', np.median([t for t in r4 if t]) if any(r4) else None,
              'timeout progress med %.2f' % np.median(prog) if prog else '', 'ratios', ratios,
              'tpath med %.1f' % np.median([x['t_path'] for x in R]))
    open(os.path.join(TAB, 'certify_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def exact_signed_table():
    """Table EC.1: as Table 1 plus the mean gain of the general model over EMTP2 at the same size"""
    gain = {}
    for fn in glob.glob(os.path.join(V2, 'posthoc', 'gain', '*.json')):
        for x in json.load(open(fn)):
            gain[(x['d'], x['ktrue'], x['n'], x['rep'], x['k'])] = x['gain']
    rows = load('exact_signed'); cfg = {}
    for r in rows:
        cfg.setdefault((r['d'], r['ktrue'], r['n']), []).append(r)
    lines = []
    for key in sorted(cfg):
        R = cfg[key]; P = [(r, p) for r in R for p in r['problems']]
        g = [gain[(r['d'], r['ktrue'], r['n'], r['rep'], p['k'])] for r, p in P if (r['d'], r['ktrue'], r['n'], r['rep'], p['k']) in gain]
        nh = sum(p['heur_optimal'] for _, p in P); nc = sum(p['cert_optimal'] for _, p in P); nf = sum(p['exact'] for _, p in P)
        sup = np.mean([p['n_supports'] for _, p in P]); ref = np.mean([p['refits'] for _, p in P]); tc = [p['t_cert'] for _, p in P]
        d, kt, n = key
        gs = f'{np.mean(g):.3f}' if len(g) == len(P) else f'({len(g)}/{len(P)})'
        lines.append(f"{d} & {kt} & {n} & {len(P)} & {nh} & {nc} & {nf} & {gs} & {sup:,.0f} & {ref:,.0f} & "
                     f"{fmt_t(np.median(tc))} & {fmt_t(np.max(tc))} \\\\".replace(',', '{,}'))
        if g:
            print(key, 'gain min %.4f median %.4f max %.4f' % (min(g), np.median(g), max(g)))
    open(os.path.join(TAB, 'exact_signed_ec_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def certify_dense():
    rows = load('certify_dense'); cfg = {}
    for r in rows:
        cfg.setdefault((r['d'], r['n']), []).append(r)
    lines = []
    for key in sorted(cfg):
        R = cfg[key]; d, n = key; B = R[0]['budget']; cells = []
        for r in (1, 2, 3):
            tc = [radius_time(x['comp'], r) for x in R]
            v = sorted([t if t is not None else np.inf for t in tc]); m = np.median(v)
            cells += [f'{sum(t is not None for t in tc)}/{len(R)}', fmt_t(m) if np.isfinite(m) else f'$>{B}$']
        r4 = [radius_time(x['comp'], 4) for x in R]
        prog = [ev['done'] / ev['total'] for x in R for ev in x['comp']['rlog'] if ev['event'] == 'timeout']
        imp = [(x['rep'], round(x['heur'] - x['comp']['F'], 4), round(x['comp']['f1'] - x['f1_heur'], 3)) for x in R if x['comp']['improved']]
        lines.append(f"{d} & {n} & {np.median([x['k'] for x in R]):.0f} & " + ' & '.join(cells) + ' \\\\')
        print(key, 'reps', len(R), 'ktrue', R[0]['ktrue'], 'k med', np.median([x['k'] for x in R]), 'tpath med %.0f' % np.median([x['t_path'] for x in R]),
              'r4', sum(t is not None for t in r4), 'timeout progress med %.2f' % np.median(prog) if prog else '', 'improved', imp)
    open(os.path.join(TAB, 'certify_dense_rows.tex'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


def mio():
    """Section 6.2: branch-and-cut with Gurobi (results of mio_gurobi.py)"""
    for g in ('0.0', '0.1'):
        for d in (8, 10):
            R = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(V2, 'mio_gurobi', f'g{g}_{d}_r*.json')))]
            if not R:
                continue
            opt = [r for r in R if r['optimal']]; no = [r for r in R if not r['optimal']]
            print(f"gamma={g} d={d}: {len(opt)}/{len(R)} solved, max time {max([r['time'] for r in opt], default=float('nan')):.1f} s;"
                  f" bound at the saturated value in {sum(abs(r['LB'] - r['f_full']) < 1e-6 for r in no)} of {len(no)} unsolved,"
                  f" median nodes {np.median([r['nodes'] for r in no]) if no else float('nan'):.0f}")


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('recovery', 'all'):
        recovery()
    if what.startswith('recovery_w'):
        recovery(what)
    if what in ('exact', 'all'):
        exact(False); exact(True)
    if what in ('oa', 'all'):
        oa()
    if what in ('certify', 'all'):
        certify()
    if what in ('exact_signed', 'all'):
        exact_signed_table()
    if what in ('certify_dense', 'all'):
        certify_dense()
    if what in ('mio', 'all'):
        mio()
