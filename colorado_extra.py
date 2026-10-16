"""
Additional analyses for the Colorado application:
  holdout     : two-fold cross-validation by odd and even years (path and competitors learned on one
                half of the years, surrogate likelihood evaluated on the other)
  thresholds  : path and BIC graph at the 0.95 and 0.975 thresholds
  boot a b    : bootstrap replicates a..b-1 of the events at the 0.99 threshold
Results go to results/colorado_*.json
"""
import sys, os, json, time
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hrl0 import *
from prep_colorado import emp_vario_missing as _evm
def emp_vario_missing(Y, min_pairs=10):
    return _evm(Y, min_pairs=min_pairs)
from experiments import dump, select_bic_path, OUT

P = 0.99

def events(years=None, p=P, run=1, min_obs=60):
    """rank transform within the selected years, event days by the maximum over stations,
    run declustering, componentwise maxima; returns exceedances, first dates of clusters"""
    M = pd.read_pickle(os.path.join(HERE, 'coprcp_matrix.pkl'))
    if years is not None:
        M = M[M.index.year.isin(list(years))]
    X = M.values.astype(float); dates = M.index
    n_days, d = X.shape
    Y = np.full_like(X, np.nan)
    for j in range(d):
        obs = ~np.isnan(X[:, j])
        r = pd.Series(X[obs, j]).rank(method='average').values
        Y[obs, j] = 1.0 / (1.0 - r / (obs.sum() + 1.0))
    u = 1.0 / (1.0 - p)
    nobs = (~np.isnan(Y)).sum(1)
    mx = np.nanmax(np.where(np.isnan(Y), -np.inf, Y), axis=1)
    idx = np.where((mx > u) & (nobs >= min_obs))[0]
    clusters = []; cur = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if (dates[b] - dates[a]).days <= run + 1:
            cur.append(b)
        else:
            clusters.append(cur); cur = [b]
    clusters.append(cur)
    Yc = np.array([np.nanmax(Y[c], axis=0) for c in clusters]) / u
    return Yc, [dates[c[0]] for c in clusters], [dates[c[-1]] for c in clusters], X, Y, dates

def score(Gtest, Theta):
    """surrogate negative log-likelihood per event of a precision matrix under a test variogram:
    -log pdet(Theta) - 1/2 tr(Gtest Theta)"""
    d = Theta.shape[0]
    ld = np.linalg.slogdet(Theta + np.ones((d, d)) / d)[1]
    return float(-ld - 0.5 * np.trace(Gtest @ Theta))

def eg_path(Gb, n, d, grid):
    items = []
    for rho in grid:
        S = eglearn(Gb, rho)
        if len(S) >= d - 1 and is_connected(d, S):
            F, Th = fit_general(Gb, S)
            if np.isfinite(F):
                items.append((len(S), F, S, Th))
    return items

def holdout():
    """two-fold cross-validation by odd and even years"""
    fold_years = {'odd': [y for y in range(1990, 2020) if y % 2 == 1], 'even': [y for y in range(1990, 2020) if y % 2 == 0]}
    data = {}
    for f, yrs in fold_years.items():
        Y, s_, e_, *_ = events(yrs)
        G, den = emp_vario_missing(Y)
        data[f] = dict(Y=Y, G=G, start=s_, end=e_)
    prev = os.path.join(OUT, 'colorado_cv.json')
    out = json.load(open(prev)) if os.path.exists(prev) else {}
    for train, test in [('odd', 'even'), ('even', 'odd')]:
        key = f'train_{train}'
        if key in out and 'eg' in out[key]:
            continue
        Gtr = data[train]['G']; Gte = data[test]['G']; ntr, d = data[train]['Y'].shape
        r = dict(n_train=int(ntr), n_test=int(data[test]['Y'].shape[0]))
        t = time.time(); path = forward_path(Gtr, 200, checkpoint=os.path.join(OUT, f'colorado_cv_ckpt_{train}.json')); r['t_path'] = time.time() - t
        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
        b, kb, Fb, Eb = select_bic_path(items, ntr)
        r['bic_k'] = kb; r['bic_edges'] = list(Eb)
        r['path'] = [dict(k=k, F_train=f.F, bic=bic(f.F, k, ntr), F_test=score(Gte, f.Theta)) for k, f in sorted(path.items())]
        full = emtp2_mle(Gtr)
        r['emtp2'] = dict(k=len(emtp2_support(full)), F_test=score(Gte, full.Theta))
        r['saturated_general_F_test'] = score(Gte, theta_from_gamma(Gtr))
        out[key] = r; dump('colorado_cv.json', out)
        items_eg = eg_path(Gtr, ntr, d, np.geomspace(0.02, 2.0, 12))
        if items_eg:
            bb = min(items_eg, key=lambda it: bic(it[1], it[0], ntr))
            sp = min(items_eg, key=lambda it: it[0])
            r['eg'] = dict(path=[dict(k=k, F_train=F, F_test=score(Gte, Th)) for k, F, S, Th in items_eg],
                           bic_k=bb[0], bic_F_test=score(Gte, bb[3]), sparsest_k=sp[0], sparsest_F_test=score(Gte, sp[3]))
        r['test_events'] = [dict(start=str(a.date()), end=str(b_.date()), exceed=[int(j) for j in np.where(np.nan_to_num(data[test]['Y'][i], nan=0) > 1)[0]])
                            for i, (a, b_) in enumerate(zip(data[test]['start'], data[test]['end']))]
        out[key] = r; dump('colorado_cv.json', out)
        print('cv fold done', train, kb, flush=True)

def thresholds():
    for p in [0.95, 0.975]:
        Gb = np.load(os.path.join(HERE, f'colorado_Gamma_p{p}.npy')); Yc = np.load(os.path.join(HERE, f'colorado_Y_p{p}.npy'))
        n, d = Yc.shape
        t = time.time(); path = forward_path(Gb, 260, checkpoint=os.path.join(OUT, f'colorado_thr_ckpt_p{p}.json'))
        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
        b, kb, Fb, Eb = select_bic_path(items, n)
        dump(f'colorado_thr_p{p}.json', dict(p=p, n=int(n), t_path=time.time() - t, bic_k=kb, bic_edges=list(Eb),
                                             path=[dict(k=k, F=f.F, bic=bic(f.F, k, n)) for k, f in sorted(path.items())]))
        print('threshold', p, kb, flush=True)

def boot(a, b):
    Yc = np.load(os.path.join(HERE, f'colorado_Y_p{P}.npy')); n, d = Yc.shape
    for rep in range(a, b):
        fn = os.path.join(OUT, f'colorado_boot_{rep}.json')
        if os.path.exists(fn):
            continue
        rng = np.random.default_rng(1000 + rep)
        Yb = Yc[rng.integers(0, n, n)]
        Gb, _ = _evm(Yb)
        path = forward_path(Gb, 200, checkpoint=os.path.join(OUT, f'colorado_boot_ckpt_{rep}.json'))
        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
        bb, kb, Fb, Eb = select_bic_path(items, n)
        json.dump(dict(rep=rep, bic_k=kb, bic_edges=[list(map(int, e)) for e in Eb], k_max=max(path)), open(fn, 'w'))
        print('boot', rep, kb, flush=True)

if __name__ == '__main__':
    what = sys.argv[1]
    if what == 'holdout': holdout()
    elif what == 'thresholds': thresholds()
    elif what == 'boot': boot(int(sys.argv[2]), int(sys.argv[3]))
