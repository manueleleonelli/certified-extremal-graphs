"""
Colorado daily precipitation (evgam::COprcp, 64 GHCN-Daily stations, April-October
1990-2019): rank transform per station, event days by the maximum over stations,
run declustering, componentwise maxima within events, empirical variogram with
pairwise-complete handling of missing values.
"""
import numpy as np, pandas as pd, sys, os
HERE = os.path.dirname(os.path.abspath(__file__))

def prepare(p=0.95, run=1, min_obs=60, verbose=True):
    M = pd.read_pickle(os.path.join(HERE, 'coprcp_matrix.pkl'))
    meta = pd.read_csv(os.path.join(HERE, 'coprcp_meta.csv'))
    X = M.values.astype(float)          # days x stations, NaN = missing
    dates = M.index
    n_days, d = X.shape
    # marginal rank transform to standard Pareto using observed values (ties: average rank)
    Y = np.full_like(X, np.nan)
    for j in range(d):
        obs = ~np.isnan(X[:, j])
        r = pd.Series(X[obs, j]).rank(method='average').values
        F = r / (obs.sum() + 1.0)
        Y[obs, j] = 1.0 / (1.0 - F)
    u = 1.0 / (1.0 - p)
    nobs = (~np.isnan(Y)).sum(1)
    mx = np.nanmax(np.where(np.isnan(Y), -np.inf, Y), axis=1)
    event = (mx > u) & (nobs >= min_obs)
    # run declustering: event days separated by at most `run` non-event days form one cluster
    idx = np.where(event)[0]
    clusters = []; cur = [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if b - a <= run + 1 and (dates[b] - dates[a]).days <= run + 1:
            cur.append(b)
        else:
            clusters.append(cur); cur = [b]
    clusters.append(cur)
    Yc = np.array([np.nanmax(Y[c], axis=0) if len(c) > 1 else Y[c[0]] for c in clusters])
    with np.errstate(invalid='ignore'):
        Yc = np.where(np.isfinite(Yc), Yc, np.nan)
    Yc = Yc / u                          # exceedances: max_j Yc_j > 1
    if verbose:
        print(f"days {n_days}, stations {d}, threshold p={p}: event days {event.sum()}, clusters {len(clusters)}, "
              f"mean cluster length {np.mean([len(c) for c in clusters]):.2f}, missing in events {np.isnan(Yc).mean():.3f}")
    return Yc, meta, dates, clusters

def emp_vario_missing(Y, thr=1.0, min_pairs=20):
    """averaged empirical variogram with pairwise-complete observations"""
    n, d = Y.shape
    LY = np.log(Y)
    num = np.zeros((d, d)); den = np.zeros((d, d))
    for k in range(d):
        rows = LY[np.nan_to_num(Y[:, k], nan=-np.inf) > thr]
        if len(rows) < 3:
            continue
        for i in range(d):
            di = rows[:, i][:, None] - rows           # (n_k, d)
            ok = ~np.isnan(di)
            cnt = ok.sum(0)
            m = np.where(cnt > 0, np.nansum(di, 0) / np.maximum(cnt, 1), 0.0)
            v = np.where(cnt > 1, np.nansum((di - m) ** 2, 0) / np.maximum(cnt - 1, 1), np.nan)
            good = cnt >= min_pairs
            num[i, good] += v[good]; den[i, good] += 1
    G = num / np.maximum(den, 1)
    G = 0.5 * (G + G.T); np.fill_diagonal(G, 0.0)
    return G, den

if __name__ == '__main__':
    p = float(sys.argv[1]) if len(sys.argv) > 1 else 0.95
    Yc, meta, dates, clusters = prepare(p=p)
    G, den = emp_vario_missing(Yc)
    d = G.shape[0]
    P = np.eye(d) - np.ones((d, d)) / d
    ev = np.linalg.eigvalsh(-0.5 * P @ G @ P)
    print("n exceedances", Yc.shape[0], "Gamma range", G[np.triu_indices(d, 1)].min(), G[np.triu_indices(d, 1)].max(),
          "Sigma eigenvalues min/2nd", ev[0], ev[1], "conditioning k used per pair min", int(den[np.triu_indices(d, 1)].min()))
    np.save(os.path.join(HERE, f'colorado_Y_p{p}.npy'), Yc)
    np.save(os.path.join(HERE, f'colorado_Gamma_p{p}.npy'), G)
