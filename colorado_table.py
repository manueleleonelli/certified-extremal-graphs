"""Table 4 of the paper: estimated graphs for the Colorado data (edges, profile value, median edge
length, share of edges under 50 km, share joining a plains and a mountain station, and mean absolute
error of the extremal correlations of all pairs against those implied by the empirical variogram).
The general model is shown at the size of the EMTP2 solution when general_matched_colorado.py has run.
Reads results/colorado_p0.99.json; writes results/colorado_table.json and figures/tab_colorado.tex."""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from hrl0 import fit_support, fit_general, gamma_from_theta
RES = os.path.join(HERE, 'results'); FIG = os.path.join(HERE, 'figures'); os.makedirs(FIG, exist_ok=True)


def chi_from_gamma(G):
    return 2.0 - 2.0 * norm.cdf(np.sqrt(np.maximum(G, 0.0)) / 2.0)


def main(p=0.99):
    o = json.load(open(os.path.join(RES, f'colorado_p{p}.json')))
    Gb = np.load(os.path.join(HERE, f'colorado_Gamma_p{p}.npy'))
    meta = pd.read_csv(os.path.join(HERE, 'coprcp_meta.csv'))
    lon, lat, elev = meta['lon'].values, meta['lat'].values, meta['elev'].values
    d = Gb.shape[0]; iu = np.triu_indices(d, 1)
    la, lo = np.radians(lat), np.radians(lon)
    dist = 6371.0 * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None, :]
                                      + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo[:, None] - lo[None, :]), -1, 1))
    plains, mount = elev < 1900, elev > 2400
    chi_emp = chi_from_gamma(Gb)[iu]

    kb, gen, comp = str(o['bic_k']), o['general'], o['competitors']
    kg = str(gen['bic_k'])
    radius = lambda c: c.get('radius', c['per_r'][-1]['radius'])
    rows = [('Minimum spanning tree', o['mst_edges'], 'EMTP$_2$'),
            ('EMTP$_2$ unpenalized support', o['emtp2_edges'], 'EMTP$_2$'),
            ('$\\ell_1$-penalized EMTP$_2$, BIC', comp['l1']['edges'], 'EMTP$_2$'),
            ('EGlearn, BIC', comp['eg']['edges'], 'general'),
            ('EGlearn, sparsest connected', comp['eg']['sparsest_edges'], 'general'),
            (f"$\\ell_0$, BIC, certified at radius {radius(o['cert'][kb])}", o['cert'][kb]['edges'], 'EMTP$_2$'),
            (f"$\\ell_0^\\pm$, BIC, certified at radius {radius(gen['cert'][kg])}", gen['cert'][kg]['edges'], 'general')]
    if 'matched' in gen:                 # Table 4 shows the general model at the size of the EMTP2 solution
        rows[-1] = (f"$\\ell_0^\\pm$, same size, certified at radius {gen['matched']['radius']}", gen['matched']['edges'], 'general')
    table, lines = {}, []
    for name, edges, model in rows:
        E = [tuple(map(int, e)) for e in edges]
        if model == 'general':
            F, Th = fit_general(Gb, E); G = gamma_from_theta(Th)
        else:
            f = fit_support(Gb, E); F, G = f.F, f.R
        L = np.array([dist[i, j] for i, j in E])
        cross = np.mean([(plains[i] and mount[j]) or (plains[j] and mount[i]) for i, j in E])
        r = dict(k=len(E), F=float(F), med=float(np.median(L)), lt50=float((L < 50).mean()), cross=float(cross),
                 chi_err=float(np.mean(np.abs(chi_from_gamma(G)[iu] - chi_emp))))
        table[name] = r
        lines.append(f"{name} & {r['k']} & {r['F']:.2f} & {model} & {r['med']:.0f} & {100 * r['lt50']:.0f} & "
                     f"{100 * r['cross']:.0f} & {r['chi_err']:.3f} \\\\")
        print(lines[-1], flush=True)
    npl, nmt = int(plains.sum()), int(mount.sum())
    table['all pairs'] = dict(med=float(np.median(dist[iu])), lt50=float((dist[iu] < 50).mean()), cross=npl * nmt / len(iu[0]))
    print('all pairs', table['all pairs'])
    json.dump(table, open(os.path.join(RES, 'colorado_table.json'), 'w'), indent=1)
    open(os.path.join(FIG, 'tab_colorado.tex'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
