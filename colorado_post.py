"""Summaries and figure for the Colorado validation analyses."""
import os, sys, json, glob
import numpy as np, pandas as pd
from scipy.stats import norm
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hrl0 import *
RES = os.path.join(HERE, 'results')
OUTP = os.path.join(HERE, 'figures'); os.makedirs(OUTP, exist_ok=True)

def chi_from_gamma(G):
    return 2.0 - 2.0 * norm.cdf(np.sqrt(np.maximum(G, 0.0)) / 2.0)

def emp_chi(Y):
    d = Y.shape[1]
    E = np.nan_to_num(Y, nan=-1.0) > 1.0
    O = ~np.isnan(Y)
    C = np.full((d, d), np.nan)
    for i in range(d):
        for j in range(d):
            if i == j: continue
            m = O[:, i] & O[:, j]
            a = (E[m, i] & E[m, j]).sum()
            C[i, j] = 0.5 * (a / max(E[m, j].sum(), 1) + a / max(E[m, i].sum(), 1))
    return C

def chi_summary():
    o = json.load(open(os.path.join(RES, 'colorado_p0.99.json')))
    Gb = np.load(os.path.join(HERE, 'colorado_Gamma_p0.99.npy')); Y = np.load(os.path.join(HERE, 'colorado_Y_p0.99.npy'))
    d = Gb.shape[0]; iu = np.triu_indices(d, 1)
    ce = emp_chi(Y)[iu]
    models = {}
    kb = str(o['bic_k'])
    for name, edges in [('l0', o['cert'][kb]['edges']), ('mst', o['mst_edges']), ('emtp2', o['emtp2_edges'])]:
        f = fit_support(Gb, [tuple(e) for e in edges])
        models[name] = (len(edges), chi_from_gamma(f.R)[iu])
    comp = o['competitors']['eg']
    for name, edges in [('eg_bic', comp['edges']), ('eg_sparsest', comp['sparsest_edges'])]:
        F, Th = fit_general(Gb, [tuple(e) for e in edges])
        models[name] = (len(edges), chi_from_gamma(gamma_from_theta(Th))[iu])
    models['variogram'] = (d * (d - 1) // 2, chi_from_gamma(Gb)[iu])
    out = {k: dict(k=v[0], mae=float(np.nanmean(np.abs(v[1] - ce))), corr=float(np.corrcoef(v[1], ce)[0, 1]),
                   mean_bias=float(np.nanmean(v[1] - ce))) for k, v in models.items()}
    out['emp_mean'] = float(np.nanmean(ce))
    json.dump(out, open(os.path.join(RES, 'colorado_chi.json'), 'w'), indent=1)
    np.save(os.path.join(RES, 'colorado_chi_pairs.npy'), np.vstack([ce, models['l0'][1], models['mst'][1]]))
    return out


def summaries():
    out = {}
    cv = json.load(open(os.path.join(RES, 'colorado_cv.json')))
    folds = {}
    for key, r in cv.items():
        p = {q['k']: q['F_test'] for q in r['path']}
        folds[key] = dict(bic_k=r['bic_k'], bic_test=p[r['bic_k']], mst_test=p[min(p)], best_k=min(p, key=p.get), best_test=min(p.values()),
                          emtp2_k=r['emtp2']['k'], emtp2_test=r['emtp2']['F_test'], sat_general_test=r['saturated_general_F_test'],
                          eg_bic_k=r['eg']['bic_k'], eg_bic_test=r['eg']['bic_F_test'], eg_sp_k=r['eg']['sparsest_k'], eg_sp_test=r['eg']['sparsest_F_test'],
                          eg_best_test=min(q['F_test'] for q in r['eg']['path']), l0_at_eg_sp=p.get(r['eg']['sparsest_k']),
                          share_gain=(p[min(p)] - p[r['bic_k']]) / (p[min(p)] - r['emtp2']['F_test']))
    out['cv'] = folds
    boots = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(RES, 'colorado_boot_[0-9]*.json')))]
    o = json.load(open(os.path.join(RES, 'colorado_p0.99.json')))
    E = set(tuple(e) for e in o['cert'][str(o['bic_k'])]['edges'])
    if boots:
        cnt = {}
        for b in boots:
            for e in b['bic_edges']:
                cnt[tuple(e)] = cnt.get(tuple(e), 0) + 1
        B = len(boots)
        freq_sel = np.array([cnt.get(e, 0) / B for e in E])
        other = [c / B for e, c in cnt.items() if e not in E]
        out['boot'] = dict(B=B, k=[b['bic_k'] for b in boots], median_freq_selected=float(np.median(freq_sel)),
                           frac_selected_ge80=float((freq_sel >= 0.8).mean()), frac_selected_ge50=float((freq_sel >= 0.5).mean()),
                           n_other_ge50=int(sum(1 for c in other if c >= 0.5)), max_other=float(max(other) if other else 0))
        meta = pd.read_csv(os.path.join(HERE, 'coprcp_meta.csv'))
        la, lo = np.radians(meta['lat'].values), np.radians(meta['lon'].values)
        dist = 6371.0 * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None, :]
                                          + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo[:, None] - lo[None, :]), -1, 1))
        L = np.array([dist[e] for e in E])      # same order as freq_sel (iteration over the set E)
        out['boot'].update(frac_short_ge80=float((freq_sel[L < 50] >= 0.8).mean()), frac_long_ge80=float((freq_sel[L > 100] >= 0.8).mean()))
    thr = {}
    for p in [0.95, 0.975]:
        fn = os.path.join(RES, f'colorado_thr_p{p}.json')
        if os.path.exists(fn):
            t = json.load(open(fn)); Et = set(tuple(e) for e in t['bic_edges'])
            thr[p] = dict(n=t['n'], bic_k=t['bic_k'], share_of_099_in_t=len(E & Et) / len(E), share_of_t_in_099=len(E & Et) / len(Et))
    out['thresholds'] = thr
    json.dump(out, open(os.path.join(RES, 'colorado_validation.json'), 'w'), indent=1, default=float)
    return out

def figure():
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    cv = json.load(open(os.path.join(RES, 'colorado_cv.json')))
    pairs = np.load(os.path.join(RES, 'colorado_chi_pairs.npy'))
    Gb = np.load(os.path.join(HERE, 'colorado_Gamma_p0.99.npy')); iu = np.triu_indices(64, 1)
    cv_var = chi_from_gamma(Gb)[iu]
    fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.9))
    styles = {'train_odd': ('black', 'train on odd years'), 'train_even': ('#2166ac', 'train on even years')}
    for key, r in cv.items():
        col, lab = styles[key]
        ks = [q['k'] for q in r['path']]; ft = np.array([q['F_test'] for q in r['path']])
        ax[0].plot(ks, ft, color=col, lw=1.4, label=lab)
        kb = r['bic_k']; ax[0].plot([kb], [ft[ks.index(kb)]], 'o', color=col, ms=6)
        ax[0].axhline(r['emtp2']['F_test'], color=col, ls='--', lw=0.9)
        egk = [q['k'] for q in r['eg']['path'] if q['k'] <= 420]; egf = [q['F_test'] for q in r['eg']['path'] if q['k'] <= 420]
        ax[0].plot(egk, egf, 'x', color=col, ms=6)
    ax[0].set_xlabel('number of edges $k$'); ax[0].set_ylabel('held-out profile value (lower is better)')
    ax[0].set_xlim(55, 430); ax[0].legend(fontsize=8, frameon=False)
    ax[0].set_title('(a) held-out fit along the path', fontsize=9)
    ax[1].plot(cv_var, pairs[2], '.', color='0.65', ms=3, label='minimum spanning tree')
    ax[1].plot(cv_var, pairs[1], '.', color='black', ms=3, label='selected graph')
    ax[1].plot([0, 1], [0, 1], color='#b2182b', lw=0.9)
    ax[1].set_xlabel(r'$\chi$ implied by the empirical variogram'); ax[1].set_ylabel(r'$\chi$ implied by the fitted graph')
    ax[1].set_xlim(0, 0.9); ax[1].set_ylim(0, 0.9); ax[1].legend(fontsize=8, frameon=False, loc='upper left')
    ax[1].set_title('(b) extremal correlations of all pairs', fontsize=9)
    plt.tight_layout(); fig.savefig(os.path.join(OUTP, 'fig_colorado_validation.pdf'))

if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    if what in ('chi', 'all'): print(json.dumps(chi_summary(), indent=1))
    if what in ('summ', 'all'): print(json.dumps(summaries(), indent=1, default=float))
    if what in ('fig', 'all'): figure()
