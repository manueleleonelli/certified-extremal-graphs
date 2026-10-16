"""Build LaTeX table fragments and figures from results/*.json."""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results'); OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)

def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else None

def fmt(x, nd=3):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"

# ---------------------------------------------------------------- exactness
def tab_exact():
    rows = (load('exact_new.json') or []) + (load('exact7_new.json') or [])
    if not rows: return
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['d'], r['ktrue'], r['n'])].append(r)
    lines = []
    tot = dict(inst=0, heur=0, cert=0)
    for key in sorted(groups):
        g = groups[key]
        n_inst = len(g); h_ok = sum(r['heur_optimal'] for r in g); c_ok = sum(r['cert_optimal'] for r in g)
        ex = sum(r['exact'] for r in g)
        tot['inst'] += n_inst; tot['heur'] += h_ok; tot['cert'] += c_ok
        lines.append(f"{key[0]} & {key[1]} & {key[2]} & {n_inst} & {h_ok} & {c_ok} & {ex} & {np.mean([r['t_brute'] for r in g]):.1f} & {np.mean([r['t_cert'] for r in g]):.1f} & {np.mean([r['refits'] for r in g]):.0f} \\\\")
    body = "\n".join(lines)
    tex = r"""\begin{tabular}{rrrrrrrrrr}
\toprule
$d$ & $|S^\circ|$ & $n$ & problems & path optimal & certified optimal & full radius & time brute (s) & time cert.\ (s) & refits \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_exact.tex'), 'w').write(tex)
    json.dump(tot, open(os.path.join(OUT, 'exact_totals.json'), 'w'))
    print("exact totals", tot)

def tab_exact_signed():
    rows = (load('exact_signed6.json') or []) + (load('exact_signed7.json') or [])
    if not rows: return
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['d'], r['ktrue'], r['n'])].append(r)
    lines = []; tot = dict(inst=0, heur=0, cert=0, gap=[])
    for key in sorted(groups):
        g = groups[key]
        n_inst = len(g); h_ok = sum(r['heur_optimal'] for r in g); c_ok = sum(r['cert_optimal'] for r in g); ex = sum(r['exact'] for r in g)
        tot['inst'] += n_inst; tot['heur'] += h_ok; tot['cert'] += c_ok
        tot['gap'] += [r['heur'] - r['brute'] for r in g if not r['heur_optimal']]
        gain = np.mean([r['brute_emtp2'] - r['brute'] for r in g])
        lines.append(f"{key[0]} & {key[1]} & {key[2]} & {n_inst} & {h_ok} & {c_ok} & {ex} & {gain:.3f} & {np.mean([r['t_brute'] for r in g]):.1f} & {np.mean([r['t_cert'] for r in g]):.1f} & {np.mean([r['refits'] for r in g]):.0f} \\\\")
    tex = r"""\begin{tabular}{rrrrrrrrrrr}
\toprule
$d$ & $|S^\circ|$ & $n$ & problems & path optimal & certified optimal & full radius & gain over EMTP$_2$ & time brute (s) & time cert.\ (s) & refits \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_exact_signed.tex'), 'w').write(tex)
    tot['gap'] = [float(x) for x in tot['gap']]
    json.dump(tot, open(os.path.join(OUT, 'exact_signed_totals.json'), 'w'))
    print("exact signed totals", tot)

# ---------------------------------------------------------------- recovery
def tab_recovery():
    rows = load('recovery.json')
    if not rows: return
    gen = load('recovery_general.json') or []
    gmap = {(r['d'], r['gtype'], r['n'], r['rep']): r for r in gen}
    for r in rows:
        g = gmap.get((r['d'], r['gtype'], r['n'], r['rep']))
        if g:
            for k, v in g.items():
                if k.startswith('l0'):
                    r[k] = v
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['d'], r['gtype'], r['n'])].append(r)
    names = {'sparse': 'sparse', 'dense': 'dense', 'signed': 'signed'}
    lines = []; lines_ec = []
    summary = {}
    have_gen = bool(gen)
    for key in sorted(groups, key=lambda k: (k[0], ['sparse', 'dense', 'signed'].index(k[1]), k[2])):
        g = groups[key]
        def m(field):
            v = [r[field] for r in g if r.get(field) is not None]
            return np.mean(v) if v else np.nan
        vals = [m('mst_f1'), m('emtp2_f1'), m('l1_f1'), m('gl_f1'), m('eg_f1'), m('l0_f1')] + ([m('l0g_f1')] if have_gen else [])
        best = np.nanmax(vals)
        cells = []
        for v in vals:
            s_ = fmt(v)
            cells.append(r"\textbf{" + s_ + "}" if abs(v - best) < 5e-4 else s_)
        lines.append(f"{key[0]} & {names[key[1]]} & {key[2]} & {g[0]['ktrue']} & " + " & ".join(cells) + " \\\\")
        ec = [fmt(m('eg_oracle_f1')), fmt(m('l0_oracle_f1'))] + ([fmt(m('l0g_oracle_f1'))] if have_gen else []) + [f"{m('eg_k'):.1f}", f"{m('l0_k'):.1f}"] + ([f"{m('l0g_k'):.1f}"] if have_gen else []) + [f"{m('eg_time'):.1f}", f"{m('l0_time'):.1f}"] + ([f"{m('l0g_time'):.1f}"] if have_gen else [])
        lines_ec.append(f"{key[0]} & {names[key[1]]} & {key[2]} & {g[0]['ktrue']} & " + " & ".join(ec) + " \\\\")
        summary[str(key)] = dict(n=len(g), vals=[float(v) for v in vals], l0_k=float(m('l0_k')), eg_k=float(m('eg_k')), l0_time=float(m('l0_time')), eg_time=float(m('eg_time')), gl_time=float(m('gl_time')),
                                 l0_oracle=float(m('l0_oracle_f1')), eg_oracle=float(m('eg_oracle_f1')), l0g_k=float(m('l0g_k')) if have_gen else None, l0g_oracle=float(m('l0g_oracle_f1')) if have_gen else None, l0g_time=float(m('l0g_time')) if have_gen else None)
    ncol = 7 if have_gen else 6
    head = r"$d$ & graph & $n$ & $|S^\circ|$ & MST & EMTP$_2$ & $\ell_1$ & EGL & EGlearn & $\ell_0$" + (r" & $\ell_0^\pm$" if have_gen else "") + r" \\"
    tex = r"""\begin{tabular}{rlrr """ + "r" * ncol + r"""}
\toprule
 & & & & \multicolumn{""" + str(ncol) + r"""}{c}{$F_1$ with BIC or without tuning} \\
\cmidrule(lr){5-""" + str(4 + ncol) + r"""}
""" + head + r"""
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_recovery.tex'), 'w').write(tex)
    n3 = 3 if have_gen else 2
    head_ec = r"$d$ & graph & $n$ & $|S^\circ|$ & EGlearn & $\ell_0$" + (r" & $\ell_0^\pm$" if have_gen else "") + r" & EGlearn & $\ell_0$" + (r" & $\ell_0^\pm$" if have_gen else "") + r" & EGlearn & $\ell_0$" + (r" & $\ell_0^\pm$" if have_gen else "") + r" \\"
    tex = r"""\begin{tabular}{rlrr """ + ("r" * n3 + " ") * 3 + r"""}
\toprule
 & & & & \multicolumn{""" + str(n3) + r"""}{c}{oracle $F_1$} & \multicolumn{""" + str(n3) + r"""}{c}{selected $\hat k$} & \multicolumn{""" + str(n3) + r"""}{c}{time (s)} \\
\cmidrule(lr){5-""" + str(4 + n3) + r"""}\cmidrule(lr){""" + str(5 + n3) + "-" + str(4 + 2 * n3) + r"""}\cmidrule(lr){""" + str(5 + 2 * n3) + "-" + str(4 + 3 * n3) + r"""}
""" + head_ec + r"""
\midrule
""" + "\n".join(lines_ec) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_recovery_ec.tex'), 'w').write(tex)
    json.dump(summary, open(os.path.join(OUT, 'recovery_summary.json'), 'w'), indent=1)
    print("recovery groups", len(groups), "general:", have_gen)

# ---------------------------------------------------------------- certification
def tab_certify():
    rows = load('certify_new.json')
    if not rows: return
    old = load('certify_nocomp.json') or []
    oldmap = collections.defaultdict(list)
    for r in old: oldmap[(r['d'], r['n'])].append(r)
    groups = collections.defaultdict(list)
    for r in rows: groups[(r['d'], r['n'])].append(r)
    lines = []
    summary = {}
    def radius_cells(g, r_):
        done = []; times = []; impr = 0; refits = []
        for r in g:
            pr = [p for p in r['per_r'] if p['r'] == r_]
            if pr:
                p = pr[0]
                if p['radius'] >= r_ and not p['timed_out']:
                    done.append(1); times.append(p['time']); refits.append(p['refits'])
                else:
                    done.append(0)
                impr += int(p['improved'])
            else:
                done.append(0)
        tm = np.mean(times) if times else None
        return done, tm, impr, refits
    for key in sorted(groups):
        g = groups[key]; go = oldmap.get(key, [])
        cells = [str(key[0]), str(key[1]), str(len(g))]
        summ = dict(n_inst=len(g), radii={})
        for r_ in [1, 2, 3, 4]:
            done, tm, impr, refits = radius_cells(g, r_)
            doneo, tmo, impro, refitso = radius_cells(go, r_) if go else ([], None, 0, [])
            summ['radii'][r_] = dict(certified=sum(done), of=len(done), time=tm, improved=impr, refits=float(np.mean(refits)) if refits else None,
                                     time_old=tmo, certified_old=sum(doneo), of_old=len(doneo), refits_old=float(np.mean(refitso)) if refitso else None)
            if r_ == 4:
                continue
            cells += [f"{sum(done)}/{len(done)}", ('--' if tm is None else (f"{tm:.1f}" if tm < 10 else f"{tm:.0f}")),
                      ('--' if tmo is None else (f"{tmo:.1f}" if tmo < 10 else f"{tmo:.0f}"))]
        lines.append(" & ".join(cells) + " \\\\")
        summ['f1_gain'] = float(np.mean([(r['f1_final'] or 0) - r['f1_heur'] for r in g]))
        summary[str(key)] = summ
    body = "\n".join(lines)
    tex = r"""\begin{tabular}{rrr rrr rrr rrr}
\toprule
 & & & \multicolumn{3}{c}{radius 1} & \multicolumn{3}{c}{radius 2} & \multicolumn{3}{c}{radius 3} \\
\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}
$d$ & $n$ & problems & certified & time (s) & pinning only & certified & time (s) & pinning only & certified & time (s) & pinning only \\
\midrule
""" + body + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_certify.tex'), 'w').write(tex)
    json.dump(summary, open(os.path.join(OUT, 'certify_summary.json'), 'w'), indent=1)
    print("certify groups", len(groups))

# ---------------------------------------------------------------- Danube
def danube():
    out = load('danube_new.json') or load('danube.json')
    if not out: return
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    info = pd.read_csv(os.path.join(HERE, 'danube_info.csv'))
    x = info['PlotCoordX'].values; y = info['PlotCoordY'].values
    flow = set(tuple(e) for e in out['flow_edges'])
    d = out['d']
    def draw(ax, edges, title):
        est = set(tuple(e) for e in edges)
        for (i, j) in flow - est:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='0.75', lw=1.2, ls=':', zorder=1)
        for (i, j) in est - flow:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='#b2182b', lw=1.2, ls='--', zorder=2)
        for (i, j) in est & flow:
            ax.plot([x[i], x[j]], [y[i], y[j]], color='black', lw=1.6, zorder=3)
        ax.scatter(x, y, s=70, color='white', edgecolor='black', zorder=4)
        for i in range(d):
            ax.text(x[i], y[i], str(i + 1), fontsize=6, ha='center', va='center', zorder=5)
        ax.set_title(title, fontsize=9); ax.set_axis_off()
    gen = out.get('general')
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 8.0)); axes = axes.ravel()
    kb = out['bic_k']
    ck = str(kb) if str(kb) in out.get('cert', {}) else None
    edges_bic = out['cert'][ck]['edges'] if ck else out['bic_edges']
    f1_bic = out['cert'][ck]['f1_flow'] if ck else out['bic_f1_flow']
    draw(axes[0], out['mst_edges'], f"Minimum spanning tree ($k=30$, $F_1={out['mst_f1_flow']:.2f}$)")
    draw(axes[1], edges_bic, f"Certified EMTP$_2$ solution at the BIC size ($k={kb}$, $F_1={f1_bic:.2f}$)")
    comp = out.get('competitors', {})
    if gen:
        kg = gen['bic_k']; cg = str(kg) if str(kg) in gen.get('cert', {}) else None
        eg_edges = gen['cert'][cg]['edges'] if cg else gen['bic_edges']; f1g = gen['cert'][cg]['f1_flow'] if cg else gen['bic_f1_flow']
        draw(axes[2], eg_edges, f"Certified general solution at the BIC size ($k={kg}$, $F_1={f1g:.2f}$)")
    if 'eg' in comp:
        draw(axes[3], comp['eg']['sparsest_edges'], f"EGlearn, sparsest connected graph ($k={comp['eg']['sparsest_k']}$, $F_1={comp['eg']['sparsest_f1']:.2f}$)")
    plt.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig_danube_all.pdf'))
    # BIC path figure: BIC and F1 against k
    ks = [p['k'] for p in out['path']]; bics = [p['bic'] for p in out['path']]; f1s = [p['f1_flow'] for p in out['path']]
    fig, ax1 = plt.subplots(figsize=(6.2, 3.2))
    ax1.plot(ks, np.array(bics) - min(bics), color='black', lw=1.4)
    ax1.set_xlabel('number of edges $k$'); ax1.set_ylabel('BIC minus its minimum')
    ax1.axvline(kb, color='0.6', ls='--', lw=1)
    ax2 = ax1.twinx(); ax2.plot(ks, f1s, color='#2166ac', lw=1.4); ax2.set_ylabel('$F_1$ against the flow graph', color='#2166ac')
    ax2.tick_params(axis='y', colors='#2166ac')
    plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_danube_path.pdf'))
    # summary table of competitors
    lines = []
    lines.append(f"Minimum spanning tree & 30 & {out['mst_F']:.3f} & EMTP$_2$ & {out['mst_f1_flow']:.3f} \\\\")
    lines.append(f"EMTP$_2$ maximum likelihood support & {len(out['emtp2_edges'])} & {out['emtp2_F']:.3f} & EMTP$_2$ & {out['emtp2_f1_flow']:.3f} \\\\")
    if 'l1' in comp: lines.append(f"$\\ell_1$-penalized EMTP$_2$, BIC & {comp['l1']['k']} & {comp['l1']['F']:.3f} & EMTP$_2$ & {comp['l1']['f1_flow']:.3f} \\\\")
    if 'gl' in comp: lines.append(f"Extremal graphical lasso, BIC & {comp['gl']['k']} & {comp['gl']['F']:.3f} & general & {comp['gl']['f1_flow']:.3f} \\\\")
    if 'eg' in comp:
        lines.append(f"EGlearn, BIC & {comp['eg']['k']} & {comp['eg']['F']:.3f} & general & {comp['eg']['f1_flow']:.3f} \\\\")
        sp = min(comp['eg']['path'], key=lambda p: p['k'])
        lines.append(f"EGlearn, sparsest connected graph & {sp['k']} & {sp['F']:.3f} & general & {sp['f1_flow']:.3f} \\\\")
    lines.append(f"Path with BIC (Algorithm 1) & {kb} & {out['bic_F']:.3f} & EMTP$_2$ & {out['bic_f1_flow']:.3f} \\\\")
    if ck: lines.append(f"Certified search at the BIC size (Algorithm 2) & {kb} & {out['cert'][ck]['F']:.3f} & EMTP$_2$ & {out['cert'][ck]['f1_flow']:.3f} \\\\")
    if gen:
        lines.append(f"Path with BIC, general model & {gen['bic_k']} & {gen['bic_F']:.3f} & general & {gen['bic_f1_flow']:.3f} \\\\")
        if cg: lines.append(f"Certified search at the BIC size, general model & {gen['bic_k']} & {gen['cert'][cg]['F']:.3f} & general & {gen['cert'][cg]['f1_flow']:.3f} \\\\")
    tex = r"""\begin{tabular}{lrrlr}
\toprule
Estimator & edges & profile value & model & $F_1$ flow graph \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_danube.tex'), 'w').write(tex)
    summ = dict(n=out['n'], kb=kb, bic_F=out['bic_F'], bic_f1=out['bic_f1_flow'], cert={k: dict(F=v['F'], f1=v['f1_flow'], per_r=v['per_r']) for k, v in out.get('cert', {}).items()},
                general=(dict(kb=gen['bic_k'], bic_F=gen['bic_F'], bic_f1=gen['bic_f1_flow'], t_path=gen['t_path'], cert={k: dict(F=v['F'], f1=v['f1_flow'], per_r=v['per_r']) for k, v in gen.get('cert', {}).items()},
                              F_general_of_emtp2_graph=gen.get('F_general_of_emtp2_graph'), F_emtp2_of_general_graph=gen.get('F_emtp2_of_general_graph'),
                              path=[(p['k'], p['F'], p['f1_flow']) for p in gen['path']]) if gen else None),
                comp={k: dict(k=v['k'], F=v['F'], f1=v['f1_flow']) for k, v in comp.items()}, mst_f1=out['mst_f1_flow'], emtp2_k=len(out['emtp2_edges']), emtp2_f1=out['emtp2_f1_flow'], t_path=out['t_path'],
                path=[(p['k'], p['F'], p['f1_flow']) for p in out['path']])
    json.dump(summ, open(os.path.join(OUT, 'danube_summary.json'), 'w'), indent=1)
    print("danube", {k: v for k, v in summ.items() if k != 'path'})

# ---------------------------------------------------------------- Colorado
def colorado(p=0.99):
    out = load(f'colorado_p{p}.json')
    if not out: return
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    meta = pd.read_csv(os.path.join(HERE, 'coprcp_meta.csv'))
    el = np.load(os.path.join(HERE, 'coelev.npz'))
    lon = meta['lon'].values; lat = meta['lat'].values; elev = meta['elev'].values
    d = out['d']
    # great-circle distances (km)
    R = 6371.0
    la = np.radians(lat); lo = np.radians(lon)
    dist = R * np.arccos(np.clip(np.sin(la)[:, None] * np.sin(la)[None, :] + np.cos(la)[:, None] * np.cos(la)[None, :] * np.cos(lo[:, None] - lo[None, :]), -1, 1))
    plains = elev < 1900; mount = elev > 2400
    def edge_stats(edges):
        E = [tuple(e) for e in edges]
        L = np.array([dist[i, j] for i, j in E])
        dz = np.array([abs(elev[i] - elev[j]) for i, j in E])
        cross = sum(1 for i, j in E if (plains[i] and mount[j]) or (plains[j] and mount[i]))
        pp = sum(1 for i, j in E if plains[i] and plains[j]); mm = sum(1 for i, j in E if mount[i] and mount[j])
        deg = np.zeros(d, int)
        for i, j in E: deg[i] += 1; deg[j] += 1
        # nearest-neighbour rank: is j among the m nearest of i or vice versa
        order = np.argsort(dist, axis=1)
        rank = np.array([min(int(np.where(order[i] == j)[0][0]), int(np.where(order[j] == i)[0][0])) for i, j in E])
        return dict(k=len(E), mean_km=float(L.mean()), median_km=float(np.median(L)), frac_lt50=float((L < 50).mean()), frac_lt100=float((L < 100).mean()),
                    frac_nn3=float((rank <= 3).mean()), frac_nn5=float((rank <= 5).mean()), mean_dz=float(dz.mean()), max_km=float(L.max()),
                    cross=int(cross), plains=int(pp), mount=int(mm), max_deg=int(deg.max()), hubs=[(str(meta['name'][i]), int(deg[i])) for i in np.argsort(-deg)[:6]])
    def draw(ax, edges, title, highlight=None):
        ax.imshow(el['z'].T, origin='lower', extent=[el['x'].min(), el['x'].max(), el['y'].min(), el['y'].max()], cmap='Greys', aspect='auto', alpha=0.55, vmin=1000, vmax=4300)
        hl = set(tuple(e) for e in highlight) if highlight is not None else set()
        for (i, j) in [tuple(e) for e in edges]:
            col = '#b2182b' if (i, j) in hl else 'black'
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color=col, lw=0.8 if col == 'black' else 1.2, alpha=0.9, zorder=2)
        ax.scatter(lon, lat, s=22, c='white', edgecolor='black', lw=0.6, zorder=3)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-106.05, -103.95); ax.set_ylim(36.95, 41.05)
    kb = out['bic_k']; ck = str(kb)
    E_bic = out['cert'][ck]['edges'] if ck in out.get('cert', {}) else out['bic_edges']
    gen = out.get('general', {})
    kg = gen.get('bic_k'); E_gen = gen['cert'][str(kg)]['edges'] if gen and str(kg) in gen.get('cert', {}) else gen.get('bic_edges')
    comp = out.get('competitors', {})
    k2 = [k for k in out.get('cert', {}) if int(k) != kb]
    E_100 = out['cert'][k2[0]]['edges'] if k2 else None
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 7.4))
    draw(axes[0], out['mst_edges'], f"Minimum spanning tree ($k={d-1}$)")
    draw(axes[1], E_bic, f"Certified solution at the BIC size ($k={kb}$)")
    plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_colorado_mst_bic.pdf'))
    # path figure
    ks = [q['k'] for q in out['path']]; bics = [q['bic'] for q in out['path']]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    ax.plot(ks, np.array(bics) - min(bics), color='black', lw=1.4); ax.axvline(kb, color='0.6', ls='--', lw=1)
    if gen:
        ksg = [q['k'] for q in gen['path']]; bg = [q['bic'] for q in gen['path']]
        ax.plot(ksg, np.array(bg) - min(bg), color='#2166ac', lw=1.4, ls='-.'); ax.axvline(kg, color='#2166ac', ls=':', lw=1)
    ax.set_xlabel('number of edges $k$'); ax.set_ylabel('BIC minus its minimum'); ax.set_ylim(0, 400)
    plt.tight_layout(); fig.savefig(os.path.join(OUT, 'fig_colorado_path.pdf'))
    # table
    rows = [("Minimum spanning tree", out['mst_edges'], out['mst_F'], 'EMTP$_2$')]
    rows.append(("EMTP$_2$ maximum likelihood support", out['emtp2_edges'], out['emtp2_F'], 'EMTP$_2$'))
    if 'l1' in comp: rows.append(("$\\ell_1$-penalized EMTP$_2$, BIC", comp['l1']['edges'], comp['l1']['F'], 'EMTP$_2$'))
    if 'eg' in comp:
        rows.append(("EGlearn, BIC", comp['eg']['edges'], comp['eg']['F'], 'general'))
        rows.append(("EGlearn, sparsest connected graph", comp['eg']['sparsest_edges'], comp['eg']['sparsest_F'], 'general'))
    if E_100 is not None:
        rows.append((f"Search at radius two with {k2[0]} edges (one hour)", E_100, out['cert'][k2[0]]['F'], 'EMTP$_2$'))
    rows.append(("Certified EMTP$_2$ solution at the BIC size", E_bic, out['cert'][ck]['F'] if ck in out.get('cert', {}) else out['bic_F'], 'EMTP$_2$'))
    if E_gen is not None:
        rows.append(("Certified general solution at the BIC size", E_gen, gen['cert'][str(kg)]['F'] if str(kg) in gen.get('cert', {}) else gen['bic_F'], 'general'))
    lines = []; stats = {}
    for name, edges, F, model in rows:
        st = edge_stats(edges); stats[name] = st
        lines.append(f"{name} & {st['k']} & {F:.3f} & {model} & {st['median_km']:.0f} & {100*st['frac_lt50']:.0f} & {100*st['frac_nn3']:.0f} \\\\")
    tex = r"""\begin{tabular}{lrrlrrr}
\toprule
Estimator & edges & profile value & model & median length (km) & \% under 50 km & \% among 3 nearest \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}"""
    open(os.path.join(OUT, 'tab_colorado_distances.tex'), 'w').write(tex)
    # all-pairs reference
    iu = np.triu_indices(d, 1)
    n_pl = int(plains.sum()); n_mt = int(mount.sum())
    allst = dict(median_km=float(np.median(dist[iu])), frac_lt50=float((dist[iu] < 50).mean()), frac_lt100=float((dist[iu] < 100).mean()),
                 n_plains=n_pl, n_mount=n_mt, pairs_cross=n_pl * n_mt, pairs_pp=n_pl * (n_pl - 1) // 2, pairs_mm=n_mt * (n_mt - 1) // 2)
    overlap = None
    if E_gen is not None:
        A = set(tuple(e) for e in E_bic); B = set(tuple(e) for e in E_gen)
        overlap = dict(common=len(A & B), only_emtp2=len(A - B), only_general=len(B - A))
    summ = dict(n=out['n'], kb=kb, kg=kg, stats=stats, allpairs=allst, overlap=overlap, cert={k: v['per_r'] for k, v in out.get('cert', {}).items()},
                cert_general={k: v['per_r'] for k, v in gen.get('cert', {}).items()} if gen else None, t_path=out['t_path'], t_path_general=gen.get('t_path'),
                comp={k: dict(k=v['k'], F=v['F']) for k, v in comp.items()}, mst_F=out['mst_F'], emtp2_k=len(out['emtp2_edges']), bic_F=out['bic_F'])
    json.dump(summ, open(os.path.join(OUT, 'colorado_summary.json'), 'w'), indent=1)
    print("colorado", {k: v for k, v in summ.items() if k not in ('stats',)}); print(stats)

def oa():
    rows = load('oa.json')
    if not rows: return
    json.dump(rows, open(os.path.join(OUT, 'oa_summary.json'), 'w'), indent=1)
    print("oa", rows)

if __name__ == '__main__':
    what = sys.argv[1:] or ['exact', 'recovery', 'certify', 'danube', 'oa']
    for w in what:
        {'exact': tab_exact, 'exact_signed': tab_exact_signed, 'recovery': tab_recovery, 'certify': tab_certify, 'danube': danube, 'oa': oa, 'colorado': colorado}[w]()
