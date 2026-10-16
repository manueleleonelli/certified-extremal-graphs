"""
Experiments for the paper.  Usage: python experiments.py {exact|recovery|certify|danube|oa}
Results are written as JSON to results/.
"""
import sys, os, json, time, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrl0 import *

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(OUT, exist_ok=True)

def dump(name, obj):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(obj, f, indent=1, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o))

# ---------------------------------------------------------------------------
def study_exact():
    """brute force vs heuristic path vs certified search on small graphs"""
    rng = np.random.default_rng(2026)
    prev = os.path.join(OUT, 'exact_new.json')
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['ktrue'], r['n'], r['rep']) for r in rows)
    configs = [(6, 8, 50), (6, 8, 200), (6, 10, 50), (6, 10, 200), (7, 9, 50)]
    reps = {6: 4, 7: 2}
    for d, ktrue, n in configs:
        for rep in range(reps[d]):
            E, w, Gam = random_hr_model(d, ktrue, rng)
            Gb = emp_vario(rmpareto_HR(n, Gam, rng))
            if (d, ktrue, n, rep) in done:
                continue
            kmax = min(ktrue + 2, d * (d - 1) // 2)
            t = time.time(); path = forward_path(Gb, kmax); tpath = time.time() - t
            for k in range(d - 1, kmax + 1):
                if k not in path:      # the saturated support has fewer than k edges
                    continue
                t = time.time(); bf, bfS = brute_force(Gb, k); tb = time.time() - t
                cert = SwapCertifier(Gb, path[k].edges)
                out = cert.certify(time_limit=600)
                # radius at which the certificate would have stopped if run only to r=2
                rows.append(dict(d=d, ktrue=ktrue, n=n, rep=rep, k=k, brute=bf, heur=path[k].F, cert=out['F'],
                                 exact=bool(out['exact']), radius=out['radius'], t_brute=tb, t_cert=out['time'],
                                 t_path=tpath, heur_optimal=bool(np.isclose(path[k].F, bf, atol=1e-7)),
                                 cert_optimal=bool(np.isclose(out['F'], bf, atol=1e-7)),
                                 refits=out['stats']['refits'], f1_true=f1_score(E, bfS)[2]))
                print(rows[-1], flush=True)
            dump('exact_new.json', rows)
    dump('exact_new.json', rows)

# ---------------------------------------------------------------------------
def select_bic_path(path_items, n):
    """path_items: list of (k, F, edges); returns the BIC-selected item"""
    best = None
    for k, F, edges in path_items:
        b = bic(F, k, n)
        if best is None or b < best[0]:
            best = (b, k, F, edges)
    return best

def study_recovery():
    rng = np.random.default_rng(7)
    rows = []
    reps = 15
    rho_grid_l1 = np.geomspace(0.001, 0.5, 10)
    rho_grid_eg = np.geomspace(0.01, 0.7, 10)
    rho_grid_gl = np.geomspace(0.005, 0.5, 8)
    for d in [10, 20]:
        for gtype in ['sparse', 'dense', 'signed']:
            ktrue = int(np.floor(1.3 * (d - 1))) if gtype != 'dense' else 2 * (d - 1)
            for n in [100, 250, 1000]:
                for rep in range(reps):
                    if gtype == 'signed':
                        E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
                    else:
                        E, w, Gam = random_hr_model(d, ktrue, rng)
                    Y = rmpareto_HR(n, Gam, rng); Gb = emp_vario(Y)
                    res = dict(d=d, gtype=gtype, ktrue=ktrue, n=n, rep=rep)
                    # ours: forward path + BIC
                    kmax = min(int(2.6 * (d - 1)), d * (d - 1) // 2)
                    t = time.time(); path = forward_path(Gb, kmax)
                    items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
                    b, kb, Fb, Eb = select_bic_path(items, n)
                    res['l0_time'] = time.time() - t
                    res['l0_k'] = kb; res['l0_f1'] = f1_score(E, Eb)[2]
                    res['l0_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
                    res['l0_F_at_ktrue'] = path[ktrue].F if ktrue in path else None
                    # MST only
                    res['mst_f1'] = f1_score(E, mst_edges(Gb))[2]
                    # EMTP2 MLE support
                    t = time.time(); full = emtp2_mle(Gb); E0 = emtp2_support(full)
                    res['emtp2_time'] = time.time() - t; res['emtp2_k'] = len(E0); res['emtp2_f1'] = f1_score(E, E0)[2]
                    res['F_full'] = full.F
                    # l1-EMTP2 path + BIC (refit on support)
                    t = time.time(); items = []
                    for rho in rho_grid_l1:
                        S = l1_emtp2(Gb, rho)
                        if len(S) < d - 1 or not is_connected(d, S):
                            continue
                        fit = fit_support(Gb, S)
                        items.append((len(S), fit.F, S))
                    if items:
                        b, kb, Fb, Eb = select_bic_path(items, n)
                        res['l1_k'] = kb; res['l1_f1'] = f1_score(E, Eb)[2]
                        res['l1_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
                    res['l1_time'] = time.time() - t
                    # EGlearn path + BIC (general HR refit on support)
                    t = time.time(); items = []
                    for rho in rho_grid_eg:
                        S = eglearn(Gb, rho)
                        if len(S) < d - 1 or not is_connected(d, S):
                            continue
                        F, _ = fit_general(Gb, S)
                        if np.isfinite(F):
                            items.append((len(S), F, S))
                    if items:
                        b, kb, Fb, Eb = select_bic_path(items, n)
                        res['eg_k'] = kb; res['eg_f1'] = f1_score(E, Eb)[2]
                        res['eg_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
                    res['eg_time'] = time.time() - t
                    # extremal graphical lasso (general) + BIC
                    t = time.time(); items = []
                    for rho in rho_grid_gl:
                        try:
                            S = extremal_glasso(Gb, rho)
                        except Exception:
                            continue
                        if len(S) < d - 1 or not is_connected(d, S):
                            continue
                        F, _ = fit_general(Gb, S)
                        if np.isfinite(F):
                            items.append((len(S), F, S))
                    if items:
                        b, kb, Fb, Eb = select_bic_path(items, n)
                        res['gl_k'] = kb; res['gl_f1'] = f1_score(E, Eb)[2]
                        res['gl_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
                    res['gl_time'] = time.time() - t
                    rows.append(res)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
                    dump('recovery.json', rows)
    dump('recovery.json', rows)

# ---------------------------------------------------------------------------
def study_certify():
    rng = np.random.default_rng(11)
    rows = []
    for d, rmax, tl, reps in [(10, 3, 150, 4), (15, 3, 150, 4), (20, 2, 150, 4), (31, 2, 900, 2)]:
        ktrue = int(np.floor(1.3 * (d - 1)))
        for n in [200, 1000]:
            for rep in range(reps if isinstance(reps, int) else 4):
                E, w, Gam = random_hr_model(d, ktrue, rng)
                Gb = emp_vario(rmpareto_HR(n, Gam, rng))
                t = time.time(); path = forward_path(Gb, ktrue + 3); tpath = time.time() - t
                items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
                b, kb, Fb, Eb = select_bic_path(items, n)
                for k in sorted(set([ktrue, kb])):
                    cert = SwapCertifier(Gb, path[k].edges)
                    per_r = []
                    t0 = time.time()
                    out = None
                    for r in range(1, rmax + 1):
                        remaining = tl - (time.time() - t0)
                        if remaining <= 0:
                            break
                        out = cert.certify(r_max=r, time_limit=remaining, restart=True)
                        per_r.append(dict(r=r, radius=out['radius'], time=time.time() - t0, refits=out['stats']['refits'],
                                          F=out['F'], improved=bool(out['F'] < path[k].F - 1e-7), timed_out=out['timed_out']))
                        if out['timed_out']:
                            break
                    rows.append(dict(d=d, n=n, rep=rep, k=k, k_is_bic=(k == kb), ktrue=ktrue, heur=path[k].F,
                                     t_path=tpath, per_r=per_r, final=out['F'] if out else None,
                                     f1_heur=f1_score(E, path[k].edges)[2], f1_final=f1_score(E, out['edges'])[2] if out else None))
                    print(rows[-1], flush=True)
                    dump('certify.json', rows)
    dump('certify.json', rows)

# ---------------------------------------------------------------------------
def study_oa():
    """outer approximation with the resistance big-M cuts: the lower bound stays
    at the saturated value (illustration of the relaxation proposition)"""
    import hrsparse as old
    rng = np.random.default_rng(3)
    rows = []
    for d, ktrue, n in [(8, 10, 200), (10, 13, 200)]:
        E, w, Gam = random_hr_model(d, ktrue, rng)
        Gb = emp_vario(rmpareto_HR(n, Gam, rng))
        oa = old.OAPath(Gb, emtp2=True, tol=1e-6)
        res = oa.run(kmax=ktrue, kmin=ktrue, time_limit=120, max_iter=400)
        r = res[ktrue]
        full = emtp2_mle(Gb)
        rows.append(dict(d=d, k=ktrue, n=n, UB=r['F'], LB=r['LB'], F_full=full.F, iters=r['iters'], time=r['time'], ncuts=r['ncuts'], status=r['status']))
        print(rows[-1], flush=True)
    dump('oa.json', rows)

# ---------------------------------------------------------------------------
def study_danube():
    import pandas as pd
    X = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'danube_clustered.npy'))
    flow = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'danube_flow_edges.npy'))
    flow_edges = sorted((int(min(a, b)) - 1, int(max(a, b)) - 1) for a, b in flow)
    n_all, d = X.shape
    p = 0.9
    Y = data2mpareto(X, p); n = Y.shape[0]
    Gb = emp_vario(Y)
    out = dict(n_all=int(n_all), d=int(d), p=p, n=int(n), flow_edges=flow_edges)
    # path and BIC
    kmax = 75
    t = time.time(); path = forward_path(Gb, kmax); out['t_path'] = time.time() - t
    out['path'] = [dict(k=k, F=f.F, bic=bic(f.F, k, n), edges=list(f.edges), f1_flow=f1_score(flow_edges, f.edges)[2]) for k, f in sorted(path.items())]
    items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
    b, kb, Fb, Eb = select_bic_path(items, n)
    out['bic_k'] = kb; out['bic_F'] = Fb; out['bic_edges'] = list(Eb); out['bic_f1_flow'] = f1_score(flow_edges, Eb)[2]
    out['mst_edges'] = mst_edges(Gb); out['mst_f1_flow'] = f1_score(flow_edges, out['mst_edges'])[2]
    out['mst_F'] = path[d - 1].F
    # saturated EMTP2
    full = emtp2_mle(Gb); E0 = emtp2_support(full)
    out['emtp2_edges'] = E0; out['emtp2_F'] = full.F; out['emtp2_f1_flow'] = f1_score(flow_edges, E0)[2]
    # certification at the BIC k and at k = 30, 40
    out['cert'] = {}
    for k in sorted(set([kb, 35, 45])):
        if k not in path: continue
        cert = SwapCertifier(Gb, path[k].edges)
        per_r = []; t0 = time.time(); res = None
        for r in [1, 2]:
            remaining = 1500 - (time.time() - t0)
            if remaining <= 0: break
            res = cert.certify(r_max=r, time_limit=remaining)
            per_r.append(dict(r=r, radius=res['radius'], time=time.time() - t0, refits=res['stats']['refits'], F=res['F'],
                              improved=bool(res['F'] < path[k].F - 1e-7), timed_out=res['timed_out']))
            print(k, per_r[-1], flush=True)
            if res['timed_out']: break
        out['cert'][str(k)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'], f1_flow=f1_score(flow_edges, res['edges'])[2])
        dump('danube.json', out)
    # competitors with BIC
    comp = {}
    items = []
    for rho in np.geomspace(0.001, 0.5, 12):
        S = l1_emtp2(Gb, rho)
        if len(S) >= d - 1 and is_connected(d, S):
            items.append((len(S), fit_support(Gb, S).F, S))
    if items:
        b, kb2, Fb2, Eb2 = select_bic_path(items, n)
        comp['l1'] = dict(k=kb2, F=Fb2, edges=list(Eb2), f1_flow=f1_score(flow_edges, Eb2)[2],
                          path=[dict(k=k, F=F, f1_flow=f1_score(flow_edges, e)[2]) for k, F, e in items])
    items = []
    for rho in np.geomspace(0.01, 0.7, 14):
        S = eglearn(Gb, rho)
        if len(S) >= d - 1 and is_connected(d, S):
            F, _ = fit_general(Gb, S)
            if np.isfinite(F): items.append((len(S), F, S))
    if items:
        b, kb2, Fb2, Eb2 = select_bic_path(items, n)
        comp['eg'] = dict(k=kb2, F=Fb2, edges=list(Eb2), f1_flow=f1_score(flow_edges, Eb2)[2],
                          path=[dict(k=k, F=F, f1_flow=f1_score(flow_edges, e)[2]) for k, F, e in items])
    items = []
    for rho in np.geomspace(0.005, 0.5, 10):
        try:
            S = extremal_glasso(Gb, rho)
        except Exception:
            continue
        if len(S) >= d - 1 and is_connected(d, S):
            F, _ = fit_general(Gb, S)
            if np.isfinite(F): items.append((len(S), F, S))
    if items:
        b, kb2, Fb2, Eb2 = select_bic_path(items, n)
        comp['gl'] = dict(k=kb2, F=Fb2, edges=list(Eb2), f1_flow=f1_score(flow_edges, Eb2)[2],
                          path=[dict(k=k, F=F, f1_flow=f1_score(flow_edges, e)[2]) for k, F, e in items])
    out['competitors'] = comp
    # general (signed) refit of our BIC graph, for reference
    Fg, _ = fit_general(Gb, Eb); out['bic_F_general'] = Fg
    dump('danube.json', out)
    print("danube done", kb, out['bic_f1_flow'], {k: v['f1_flow'] for k, v in comp.items()}, flush=True)


# ---------------------------------------------------------------------------
def _replay_exact(seed, configs, reps, kmax_rule, oldrows, outname, signed=False):
    old = {(r['d'], r['ktrue'], r['n'], r['rep'], r['k']): r for r in oldrows}
    rng = np.random.default_rng(seed)
    prev = os.path.join(OUT, outname)
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['ktrue'], r['n'], r['rep'], r['k']) for r in rows)
    for d, ktrue, n in configs:
        for rep in range(reps[d]):
            E, w, Gam = random_hr_model(d, ktrue, rng)
            Gb = emp_vario(rmpareto_HR(n, Gam, rng))
            kmax = kmax_rule(d, ktrue)
            if all((d, ktrue, n, rep, k) in done for k in range(d - 1, kmax + 1) if (d, ktrue, n, rep, k) in old):
                continue
            t = time.time(); path = forward_path(Gb, kmax, signed=signed); tpath = time.time() - t
            for k in range(d - 1, kmax + 1):
                if k not in path:
                    continue
                key = (d, ktrue, n, rep, k)
                if key not in old or key in done:
                    continue
                bf = old[key]['brute']
                cert = SwapCertifier(Gb, path[k].edges, signed=signed)
                out = cert.certify(time_limit=600)
                rows.append(dict(d=d, ktrue=ktrue, n=n, rep=rep, k=k, brute=bf, heur=path[k].F, cert=out['F'],
                                 exact=bool(out['exact']), radius=out['radius'], t_brute=old[key]['t_brute'], t_cert=out['time'],
                                 t_path=tpath, heur_optimal=bool(np.isclose(path[k].F, bf, atol=1e-7)),
                                 cert_optimal=bool(np.isclose(out['F'], bf, atol=1e-7)),
                                 refits=out['stats']['refits'], refits_old=old[key]['refits'], t_cert_old=old[key]['t_cert'],
                                 f1_true=old[key]['f1_true']))
                print(rows[-1], flush=True)
                dump(outname, rows)
    dump(outname, rows)
    return rows

def study_exact_recert():
    """replay the exact study with the new certifier; brute-force values are reused"""
    oldrows = json.load(open(os.path.join(OUT, 'exact.json')))
    configs = [(6, 8, 50), (6, 8, 200), (6, 10, 50), (6, 10, 200), (7, 9, 50)]
    _replay_exact(2026, configs, {6: 4, 7: 2}, lambda d, k: min(k + 2, d * (d - 1) // 2), oldrows, 'exact_new.json')
    old7 = json.load(open(os.path.join(OUT, 'exact7.json')))
    _replay_exact(77, [(7, 9, 200)], {7: 1}, lambda d, k: 10, old7, 'exact7_new.json')

def study_exact_signed(which='6'):
    """exhaustive enumeration for the general model on small signed graphs"""
    if which == '6':
        rng = np.random.default_rng(2027); configs = [(6, 8, 200), (6, 9, 100)]; reps = {6: 3}; extra = 2; outname = 'exact_signed6.json'
    else:
        rng = np.random.default_rng(2028); configs = [(7, 9, 200)]; reps = {7: 1}; extra = 1; outname = 'exact_signed7.json'
    prev = os.path.join(OUT, outname)
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['ktrue'], r['n'], r['rep'], r['k']) for r in rows)
    for d, ktrue, n in configs:
        for rep in range(reps[d]):
            E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
            Gb = emp_vario(rmpareto_HR(n, Gam, rng))
            kmax = min(ktrue + extra, d * (d - 1) // 2)
            if all((d, ktrue, n, rep, k) in done for k in range(d - 1, kmax + 1)):
                continue
            t = time.time(); path = forward_path(Gb, kmax, signed=True); tpath = time.time() - t
            for k in range(d - 1, kmax + 1):
                if k not in path or (d, ktrue, n, rep, k) in done:
                    continue
                t = time.time(); bf, bfS = brute_force(Gb, k, signed=True); tb = time.time() - t
                bf0, _ = brute_force(Gb, k)
                cert = SwapCertifier(Gb, path[k].edges, signed=True)
                out = cert.certify(time_limit=900)
                rows.append(dict(d=d, ktrue=ktrue, n=n, rep=rep, k=k, brute=bf, brute_emtp2=bf0, heur=path[k].F, cert=out['F'],
                                 exact=bool(out['exact']), radius=out['radius'], t_brute=tb, t_cert=out['time'], t_path=tpath,
                                 heur_optimal=bool(np.isclose(path[k].F, bf, atol=1e-7)), cert_optimal=bool(np.isclose(out['F'], bf, atol=1e-7)),
                                 refits=out['stats']['refits'], f1_true=f1_score(E, bfS)[2], neg_edges=int((w < 0).sum())))
                print(rows[-1], flush=True)
                dump(outname, rows)
    dump(outname, rows)

def study_recovery_general():
    """the general (signed) l0 estimator on the data of the recovery study"""
    rng = np.random.default_rng(7)
    prev = os.path.join(OUT, 'recovery_general.json')
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['gtype'], r['n'], r['rep']) for r in rows)
    reps = 15
    for d in [10, 20]:
        for gtype in ['sparse', 'dense', 'signed']:
            ktrue = int(np.floor(1.3 * (d - 1))) if gtype != 'dense' else 2 * (d - 1)
            for n in [100, 250, 1000]:
                for rep in range(reps):
                    if gtype == 'signed':
                        E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
                    else:
                        E, w, Gam = random_hr_model(d, ktrue, rng)
                    Y = rmpareto_HR(n, Gam, rng); Gb = emp_vario(Y)
                    if (d, gtype, n, rep) in done:
                        continue
                    res = dict(d=d, gtype=gtype, ktrue=ktrue, n=n, rep=rep)
                    kmax = min(int(2.6 * (d - 1)), d * (d - 1) // 2)
                    for tag, sg in [('l0', False), ('l0g', True)]:
                        t = time.time(); path = forward_path(Gb, kmax, signed=sg)
                        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
                        b, kb, Fb, Eb = select_bic_path(items, n)
                        res[tag + '_time'] = time.time() - t
                        res[tag + '_k'] = kb; res[tag + '_f1'] = f1_score(E, Eb)[2]
                        res[tag + '_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
                        res[tag + '_F_at_ktrue'] = path[ktrue].F if ktrue in path else None
                    rows.append(res)
                    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}, flush=True)
                    dump('recovery_general.json', rows)
    dump('recovery_general.json', rows)

def study_certify_new():
    rng = np.random.default_rng(11)
    prev = os.path.join(OUT, 'certify_new.json')
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['n'], r['rep']) for r in rows)
    for d, rmax, tl, reps in [(10, 4, 150, 4), (15, 3, 300, 4), (20, 3, 600, 2), (31, 3, 1200, 2)]:
        ktrue = int(np.floor(1.3 * (d - 1)))
        for n in [200, 1000]:
            for rep in range(reps):
                E, w, Gam = random_hr_model(d, ktrue, rng)
                Gb = emp_vario(rmpareto_HR(n, Gam, rng))
                if (d, n, rep) in done:
                    continue
                t = time.time(); path = forward_path(Gb, ktrue + 3); tpath = time.time() - t
                items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
                b, kb, Fb, Eb = select_bic_path(items, n)
                for k in sorted(set([ktrue, kb])):
                    cert = SwapCertifier(Gb, path[k].edges)
                    per_r = []
                    t0 = time.time()
                    out = None
                    for r in range(1, rmax + 1):
                        remaining = tl - (time.time() - t0)
                        if remaining <= 0:
                            break
                        out = cert.certify(r_max=r, time_limit=remaining, restart=True)
                        per_r.append(dict(r=r, radius=out['radius'], time=time.time() - t0, refits=out['stats']['refits'],
                                          F=out['F'], improved=bool(out['F'] < path[k].F - 1e-7), timed_out=out['timed_out']))
                        if out['timed_out']:
                            break
                    rows.append(dict(d=d, n=n, rep=rep, k=k, k_is_bic=(k == kb), ktrue=ktrue, heur=path[k].F,
                                     t_path=tpath, per_r=per_r, final=out['F'] if out else None,
                                     f1_heur=f1_score(E, path[k].edges)[2], f1_final=f1_score(E, out['edges'])[2] if out else None))
                    print(rows[-1], flush=True)
                    dump('certify_new.json', rows)
    dump('certify_new.json', rows)

def study_certify_nocomp():
    """the same certificates without the completion of the dual point (ablation)"""
    rng = np.random.default_rng(11)
    prev = os.path.join(OUT, 'certify_nocomp.json')
    rows = json.load(open(prev)) if os.path.exists(prev) else []
    done = set((r['d'], r['n'], r['rep']) for r in rows)
    for d, rmax, tl, reps in [(10, 3, 150, 4), (15, 3, 300, 4), (20, 2, 300, 2), (31, 2, 600, 2)]:
        ktrue = int(np.floor(1.3 * (d - 1)))
        for n in [200, 1000]:
            for rep in range(reps):
                E, w, Gam = random_hr_model(d, ktrue, rng)
                Gb = emp_vario(rmpareto_HR(n, Gam, rng))
                if (d, n, rep) in done:
                    continue
                t = time.time(); path = forward_path(Gb, ktrue + 3); tpath = time.time() - t
                items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
                b, kb, Fb, Eb = select_bic_path(items, n)
                for k in sorted(set([ktrue, kb])):
                    cert = SwapCertifier(Gb, path[k].edges, completion=False)
                    per_r = []; t0 = time.time(); out = None
                    for r in range(1, rmax + 1):
                        remaining = tl - (time.time() - t0)
                        if remaining <= 0:
                            break
                        out = cert.certify(r_max=r, time_limit=remaining, restart=True)
                        per_r.append(dict(r=r, radius=out['radius'], time=time.time() - t0, refits=out['stats']['refits'],
                                          F=out['F'], improved=bool(out['F'] < path[k].F - 1e-7), timed_out=out['timed_out']))
                        if out['timed_out']:
                            break
                    rows.append(dict(d=d, n=n, rep=rep, k=k, k_is_bic=(k == kb), ktrue=ktrue, heur=path[k].F, t_path=tpath, per_r=per_r))
                    print(rows[-1], flush=True)
                    dump('certify_nocomp.json', rows)
    dump('certify_nocomp.json', rows)

def _certify_path(Gb, path, k, radii, budget, signed=False):
    cert = SwapCertifier(Gb, path[k].edges, signed=signed)
    per_r = []; t0 = time.time(); res = None
    for r in radii:
        remaining = budget - (time.time() - t0)
        if remaining <= 0: break
        res = cert.certify(r_max=r, time_limit=remaining)
        per_r.append(dict(r=r, radius=res['radius'], time=time.time() - t0, refits=res['stats']['refits'], F=res['F'],
                          improved=bool(res['F'] < path[k].F - 1e-7), timed_out=res['timed_out']))
        print(k, per_r[-1], flush=True)
        if res['timed_out']: break
    return per_r, res

def study_danube_new():
    here = os.path.dirname(os.path.abspath(__file__))
    old = json.load(open(os.path.join(OUT, 'danube.json')))
    X = np.load(os.path.join(here, 'danube_clustered.npy'))
    flow = np.load(os.path.join(here, 'danube_flow_edges.npy'))
    flow_edges = sorted((int(min(a, b)) - 1, int(max(a, b)) - 1) for a, b in flow)
    n_all, d = X.shape
    p = 0.9
    Y = data2mpareto(X, p); n = Y.shape[0]
    Gb = emp_vario(Y)
    prev = os.path.join(OUT, 'danube_new.json')
    out = json.load(open(prev)) if os.path.exists(prev) else dict(old)
    kmax = 75
    t = time.time(); path = forward_path(Gb, kmax, checkpoint=os.path.join(OUT, 'danube_path_ckpt.json'))
    if 'path_new' not in out:
        out['t_path'] = time.time() - t
    items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
    b, kb, Fb, Eb = select_bic_path(items, n)
    out['path'] = [dict(k=k, F=f.F, bic=bic(f.F, k, n), edges=list(f.edges), f1_flow=f1_score(flow_edges, f.edges)[2]) for k, f in sorted(path.items())]
    out['bic_k'] = kb; out['bic_F'] = Fb; out['bic_edges'] = list(Eb); out['bic_f1_flow'] = f1_score(flow_edges, Eb)[2]; out['path_new'] = True
    out.setdefault('cert', {})
    if 'cert_old' not in out:
        out['cert_old'] = old['cert']; out['cert'] = {}
    dump('danube_new.json', out)
    for k in sorted(set([kb, 35, 45])):
        if str(k) in out['cert']:
            continue
        per_r, res = _certify_path(Gb, path, k, [1, 2, 3] if k == kb else [1, 2], 1800)
        out['cert'][str(k)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'], f1_flow=f1_score(flow_edges, res['edges'])[2])
        dump('danube_new.json', out)
    if 'general' not in out or 'cert' not in out['general'] or not out['general']['cert']:
        t = time.time(); pathg = forward_path(Gb, kmax, signed=True, checkpoint=os.path.join(OUT, 'danube_pathg_ckpt.json')); tg = time.time() - t
        itemsg = [(k, f.F, f.edges) for k, f in sorted(pathg.items())]
        bg, kbg, Fbg, Ebg = select_bic_path(itemsg, n)
        out['general'] = dict(t_path=tg, path=[dict(k=k, F=f.F, bic=bic(f.F, k, n), f1_flow=f1_score(flow_edges, f.edges)[2]) for k, f in sorted(pathg.items())],
                              bic_k=kbg, bic_F=Fbg, bic_edges=list(Ebg), bic_f1_flow=f1_score(flow_edges, Ebg)[2], cert={})
        dump('danube_new.json', out)
        per_r, res = _certify_path(Gb, pathg, kbg, [1, 2], 1800, signed=True)
        out['general']['cert'][str(kbg)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'], f1_flow=f1_score(flow_edges, res['edges'])[2])
        out['general']['F_general_of_emtp2_graph'] = fit_general(Gb, out['cert'][str(kb)]['edges'])[0]
        out['general']['F_emtp2_of_general_graph'] = profile_value(Gb, res['edges'])
        dump('danube_new.json', out)
    print("danube done", flush=True)

def study_colorado(p=0.99, kmax=200, kmax_g=180):
    here = os.path.dirname(os.path.abspath(__file__))
    Yc = np.load(os.path.join(here, f'colorado_Y_p{p}.npy'))
    Gb = np.load(os.path.join(here, f'colorado_Gamma_p{p}.npy'))
    n, d = Yc.shape
    prev = os.path.join(OUT, f'colorado_p{p}.json')
    out = json.load(open(prev)) if os.path.exists(prev) else dict(n=int(n), d=int(d), p=p)
    if 'emtp2_edges' not in out:
        full = emtp2_mle(Gb); E0 = emtp2_support(full)
        out['emtp2_edges'] = E0; out['emtp2_F'] = full.F; out['mst_edges'] = mst_edges(Gb)
        dump(f'colorado_p{p}.json', out)
    t = time.time(); path = forward_path(Gb, kmax, checkpoint=os.path.join(OUT, f'colorado_path_ckpt_p{p}.json'))
    if 'bic_k' not in out:
        out['t_path'] = time.time() - t
    items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
    b, kb, Fb, Eb = select_bic_path(items, n)
    out['path'] = [dict(k=k, F=f.F, bic=bic(f.F, k, n), edges=list(f.edges)) for k, f in sorted(path.items())]
    out['bic_k'] = kb; out['bic_F'] = Fb; out['bic_edges'] = list(Eb); out['mst_F'] = path[d - 1].F
    out.setdefault('cert', {})
    dump(f'colorado_p{p}.json', out)
    if str(kb) not in out['cert']:
        per_r, res = _certify_path(Gb, path, kb, [1], 1200)
        out['cert'][str(kb)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'])
        dump(f'colorado_p{p}.json', out)
    k2 = 100 if 100 in path else min(path, key=lambda k: abs(k - 100))
    if str(k2) not in out['cert']:
        per_r, res = _certify_path(Gb, path, k2, [1, 2], 3600)
        out['cert'][str(k2)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'])
        dump(f'colorado_p{p}.json', out)
    if 'general' not in out or not out['general'].get('cert'):
        t = time.time(); pathg = forward_path(Gb, kmax_g, signed=True, checkpoint=os.path.join(OUT, f'colorado_pathg_ckpt_p{p}.json')); tg = time.time() - t
        itemsg = [(k, f.F, f.edges) for k, f in sorted(pathg.items())]
        bg, kbg, Fbg, Ebg = select_bic_path(itemsg, n)
        out['general'] = dict(t_path=tg, path=[dict(k=k, F=f.F, bic=bic(f.F, k, n)) for k, f in sorted(pathg.items())],
                              bic_k=kbg, bic_F=Fbg, bic_edges=list(Ebg), cert={})
        dump(f'colorado_p{p}.json', out)
        per_r, res = _certify_path(Gb, pathg, kbg, [1], 1200, signed=True)
        out['general']['cert'][str(kbg)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'])
        out['general']['F_general_of_emtp2_graph'] = fit_general(Gb, out['cert'][str(kb)]['edges'])[0]
        dump(f'colorado_p{p}.json', out)
    comp = out.get('competitors', {})
    if 'eg' not in comp:
        items = []
        t = time.time()
        for rho in np.geomspace(0.02, 2.0, 12):
            S = eglearn(Gb, rho)
            if len(S) >= d - 1 and is_connected(d, S):
                F, _ = fit_general(Gb, S)
                if np.isfinite(F): items.append((len(S), F, S))
        if items:
            b2, kb2, Fb2, Eb2 = select_bic_path(items, n)
            sp = min(items, key=lambda x: x[0])
            comp['eg'] = dict(k=kb2, F=Fb2, edges=list(Eb2), sparsest_k=sp[0], sparsest_F=sp[1], sparsest_edges=list(sp[2]),
                              path=[dict(k=k, F=F) for k, F, e in items], time=time.time() - t)
        out['competitors'] = comp
        dump(f'colorado_p{p}.json', out)
    if 'l1' not in comp:
        items = []
        t = time.time()
        for rho in np.geomspace(0.002, 0.5, 8):
            S = l1_emtp2(Gb, rho)
            if len(S) >= d - 1 and is_connected(d, S):
                items.append((len(S), fit_support(Gb, S).F, S))
        if items:
            b2, kb2, Fb2, Eb2 = select_bic_path(items, n)
            comp['l1'] = dict(k=kb2, F=Fb2, edges=list(Eb2), path=[dict(k=k, F=F) for k, F, e in items], time=time.time() - t)
        out['competitors'] = comp
        dump(f'colorado_p{p}.json', out)
    print("colorado done", kb, out['general']['bic_k'], {k: v['k'] for k, v in comp.items()}, flush=True)

if __name__ == '__main__':
    what = sys.argv[1]
    fn = {'exact': study_exact, 'recovery': study_recovery, 'certify': study_certify, 'danube': study_danube, 'oa': study_oa,
          'exact_recert': study_exact_recert, 'exact_signed': study_exact_signed, 'recovery_general': study_recovery_general,
          'certify_new': study_certify_new, 'certify_nocomp': study_certify_nocomp, 'danube_new': study_danube_new, 'colorado': study_colorado}[what]
    if what == 'colorado' and len(sys.argv) > 2:
        fn(p=float(sys.argv[2]))
    elif what == 'exact_signed' and len(sys.argv) > 2:
        fn(which=sys.argv[2])
    else:
        fn()
