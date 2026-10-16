"""
Experiments of Section 6, second protocol: 25 replications per configuration.

Every replication is a unit (experiment, configuration, replication) with its own
seed, np.random.default_rng([experiment code, *configuration, replication]), so a
unit can be rerun alone and units can run in parallel.  Each unit is written to
results/v2/<experiment>/<unit>.json when it finishes.

Usage:  python experiments_v2.py worker <experiment> [<experiment> ...]
        python experiments_v2.py list
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrl0 import *

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'results', 'v2')
REPS = 25
CODE = dict(certify_dense=8, recovery=1, recovery_w2=6, recovery_w4=7, exact=2, exact_signed=3, oa=4, certify=5)
GCODE = dict(sparse=0, dense=1, signed=2)


def _json(o):
    if hasattr(o, 'tolist'):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def rng_for(exp, key, rep):
    ints = [CODE[exp]] + [GCODE[x] if isinstance(x, str) else int(x) for x in key] + [rep]
    return np.random.default_rng(ints)


def edges_list(S):
    return [list(map(int, e)) for e in S]


def select_bic(items, n):
    best = None
    for k, F, edges in items:
        b = bic(F, k, n)
        if best is None or b < best[0]:
            best = (b, k, F, edges)
    return best


# ---------------------------------------------------------------------------
# units
# ---------------------------------------------------------------------------
def units():
    U = []
    for d in [10, 20]:
        for g in ['sparse', 'dense', 'signed']:
            for n in [100, 250, 1000]:
                for rep in range(REPS):
                    U.append(('recovery', (d, g, n), rep))
    for sc in (2, 4):
        for d in [10, 20]:
            for g in ['sparse', 'dense', 'signed']:
                for n in [100, 250, 1000]:
                    for rep in range(REPS):
                        U.append((f'recovery_w{sc}', (d, g, n), rep))
    for d, kt, n in [(6, 8, 50), (6, 8, 200), (6, 10, 50), (6, 10, 200), (7, 9, 50), (7, 9, 200)]:
        for rep in range(REPS):
            U.append(('exact', (d, kt, n), rep))
    for d, kt, n in [(6, 8, 200), (6, 9, 100), (7, 9, 200)]:
        for rep in range(REPS):
            U.append(('exact_signed', (d, kt, n), rep))
    for d, kt, n in [(8, 10, 200), (10, 13, 200)]:
        for rep in range(REPS):
            U.append(('oa', (d, kt, n), rep))
    for d in [10, 15, 20, 31]:
        for n in [200, 1000]:
            for rep in range(REPS):
                U.append(('certify', (d, n), rep))
    for rep in range(REPS):                    # replication-major, so that a partial run is balanced
        for d in [10, 15, 20, 31]:
            for n in [200, 1000]:
                U.append(('certify_dense', (d, n), rep))
    return U


def unit_name(exp, key, rep):
    return exp + '_' + '_'.join(str(x) for x in key) + f'_r{rep:02d}'


def unit_path(exp, key, rep):
    return os.path.join(OUT, exp, unit_name(exp, key, rep) + '.json')


# ---------------------------------------------------------------------------
def run_recovery(key, rep, scale=1):
    """scale > 1 divides the conductances by scale, which multiplies the variogram by
    scale (weaker dependence); the seed is that of the strong design, so the graph and
    the relative conductances are the same at every scale"""
    d, gtype, n = key
    rng = rng_for('recovery', key, rep)
    ktrue = int(np.floor(1.3 * (d - 1))) if gtype != 'dense' else 2 * (d - 1)
    if gtype == 'signed':
        E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
    else:
        E, w, Gam = random_hr_model(d, ktrue, rng)
    Gam = scale * Gam
    Gb = emp_vario(rmpareto_HR(n, Gam, rng))
    res = dict(d=d, gtype=gtype, ktrue=ktrue, n=n, rep=rep, scale=scale, true_edges=edges_list(E))
    kmax = min(int(2.6 * (d - 1)), d * (d - 1) // 2)
    for tag, sg in [('l0', False), ('l0g', True)]:
        t = time.time(); path = forward_path(Gb, kmax, signed=sg); tt = time.time() - t
        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
        b, kb, Fb, Eb = select_bic(items, n)
        res[tag + '_time'] = tt; res[tag + '_k'] = kb; res[tag + '_f1'] = f1_score(E, Eb)[2]
        res[tag + '_edges'] = edges_list(Eb)
        res[tag + '_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
        res[tag + '_path'] = [[k, F] for k, F, _ in items]
    S = mst_edges(Gb)
    res['mst_f1'] = f1_score(E, S)[2]; res['mst_k'] = len(S); res['mst_edges'] = edges_list(S)
    t = time.time(); full = emtp2_mle(Gb); E0 = emtp2_support(full)
    res['emtp2_time'] = time.time() - t; res['emtp2_k'] = len(E0); res['emtp2_f1'] = f1_score(E, E0)[2]
    res['emtp2_edges'] = edges_list(E0)

    def tuned(tag, grid, est, refit):
        t = time.time(); items = []
        for rho in grid:
            try:
                S = est(Gb, rho)
            except Exception:
                continue
            if len(S) < d - 1 or not is_connected(d, S):
                continue
            F = refit(S)
            if np.isfinite(F):
                items.append((len(S), F, S))
        if items:
            b, kb, Fb, Eb = select_bic(items, n)
            res[tag + '_k'] = kb; res[tag + '_f1'] = f1_score(E, Eb)[2]; res[tag + '_edges'] = edges_list(Eb)
            res[tag + '_oracle_f1'] = max(f1_score(E, e)[2] for _, _, e in items)
        res[tag + '_ngrid'] = len(items)
        res[tag + '_time'] = time.time() - t

    tuned('l1', np.geomspace(0.001, 0.5, 10), l1_emtp2, lambda S: fit_support(Gb, S).F)
    tuned('eg', np.geomspace(0.01, 0.7, 10), eglearn, lambda S: fit_general(Gb, S)[0])
    tuned('gl', np.geomspace(0.005, 0.5, 8), extremal_glasso, lambda S: fit_general(Gb, S)[0])
    return res


def run_exact(key, rep, signed=False):
    from enum_batch import enumerate_min
    d, ktrue, n = key
    exp = 'exact_signed' if signed else 'exact'
    rng = rng_for(exp, key, rep)
    if signed:
        E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
    else:
        E, w, Gam = random_hr_model(d, ktrue, rng)
    Gb = emp_vario(rmpareto_HR(n, Gam, rng))
    kmax = min(ktrue + 2, d * (d - 1) // 2)
    t = time.time(); path = forward_path(Gb, kmax, signed=signed); tpath = time.time() - t
    rows = []
    for k in range(d - 1, kmax + 1):
        if k not in path:          # the saturated support has fewer than k edges
            continue
        t = time.time(); en = enumerate_min(Gb, k, signed=signed); tb = time.time() - t
        cert = SwapCertifier(Gb, path[k].edges, signed=signed)
        out = cert.certify(time_limit=900 if signed else 600)
        rows.append(dict(k=k, brute=en['F'], brute_edges=edges_list(en['edges']), n_supports=en['n_supports'],
                         n_fallback=en['n_fallback'], t_brute=tb, heur=path[k].F, cert=out['F'],
                         exact=bool(out['exact']), radius=out['radius'], t_cert=out['time'],
                         heur_optimal=bool(path[k].F <= en['F'] + 1e-7),
                         cert_optimal=bool(out['F'] <= en['F'] + 1e-7),
                         gap_heur=float(path[k].F - en['F']), refits=out['stats']['refits'],
                         f1_brute=f1_score(E, en['edges'])[2]))
        print(exp, key, rep, {a: rows[-1][a] for a in ('k', 'heur_optimal', 'cert_optimal', 'exact', 't_brute', 't_cert')}, flush=True)
    return dict(d=d, ktrue=ktrue, n=n, rep=rep, signed=signed, t_path=tpath, true_edges=edges_list(E), problems=rows)


def run_oa(key, rep):
    import hrsparse as old
    d, ktrue, n = key
    rng = rng_for('oa', key, rep)
    E, w, Gam = random_hr_model(d, ktrue, rng)
    Gb = emp_vario(rmpareto_HR(n, Gam, rng))
    full = emtp2_mle(Gb)
    oa = old.OAPath(Gb, emtp2=True, tol=1e-6)
    t = time.time()
    try:
        res = oa.run(kmax=ktrue, kmin=ktrue, time_limit=OA_LIMIT, max_iter=400); r = res[ktrue]
    except RuntimeError as ex:            # the MILP solver failed on a master problem
        UB = min(f['F'] for key, f in oa.visited.items() if len(key) <= ktrue)
        r = dict(F=UB, LB=float('nan'), iters=len(oa.cuts), time=time.time() - t, ncuts=len(oa.cuts),
                 status='solver error: ' + str(ex))
    toa = time.time() - t
    path = forward_path(Gb, ktrue)
    cert = SwapCertifier(Gb, path[ktrue].edges)
    out = cert.certify(time_limit=OA_CERT_LIMIT)
    return dict(d=d, k=ktrue, n=n, rep=rep, UB=r['F'], LB=r['LB'], F_full=full.F, iters=r['iters'], time=r['time'],
                time_total=toa, ncuts=r['ncuts'], status=r['status'], heur=path[ktrue].F, cert=out['F'],
                cert_exact=bool(out['exact']), cert_radius=out['radius'], cert_time=out['time'],
                cert_refits=out['stats']['refits'], rlog=out['rlog'])


OA_LIMIT, OA_CERT_LIMIT = 120, 120
CERT_DESIGN = {10: (4, 3, 300), 15: (3, 3, 300), 20: (3, 2, 600), 31: (3, 2, 600)}   # r_max, r_max without completion, budget


def run_certify(key, rep, dense=False):
    """dense=True: true graph with 2(d-1) edges, certified with the completion only"""
    d, n = key
    rng = rng_for('certify_dense' if dense else 'certify', key, rep)
    ktrue = 2 * (d - 1) if dense else int(np.floor(1.3 * (d - 1)))
    E, w, Gam = random_hr_model(d, ktrue, rng)
    Gb = emp_vario(rmpareto_HR(n, Gam, rng))
    kmax = min(int(2.6 * (d - 1)), d * (d - 1) // 2)
    t = time.time(); path = forward_path(Gb, kmax); tpath = time.time() - t
    items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
    b, kb, Fb, Eb = select_bic(items, n)
    rmax, rmax_nc, budget = CERT_DESIGN[d]
    res = dict(d=d, n=n, rep=rep, ktrue=ktrue, k=kb, heur=path[kb].F, t_path=tpath, budget=budget,
               true_edges=edges_list(E), f1_heur=f1_score(E, path[kb].edges)[2])
    for tag, comp, rm in [('comp', True, rmax), ('nocomp', False, rmax_nc)][:1 if dense else 2]:
        cert = SwapCertifier(Gb, path[kb].edges, completion=comp)
        out = cert.certify(r_max=rm, time_limit=budget)
        res[tag] = dict(r_max=rm, F=out['F'], radius=out['radius'], timed_out=out['timed_out'], time=out['time'],
                        refits=out['stats']['refits'], rlog=out['rlog'], improved=bool(out['F'] < path[kb].F - 1e-7),
                        f1=f1_score(E, out['edges'])[2], edges=edges_list(out['edges']))
        print('certify', key, rep, tag, out['radius'], round(out['time'], 1), out['timed_out'], flush=True)
    return res


RUN = dict(recovery=run_recovery, recovery_w2=lambda key, rep: run_recovery(key, rep, 2),
           recovery_w4=lambda key, rep: run_recovery(key, rep, 4), exact=run_exact, exact_signed=lambda key, rep: run_exact(key, rep, signed=True),
           oa=run_oa, certify=run_certify, certify_dense=lambda key, rep: run_certify(key, rep, dense=True))


# ---------------------------------------------------------------------------
def claim(path):
    if os.path.exists(path):
        return False
    lock = path + '.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode()); os.close(fd)
        return True
    except FileExistsError:
        try:
            pid = int(open(lock).read().strip() or 0)
        except Exception:
            pid = 0
        if pid and not os.path.exists(f'/proc/{pid}'):
            os.remove(lock)
            return claim(path)
        return False


def worker(exps):
    order = [u for e in exps for u in units() if u[0] == e]
    deadline = float(os.environ.get('V2_DEADLINE', 'inf'))
    for exp, key, rep in order:
        if time.time() > deadline:
            print('DEADLINE reached, no new units', flush=True); break
        p = unit_path(exp, key, rep)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        if not claim(p):
            continue
        t = time.time()
        try:
            res = RUN[exp](key, rep)
            res['unit_time'] = time.time() - t
            with open(p + '.tmp', 'w') as f:
                json.dump(res, f, default=_json)
            os.replace(p + '.tmp', p)
            print('DONE', unit_name(exp, key, rep), round(time.time() - t, 1), flush=True)
        except Exception as ex:
            import traceback; traceback.print_exc()
            print('FAIL', unit_name(exp, key, rep), repr(ex), flush=True)
        finally:
            try:
                os.remove(p + '.lock')
            except FileNotFoundError:
                pass


if __name__ == '__main__':
    if sys.argv[1] == 'worker':
        worker(sys.argv[2:])
    elif sys.argv[1] == 'list':
        from collections import Counter
        c = Counter(); dn = Counter()
        for exp, key, rep in units():
            c[exp] += 1; dn[exp] += os.path.exists(unit_path(exp, key, rep))
        for e in c:
            print(e, dn[e], '/', c[e])
