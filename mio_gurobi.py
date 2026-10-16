"""
Branch-and-cut for problem (8) with Gurobi (size-limited license), single tree with lazy constraints,
on the data sets of the outer-approximation experiment of Section 6.2.

  gamma = 0 : big-M formulation (11), cuts  theta >= f(S) - sum_e (R_e/Gam_e - 1)_+ (z_e - z_e(S))
  gamma > 0 : perspective formulation of the ridge-regularized problem
              min f_gamma(S) = min_{w >= 0, supp(w) in S} F(w) + gamma/2 sum_e w_e^2,
              cuts  theta >= f_gamma(S) - sum_e alpha_e^2 / (2 gamma) (z_e - z_e(S)),
              alpha_e = gamma w_e on S and (R_e - Gam_e)_+ off S  (Bertsimas et al. 2020)
Connectivity: degree constraints and lazy cut-set constraints.  Incumbents of the path give initial cuts.
Usage: python mio_gurobi.py <gamma> <d> [reps]
"""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from hrl0 import *
from experiments_v2 import rng_for

OUTD = os.path.join(HERE, 'results', 'v2', 'mio_gurobi'); os.makedirs(OUTD, exist_ok=True)
TIME_LIMIT = 120.0

def fit_ridge(Gam, edges, gamma, tol=1e-11, maxit=200):
    """projected Newton for F(w) + gamma/2 |w|^2 on a connected support, w >= 0"""
    d = Gam.shape[0]; edges = [tuple(e) for e in edges]; k = len(edges)
    g_e = edge_vec(Gam, edges); B = incidence(d, edges); J = np.ones((d, d)) / d
    def val(w):
        try:
            C = np.linalg.cholesky((B * w) @ B.T + J)
        except np.linalg.LinAlgError:
            return np.inf
        return -2.0 * np.log(np.diag(C)).sum() + float(g_e @ w) + 0.5 * gamma * float(w @ w)
    w = 1.0 / g_e; F = val(w)
    for it in range(maxit):
        G = np.linalg.inv((B * w) @ B.T + J); Q = B.T @ G @ B; R = np.diag(Q)
        grad = g_e - R + gamma * w
        act = w > 1e-13
        kkt = max(np.abs(grad[act]).max() if act.any() else 0.0, (-grad[~act]).max() if (~act).any() else 0.0)
        if kkt < tol * (1 + np.abs(g_e).max()):
            break
        free = act | (grad < 0)
        H = Q * Q + gamma * np.eye(k)
        p = np.zeros(k); p[free] = -np.linalg.solve(H[np.ix_(free, free)], grad[free])
        t = 1.0; ok = False
        for _ in range(60):
            wn = np.maximum(w + t * p, 0.0); Fn = val(wn)
            if np.isfinite(Fn) and Fn <= F + 1e-4 * float(grad @ (wn - w)) and Fn <= F:
                ok = True; break
            t *= 0.5
        if not ok:
            break
        w, F = wn, Fn
    Rfull, _ = resistance_from_L((B * w) @ B.T)
    return dict(edges=edges, w=w, F=F, R=Rfull)

def run(Gb, k, gamma, init_supports, time_limit=TIME_LIMIT, verbose=False):
    d = Gb.shape[0]; E = edges_of(d); m = len(E); idx = {e: t for t, e in enumerate(E)}
    Gv = edge_vec(Gb, E)
    cache = {}
    def f_of(S):
        key = frozenset(S)
        if key not in cache:
            cache[key] = fit_ridge(Gb, sorted(S), gamma) if gamma > 0 else fit_support(Gb, sorted(S))
            if gamma == 0:
                f = cache[key]; cache[key] = dict(edges=f.edges, w=f.w, F=f.F, R=f.R)
        return cache[key]
    def cut(S):
        f = f_of(S); zS = np.zeros(m); zS[[idx[e] for e in S]] = 1
        Rv = np.array([f['R'][e] for e in E])
        if gamma > 0:
            wfull = np.zeros(m); wfull[[idx[e] for e in f['edges']]] = f['w']
            alpha = np.where(zS > 0.5, gamma * wfull, np.maximum(Rv - Gv, 0.0))
            slope = -alpha ** 2 / (2 * gamma)
        else:
            slope = np.where(zS > 0.5, 0.0, -np.maximum(Rv / Gv - 1.0, 0.0))
        return f['F'], slope, zS
    full = fit_ridge(Gb, E, gamma) if gamma > 0 else fit_support(Gb, E)
    f_full = full['F'] if gamma > 0 else full.F
    mdl = gp.Model(); mdl.Params.OutputFlag = 1 if verbose else 0
    mdl.Params.Threads = 1; mdl.Params.TimeLimit = time_limit; mdl.Params.LazyConstraints = 1
    mdl.Params.MIPGap = 1e-9; mdl.Params.MIPGapAbs = 1e-7
    z = mdl.addVars(m, vtype=GRB.BINARY); th = mdl.addVar(lb=f_full - 1e-9, ub=GRB.INFINITY)
    mdl.setObjective(th, GRB.MINIMIZE)
    mdl.addConstr(gp.quicksum(z[t] for t in range(m)) <= k)
    mdl.addConstr(gp.quicksum(z[t] for t in range(m)) >= d - 1)
    for i in range(d):
        mdl.addConstr(gp.quicksum(z[t] for t, (a, b) in enumerate(E) if i in (a, b)) >= 1)
    ncuts = [0, 0]
    best = [np.inf, None]
    for S in init_supports:
        F, sl, zS = cut(S)
        mdl.addConstr(th >= F + gp.quicksum(sl[t] * (z[t] - zS[t]) for t in range(m) if sl[t] != 0)); ncuts[0] += 1
        if len(S) <= k and F < best[0]:
            best = [F, S]
    if best[1] is not None:
        zS = np.zeros(m); zS[[idx[e] for e in best[1]]] = 1
        for t in range(m): z[t].Start = zS[t]
        th.Start = best[0]
    def cb(model, where):
        if where != GRB.Callback.MIPSOL:
            return
        zv = model.cbGetSolution([z[t] for t in range(m)]); tv = model.cbGetSolution(th)
        S = [E[t] for t in range(m) if zv[t] > 0.5]
        comps = components_of(d, S)
        if len(set(comps)) > 1:
            for c in set(comps):
                U = {i for i in range(d) if comps[i] == c}
                model.cbLazy(gp.quicksum(z[t] for t, (a, b) in enumerate(E) if (a in U) != (b in U)) >= 1); ncuts[1] += 1
            return
        F, sl, zS = cut(S)
        if tv < F - 1e-7 * (1 + abs(F)):
            model.cbLazy(th >= F + gp.quicksum(sl[t] * (z[t] - zS[t]) for t in range(m) if sl[t] != 0)); ncuts[0] += 1
    t0 = time.time(); mdl.optimize(cb); el = time.time() - t0
    return dict(status=int(mdl.Status), optimal=mdl.Status == GRB.OPTIMAL, UB=float(mdl.ObjVal) if mdl.SolCount else None,
                LB=float(mdl.ObjBound), f_full=float(f_full), nodes=float(mdl.NodeCount), oa_cuts=ncuts[0], cutsets=ncuts[1],
                fits=len(cache), time=el)

def unit(gamma, d, rep):
    ktrue = {8: 10, 10: 13}[d]; n = 200
    fn = os.path.join(OUTD, f'g{gamma}_{d}_r{rep:02d}.json')
    if os.path.exists(fn):
        return json.load(open(fn))
    rng = rng_for('oa', (d, ktrue, n), rep)
    E, w, Gam = random_hr_model(d, ktrue, rng)
    Gb = emp_vario(rmpareto_HR(n, Gam, rng))
    path = forward_path(Gb, ktrue)
    init = [list(path[k].edges) for k in sorted(path)]
    res = run(Gb, ktrue, gamma, init)
    res.update(d=d, k=ktrue, n=n, rep=rep, gamma=gamma, heur=float(path[ktrue].F))
    json.dump(res, open(fn, 'w'))
    print(fn, {a: res[a] for a in ('optimal', 'UB', 'LB', 'f_full', 'nodes', 'oa_cuts', 'cutsets', 'time')}, flush=True)
    return res

if __name__ == '__main__':
    gamma = float(sys.argv[1]); d = int(sys.argv[2])
    reps = range(int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else range(25)
    for rep in reps:
        unit(gamma, d, rep)
