"""
Exhaustive enumeration of the connected supports with k edges, fitted in batches.

Every connected support is fitted by a vectorized projected Newton method (the
unconstrained Newton method in the general model).  A fit is accepted when the KKT
residual is below tol(1 + max Gamma); supports that do not reach it are refitted one
at a time with hrl0.fit_support, and the best supports are refitted with
hrl0.fit_support as well, so that the reported minimum is computed by the same
solver as the path.  This is an implementation of the enumeration independent of
the certified search.
"""
import os, itertools
import numpy as np
from hrl0 import edges_of, is_connected, fit_support

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'v2', 'cache')
_mem = {}


def connected_supports(d, k):
    if (d, k) in _mem:
        return _mem[(d, k)]
    os.makedirs(CACHE, exist_ok=True)
    fn = os.path.join(CACHE, f'supports_d{d}_k{k}.npy')
    if os.path.exists(fn):
        arr = np.load(fn)
    else:
        E = edges_of(d)
        arr = np.array([c for c in itertools.combinations(range(len(E)), k)
                        if is_connected(d, [E[t] for t in c])], dtype=np.int16).reshape(-1, k)
        np.save(fn + '.tmp.npy', arr); os.replace(fn + '.tmp.npy', fn)
    _mem[(d, k)] = arr
    return arr


def _values(Inc, g, w, Jm, signed):
    L = np.einsum('bik,bk,bjk->bij', Inc, w, Inc) + Jm
    if signed:
        ev = np.linalg.eigvalsh(L)
        ok = ev[:, 0] > 1e-14
        ld = np.log(np.where(ok[:, None], ev, 1.0)).sum(1)
    else:
        sign, ld = np.linalg.slogdet(L)
        ok = (sign > 0) & np.isfinite(ld)
    F = -ld + (g * w).sum(1)
    return np.where(ok, F, np.inf)


def batch_fit(Gam, supp, signed=False, tol=1e-10, maxit=100):
    """values of the supports (rows of supp, indices into edges_of(d)); returns
    (F, converged)"""
    d = Gam.shape[0]
    E = edges_of(d)
    ei = np.array([e[0] for e in E]); ej = np.array([e[1] for e in E])
    gall = np.array([Gam[e] for e in E])
    B, k = supp.shape
    I = ei[supp]; J = ej[supp]; g = gall[supp]
    Inc = np.zeros((B, d, k))
    ar = np.arange(B)[:, None]; kk = np.arange(k)[None, :]
    Inc[ar, I, kk] = 1.0; Inc[ar, J, kk] = -1.0
    Jm = np.full((d, d), 1.0 / d)
    wmax = 1.0 / g
    w = 1.0 / g
    F = _values(Inc, g, w, Jm, signed)
    conv = np.zeros(B, bool); fail = ~np.isfinite(F)
    act_idx = np.where(~fail)[0]
    eye = np.eye(k)
    for it in range(maxit):
        if act_idx.size == 0:
            break
        Ia = Inc[act_idx]; wa = w[act_idx]; ga = g[act_idx]; Fa = F[act_idx]
        L = np.einsum('bik,bk,bjk->bij', Ia, wa, Ia) + Jm
        G = np.linalg.inv(L)
        Q = np.matmul(Ia.transpose(0, 2, 1), np.matmul(G, Ia))
        R = np.diagonal(Q, axis1=1, axis2=2)
        grad = ga - R
        act = np.ones_like(wa, bool) if signed else (wa > 1e-13)
        kkt = np.where(act, np.abs(grad), np.maximum(-grad, 0.0)).max(1)
        scale = 1.0 + np.abs(ga).max(1)
        done = kkt < tol * scale
        conv[act_idx[done]] = True
        keep = ~done
        if not keep.any():
            act_idx = act_idx[:0]; break
        act_idx = act_idx[keep]; Ia = Ia[keep]; wa = wa[keep]; ga = ga[keep]; Fa = Fa[keep]
        grad = grad[keep]; Q = Q[keep]; act = act[keep]; kkt = kkt[keep]; scale = scale[keep]
        free = np.ones_like(act) if signed else (act | (grad < 0))
        H = Q * Q
        ff = free[:, :, None] & free[:, None, :]
        Hm = np.where(ff, H, eye) + 1e-12 * eye
        rhs = np.where(free, grad, 0.0)
        p = -np.linalg.solve(Hm, rhs[..., None])[..., 0]
        p = np.where(free, p, 0.0)
        nd = (rhs * p).sum(1) >= 0
        p[nd] = -rhs[nd]
        # backtracking line search, vectorized over the instances
        nb = act_idx.size
        t = np.ones(nb); acc = np.zeros(nb, bool)
        wnew = wa.copy(); Fnew = Fa.copy()
        pend = np.arange(nb)
        wm = wmax[act_idx]
        for _ in range(60):
            if pend.size == 0:
                break
            wn = wa[pend] + t[pend, None] * p[pend]
            if not signed:
                wn = np.minimum(np.maximum(wn, 0.0), wm[pend])
            Fn = _values(Ia[pend], ga[pend], wn, Jm, signed)
            ok = np.isfinite(Fn) & (Fn <= Fa[pend] + 1e-4 * (grad[pend] * (wn - wa[pend])).sum(1)) & (Fn < Fa[pend])
            sel = pend[ok]
            wnew[sel] = wn[ok]; Fnew[sel] = Fn[ok]; acc[sel] = True
            pend = pend[~ok]; t[pend] *= 0.5
        # no decrease possible: converged to working precision, or handed to the fallback
        stuck = ~acc
        if stuck.any():
            small = stuck & (kkt < 1e-7 * scale)
            conv[act_idx[small]] = True
            fail[act_idx[stuck & ~small]] = True
        w[act_idx] = wnew; F[act_idx] = Fnew
        act_idx = act_idx[acc]
    fail[act_idx] = True                                  # not converged within maxit
    return F, conv & ~fail


def enumerate_min(Gam, k, signed=False, chunk=20000, top=10):
    d = Gam.shape[0]
    E = edges_of(d)
    supp = connected_supports(d, k)
    vals = np.empty(len(supp)); ok = np.empty(len(supp), bool)
    for s in range(0, len(supp), chunk):
        v, c = batch_fit(Gam, supp[s:s + chunk].astype(np.int64), signed=signed)
        vals[s:s + chunk] = v; ok[s:s + chunk] = c
    fb = np.where(~ok)[0]
    for t in fb:                                          # fallback: one at a time
        fit = fit_support(Gam, [E[e] for e in supp[t]], signed=signed)
        vals[t] = np.inf if fit is None else fit.F
    best = np.argsort(vals)[:top]
    refit = []
    for t in best:                                        # the same solver as the path
        fit = fit_support(Gam, [E[e] for e in supp[t]], signed=signed)
        refit.append(np.inf if fit is None else fit.F)
    refit = np.array(refit)
    j = int(np.argmin(refit))
    return dict(F=float(refit[j]), edges=[E[e] for e in supp[best[j]]], n_supports=int(len(supp)),
                n_fallback=int(len(fb)), max_batch_vs_refit=float(np.max(np.abs(refit - vals[best]))))
