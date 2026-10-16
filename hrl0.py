"""
hrl0: cardinality-constrained maximum likelihood for Huesler-Reiss graphical
models under extremal total positivity (Laplacian parametrisation).

Contents
  graph and model utilities, simulation, empirical variogram
  fit_support      : profile likelihood on a fixed support (projected Newton)
  forward_path     : greedy forward search from the minimum spanning tree with
                     swap local search (the whole cardinality path)
  SwapCertifier    : exact search over the r-swap neighbourhood of an incumbent
                     (global optimality when r reaches the cardinality)
  competitors      : EMTP2 MLE, l1-penalised EMTP2 (extremal graphical lasso
                     under total positivity), EGlearn, general extremal glasso
  bic              : surrogate BIC
"""
import numpy as np, networkx as nx, time, itertools
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.linalg import cho_factor, cho_solve

# ---------------------------------------------------------------------------
# graphs
# ---------------------------------------------------------------------------
def edges_of(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]

def incidence(d, edges):
    B = np.zeros((d, len(edges)))
    for t, (i, j) in enumerate(edges):
        B[i, t] = 1.0; B[j, t] = -1.0
    return B

def laplacian(d, edges, w):
    L = np.zeros((d, d))
    for (i, j), we in zip(edges, w):
        L[i, i] += we; L[j, j] += we; L[i, j] -= we; L[j, i] -= we
    return L

def resistance_from_L(L):
    d = L.shape[0]
    G = np.linalg.inv(L + np.ones((d, d)) / d)
    g = np.diag(G)
    return g[:, None] + g[None, :] - 2.0 * G, G

def components_of(d, edges):
    A = lil_matrix((d, d))
    for i, j in edges:
        A[i, j] = 1; A[j, i] = 1
    return connected_components(csr_matrix(A), directed=False)[1]

def is_connected(d, edges):
    if len(edges) < d - 1:
        return False
    return components_of(d, edges).max() == 0

def edge_vec(Gam, edges):
    return np.array([Gam[i, j] for i, j in edges])

# ---------------------------------------------------------------------------
# Huesler-Reiss model
# ---------------------------------------------------------------------------
def sigma_k(Gam, k):
    d = Gam.shape[0]; idx = [i for i in range(d) if i != k]
    S = 0.5 * (Gam[np.ix_(idx, [k])] + Gam[np.ix_([k], idx)] - Gam[np.ix_(idx, idx)])
    return 0.5 * (S + S.T)

def theta_from_gamma(Gam):
    d = Gam.shape[0]; P = np.eye(d) - np.ones((d, d)) / d
    return np.linalg.pinv(-0.5 * P @ Gam @ P)

def gamma_from_theta(Theta):
    return resistance_from_L(Theta)[0]

def rmpareto_HR(n, Gam, rng):
    """Exact rejection sampler for the multivariate Pareto HR distribution
    (standard Pareto margins, support {max y > 1})."""
    d = Gam.shape[0]
    chols = [np.linalg.cholesky(sigma_k(Gam, k) + 1e-12 * np.eye(d - 1)) for k in range(d)]
    out = []
    batch = max(4 * n, 1000)
    while len(out) < n:
        ks = rng.integers(0, d, batch)
        U = rng.uniform(size=batch)
        Z = rng.standard_normal((batch, d - 1))
        for k in range(d):
            sel = np.where(ks == k)[0]
            if len(sel) == 0:
                continue
            idx = [i for i in range(d) if i != k]
            logratio = Z[sel] @ chols[k].T - Gam[idx, k] / 2.0
            yk = 1.0 / U[sel]
            Y = np.empty((len(sel), d)); Y[:, k] = yk
            Y[:, idx] = yk[:, None] * np.exp(logratio)
            N = (Y > 1).sum(1)
            acc = rng.uniform(size=len(sel)) < 1.0 / N
            out.extend(Y[acc])
    return np.array(out[:n])

def data2mpareto(X, p):
    """Empirical rank transform to standard Pareto margins and thresholding at
    the p-quantile of the max; returns the rescaled exceedances."""
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    F = ranks / (n + 1.0)
    Y = 1.0 / (1.0 - F)
    u = 1.0 / (1.0 - p)
    keep = Y.max(1) > u
    return Y[keep] / u

def emp_vario(Y, thr=1.0):
    """Averaged empirical variogram (Engelke and Hitz 2020)."""
    n, d = Y.shape
    LY = np.log(Y)
    G = np.zeros((d, d)); cnt = 0
    for k in range(d):
        rows = LY[Y[:, k] > thr]
        if len(rows) < 2:
            continue
        diff = rows[:, :, None] - rows[:, None, :]
        G += diff.var(axis=0, ddof=1); cnt += 1
    return G / cnt

# ---------------------------------------------------------------------------
# profile objective on a fixed support (EMTP2): projected Newton
#   F(w) = -log det(L(w)+J) + sum_e Gam_e w_e,  w >= 0
# ---------------------------------------------------------------------------
class Fit:
    __slots__ = ('edges', 'w', 'F', 'R', 'Theta', 'G', 'iters', 'conv')
    def __init__(self, edges, w, F, R, Theta, G, iters, conv):
        self.edges = edges; self.w = w; self.F = F; self.R = R; self.Theta = Theta
        self.G = G; self.iters = iters; self.conv = conv

def fit_support(Gam, edges, w0=None, tol=1e-11, maxit=300, signed=False):
    d = Gam.shape[0]
    edges = [tuple(e) for e in edges]
    k = len(edges)
    if k < d - 1 or not is_connected(d, edges):
        return None
    g_e = edge_vec(Gam, edges)
    B = incidence(d, edges)
    J = np.ones((d, d)) / d
    if w0 is None:
        if k <= 2 * d:
            w = 1.0 / g_e
        else:
            # spanning-tree warm start: exact on the minimum spanning tree, small elsewhere
            Ts = set(mst_edges(Gam))
            w = np.array([1.0 / Gam[e] if e in Ts else 1e-3 / Gam[e] for e in edges])
    else:
        w = np.array(w0, dtype=float)

    def evalF(w):
        L = (B * w) @ B.T
        try:
            C = np.linalg.cholesky(L + J)
        except np.linalg.LinAlgError:
            return np.inf
        return -2.0 * np.log(np.diag(C)).sum() + float(g_e @ w)

    wmax = 1.0 / g_e                      # valid bound on the optimal conductances
    w = np.minimum(w, wmax)
    F = evalF(w)
    if not np.isfinite(F):
        w = 1.0 / g_e; F = evalF(w)
    conv = False; it = 0; fails = 0
    for it in range(1, maxit + 1):
        L = (B * w) @ B.T
        G = np.linalg.inv(L + J)
        Q = B.T @ G @ B
        R = np.diag(Q).copy()
        grad = g_e - R
        # stationarity check (KKT): grad == 0 on active, grad >= 0 on zeros
        act = np.ones(k, bool) if signed else (w > 1e-13)
        kkt = max(np.abs(grad[act]).max() if act.any() else 0.0,
                  (-grad[~act]).max() if (~act).any() else 0.0)
        if kkt < tol * (1.0 + np.abs(g_e).max()):
            conv = True; break
        # projected Newton on the free coordinates
        free = np.ones(k, bool) if signed else (act | (grad < 0))
        H = Q * Q
        p = np.zeros(k)
        try:
            p[free] = -np.linalg.solve(H[np.ix_(free, free)] + 1e-12 * np.eye(int(free.sum())), grad[free])
        except np.linalg.LinAlgError:
            p[free] = -grad[free]
        if float(grad[free] @ p[free]) >= 0:          # not a descent direction
            p = np.zeros(k); p[free] = -grad[free]
        improved = False
        for direction in (p, -np.where(free, grad, 0.0)):
            t = 1.0
            for _ in range(60):
                wn = (w + t * direction) if signed else np.minimum(np.maximum(w + t * direction, 0.0), wmax)
                Fn = evalF(wn)
                if np.isfinite(Fn) and Fn <= F + 1e-4 * float(grad @ (wn - w)) and Fn < F:
                    improved = True; break
                t *= 0.5
            if improved:
                break
        if not improved:
            fails += 1
            if kkt < 1e-7 * (1.0 + np.abs(g_e).max()):
                conv = True
            if fails >= 2 or conv:
                break
            continue
        w, F = wn, Fn
    L = (B * w) @ B.T
    Rfull, G = resistance_from_L(L)
    return Fit(edges, w, F, Rfull, L, G, it, conv)

def profile_value(Gam, edges, signed=False):
    fit = fit_support(Gam, edges, signed=signed)
    return np.inf if fit is None else fit.F

# ---------------------------------------------------------------------------
# closed forms: trees, block joins
# ---------------------------------------------------------------------------
def mst_edges(Gam):
    d = Gam.shape[0]
    T = minimum_spanning_tree(csr_matrix(np.triu(Gam, 1))).tocoo()
    return sorted((int(min(i, j)), int(max(i, j))) for i, j in zip(T.row, T.col))

def tree_value(Gam, edges):
    """profile value of a spanning tree: log d - (d-1) - sum_e log Gam_e ... in
    F-units, F = -log det(L+J) + <Gam,w> = -log d + sum_e log Gam_e + (d-1)"""
    d = Gam.shape[0]
    return -np.log(d) + sum(np.log(Gam[i, j]) for i, j in edges) + (d - 1)

def divergence(rho):
    return np.log(rho) - 1.0 + 1.0 / rho

# ---------------------------------------------------------------------------
# heuristics: forward greedy from the MST, swap local search
# ---------------------------------------------------------------------------
def _screen_additions(Gam, fit, absent, ncand, signed=False):
    rho = np.array([fit.R[e] / Gam[e] for e in absent])
    score = divergence(rho) if signed else np.where(rho > 1.0, divergence(np.maximum(rho, 1.0)), 0.0)
    order = np.argsort(-score)[:ncand]
    return [absent[t] for t in order if score[t] > 0]

def local_search(Gam, fit, n_out=6, n_in=6, max_rounds=30, tol=1e-9, allE=None, signed=False):
    """1-swap local search at fixed cardinality (screened candidates)."""
    d = Gam.shape[0]
    allE = edges_of(d) if allE is None else allE
    E = list(fit.edges); F = fit.F
    for _ in range(max_rounds):
        improved = False
        Es = set(E)
        absent = [e for e in allE if e not in Es]
        cand_in = _screen_additions(Gam, fit, absent, n_in, signed=signed)
        y = np.array([Gam[e] * we for e, we in zip(E, fit.w)])   # spanning-tree marginals (EMTP2); Gam_e w_e in general
        with np.errstate(divide='ignore', invalid='ignore'):
            loss_out = np.where(y < 1.0, -np.log1p(-np.minimum(y, 1.0 - 1e-12)) - y, np.inf)   # fixed-rest removal loss
        cand_out = [E[t] for t in np.argsort(loss_out)[:n_out]]
        best = None
        wmap = dict(zip(E, fit.w))
        for eo in cand_out:
            for ei in cand_in:
                En = sorted([e for e in E if e != eo] + [ei])
                if signed:
                    w0 = np.array([wmap[x] if x in wmap else 1.0 / Gam[x] - 1.0 / fit.R[x] for x in En])
                else:
                    w0 = np.array([wmap[x] if x in wmap else max(1.0 / Gam[x] - 1.0 / fit.R[x], 1e-8) for x in En])
                fn = fit_support(Gam, En, w0=w0, signed=signed)
                if fn is not None and fn.F < F - tol and (best is None or fn.F < best.F):
                    best = fn
        if best is not None:
            E, F, fit = list(best.edges), best.F, best; improved = True
        if not improved:
            break
    return fit

def forward_path(Gam, kmax, ncand=8, do_local=True, verbose=False, signed=False, two_in=True, n_out=6, checkpoint=None):
    """Greedy forward selection from the MST; returns {k: Fit}.  With a checkpoint
    file the supports of the path are saved after every step and the path resumes
    from the last saved support."""
    import json, os
    d = Gam.shape[0]
    allE = edges_of(d)
    E = mst_edges(Gam)
    fit = fit_support(Gam, E, signed=signed)
    path = {len(E): fit}
    if checkpoint is not None and os.path.exists(checkpoint):
        saved = json.load(open(checkpoint))
        for kk in sorted(int(k) for k in saved):
            Ek = [tuple(e) for e in saved[str(kk)]]
            path[kk] = fit_support(Gam, Ek, signed=signed)
        kmax_saved = max(path)
        E = list(path[kmax_saved].edges); fit = path[kmax_saved]
    def save():
        if checkpoint is not None:
            tmp = checkpoint + '.tmp'
            json.dump({str(k): [list(map(int, e)) for e in f.edges] for k, f in path.items()}, open(tmp, 'w'))
            os.replace(tmp, checkpoint)
    while len(E) < kmax:
        Es = set(E)
        absent = [e for e in allE if e not in Es]
        cands = _screen_additions(Gam, fit, absent, ncand, signed=signed)
        if not cands:
            break
        best = None
        wmap = dict(zip(fit.edges, fit.w))
        def start(En):
            if signed:
                return np.array([wmap[x] if x in wmap else 1.0 / Gam[x] - 1.0 / fit.R[x] for x in En])
            return np.array([wmap[x] if x in wmap else max(1.0 / Gam[x] - 1.0 / fit.R[x], 1e-8) for x in En])
        for e in cands:
            En = sorted(E + [e])
            fn = fit_support(Gam, En, w0=start(En), signed=signed)
            if fn is not None and (best is None or fn.F < best.F):
                best = fn
        if best is None:
            break
        if two_in and len(cands) >= 2:
            # moves that add two screened edges and remove one: escape the greedy trap of edges that only pay off jointly
            y = np.array([Gam[e] * we for e, we in zip(E, fit.w)])
            with np.errstate(divide='ignore', invalid='ignore'):
                loss_out = np.where(y < 1.0, -np.log1p(-np.minimum(y, 1.0 - 1e-12)) - y, np.inf)
            outs = [E[t] for t in np.argsort(loss_out)[:n_out]]
            for e1, e2 in itertools.combinations(cands, 2):
                for a in outs:
                    En = sorted([x for x in E if x != a] + [e1, e2])
                    if not is_connected(d, En):
                        continue
                    fn = fit_support(Gam, En, w0=start(En), signed=signed)
                    if fn is not None and fn.F < best.F - 1e-12:
                        best = fn
        if do_local:
            best = local_search(Gam, best, allE=allE, signed=signed)
        E = list(best.edges); fit = best
        path[len(E)] = fit
        save()
        if verbose:
            print(f"  k={len(E)} F={fit.F:.6f}", flush=True)
    return path

# ---------------------------------------------------------------------------
# exact search over the swap neighbourhood of an incumbent
# ---------------------------------------------------------------------------
def psi(sigma):
    """upper bound on the pinning loss of a set whose s-values sum to sigma"""
    s2 = np.sqrt(2.0)
    if sigma >= s2:
        return np.inf
    return sigma ** 2 / (2.0 * (1.0 - sigma / s2))

def omega(t):
    return t - np.log1p(t)

def sc_gain_single(a, h, M):
    """max_{0<=v<=M} a v - omega(sqrt(h) v)   (closed form)"""
    if a <= 0:
        return 0.0
    rh = np.sqrt(max(h, 0.0))
    if rh <= 0 or a >= rh:
        return a * M - omega(rh * M)
    v = min(a / (rh * (rh - a)), M)
    return a * v - omega(rh * v)

def sc_gain_single_vec(a, h, M):
    """vectorized closed form of sc_gain_single"""
    a = np.asarray(a, float); h = np.maximum(np.asarray(h, float), 0.0); M = np.asarray(M, float)
    rh = np.sqrt(h)
    out = np.zeros(len(a))
    pos = a > 0
    cap = pos & ((rh <= 0) | (a >= rh))
    out[cap] = a[cap] * M[cap] - omega(rh[cap] * M[cap])
    mid = pos & ~cap
    v = np.minimum(a[mid] / (rh[mid] * (rh[mid] - a[mid])), M[mid])
    out[mid] = a[mid] * v - omega(rh[mid] * v)
    return out

def sc_gain(a, Hs, M, iters=100, isotropic=False):
    """max_{0<=v<=M} a'v - omega(sqrt(v'Hs v))  (concave; projected gradient).
    With isotropic=True the quadratic form is replaced by its smallest eigenvalue
    times the identity, which gives a valid bound for the signed model where the
    maximization runs over |v| <= M."""
    r = len(a)
    if r == 1:
        return sc_gain_single(a[0], Hs[0, 0], M[0])
    if isotropic:
        Hs = max(float(np.linalg.eigvalsh(0.5 * (Hs + Hs.T))[0]), 0.0) * np.eye(r)
    def val(v):
        q = float(v @ Hs @ v)
        return float(a @ v) - omega(np.sqrt(max(q, 0.0)))
    v = np.zeros(r)
    for t in range(r):
        if a[t] > 0:
            rh = np.sqrt(max(Hs[t, t], 0.0))
            v[t] = M[t] if (rh <= 0 or a[t] >= rh) else min(a[t] / (rh * (rh - a[t])), M[t])
    v = np.clip(v, 0.0, M)
    f = val(v); step = 1.0
    for _ in range(iters):
        q = float(v @ Hs @ v); nq = np.sqrt(max(q, 1e-300))
        grad = a - (nq / (1.0 + nq)) * (Hs @ v) / nq
        improved = False
        for _ in range(40):
            vn = np.clip(v + step * grad, 0.0, M)
            fn = val(vn)
            if fn > f + 1e-14:
                v, f = vn, fn; improved = True; step *= 1.3; break
            step *= 0.5
        if not improved:
            break
    return f

class BaseInfo:
    """quantities of a connected base support needed by the bounds"""
    def __init__(self, Gam, edges, w, F, L, G, signed=False):
        self.edges = list(edges); self.w = np.asarray(w); self.F = F
        self.L = L; self.G = G; self.signed = signed
        g = np.diag(G)
        self.R = g[:, None] + g[None, :] - 2.0 * G
        self.Theta = L
        d = Gam.shape[0]; self.d = d
        B = incidence(d, self.edges)
        Q_II = B.T @ G @ B
        H_II = Q_II * Q_II
        try:
            self.H_II_inv = np.linalg.inv(H_II + 1e-12 * np.eye(len(self.edges)))
        except np.linalg.LinAlgError:
            self.H_II_inv = np.linalg.pinv(H_II)
        self.B = B
        self._S = None; self._Kinv = None; self._Npos = None; self._IN = None; self._JN = None
        self._Tbox = None
        self.logdetG = float(2.0 * np.log(np.diag(np.linalg.cholesky(G))).sum())

    def schur(self, Bset):
        """Schur complement of the Hessian on the candidate set Bset"""
        r = len(Bset)
        QBI = np.zeros((r, len(self.edges))); QBB = np.zeros((r, r))
        rows = []
        for t, e in enumerate(Bset):
            i, j = e
            rows.append(self.G[i, :] - self.G[j, :])
            QBI[t] = rows[t] @ self.B
        for t in range(r):
            for u in range(r):
                i, j = Bset[u]
                QBB[t, u] = rows[t][i] - rows[t][j]
        HBB = QBB * QBB; HBI = QBI * QBI
        return HBB - HBI @ self.H_II_inv @ HBI.T

    def sqrt_theta(self):
        if self._S is None:
            lam, V = np.linalg.eigh(self.Theta)
            self._S = (V * np.sqrt(np.maximum(lam, 0.0))) @ V.T
        return self._S

    def completion_setup(self, allE):
        """non-edges N of the base, and the pieces of the Schur complement of the Hessian
        with respect to the base edges that cost O(|N| d |I|)"""
        if self._Npos is None:
            Iset = set(self.edges)
            N = [e for e in allE if e not in Iset]
            self._Npos = {e: t for t, e in enumerate(N)}
            IN = np.fromiter((e[0] for e in N), int, len(N)); JN = np.fromiter((e[1] for e in N), int, len(N))
            self._IN = IN; self._JN = JN
            self._rowsN = self.G[IN, :] - self.G[JN, :]           # b_e^T G for e in N
            Q_NI = self._rowsN @ self.B
            self._H_NI = Q_NI * Q_NI
            self._HNI_HIIinv = self._H_NI @ self.H_II_inv
        return self._Npos, self._IN, self._JN

    def kinv_columns(self, pos):
        """columns of (K_NN)^{-1} = (1/4) Ht_NN for the non-edges at positions pos, O(|N| |I|) each"""
        rows = self._rowsN
        Q = rows[:, self._IN[pos]] - rows[:, self._JN[pos]]       # |N| x len(pos)
        Ht = Q * Q - self._HNI_HIIinv @ self._H_NI[pos, :].T
        return 0.25 * Ht

    def full_completion(self, allE):
        """(K_NN)^{-1} on all non-edges of the base, K_ef = tr(Theta E_e Theta E_f), as one
        quarter of the Schur complement of the Hessian with respect to the base edges"""
        if self._Kinv is None:
            self.completion_setup(allE)
            rows = self._rowsN; IN = self._IN; JN = self._JN
            Q_NN = rows[:, IN] - rows[:, JN]
            Ht = Q_NN * Q_NN - self._HNI_HIIinv @ self._H_NI.T
            self._Kinv = 0.125 * (Ht + Ht.T)
        return self._Kinv, self._Npos, self._IN, self._JN


def kmat(Th, Ia, Ja, Ib, Jb):
    """K_{ef} = tr(Theta E_e Theta E_f) = 2 (Th_ik Th_jl + Th_il Th_jk), e=(i,j), f=(k,l)"""
    return 2.0 * (Th[np.ix_(Ia, Ib)] * Th[np.ix_(Ja, Jb)] + Th[np.ix_(Ia, Jb)] * Th[np.ix_(Ja, Ib)])


def phi_bound(sF, s2):
    """loss bound sF^2 / (2 (1 - s2)) for a traceless symmetric perturbation with
    Frobenius norm <= sF and spectral norm <= s2 < 1"""
    sF = np.asarray(sF, float); s2 = np.asarray(s2, float)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(s2 < 1.0, sF ** 2 / (2.0 * np.maximum(1.0 - s2, 1e-300)), np.inf)
    return out


class SwapCertifier:
    """
    Certifies that no connected support S with |S| <= k and |S \\ S*| <= r has a
    smaller profile value than the incumbent S* (up to tol), or replaces S*
    by an improving support and restarts from radius one.  An improving support
    with fewer than k edges is padded to k edges with the absent pairs of largest
    misfit, which cannot increase its profile value, so the incumbent always has
    k edges.  Exact (global) when r = k.

    signed=True      : general Huesler-Reiss model (no sign constraint on the
                       conductances); the box bound of the self-concordance
                       test comes from the objective value.
    completion=True  : pinning bound with completion of the free variogram
                       entries (removed and excluded pairs, and at the leaves
                       all pairs outside the tested support).
    """
    def __init__(self, Gam, edges, tol=1e-7, verbose=False, signed=False, completion=True, max_rounds=4):
        self.Gam = Gam; self.d = Gam.shape[0]
        self.allE = edges_of(self.d); self.m = len(self.allE)
        self.Mv = {e: 1.0 / Gam[e] for e in self.allE}
        self.tol = tol; self.verbose = verbose; self.deadline = None
        self.signed = signed; self.completion = completion; self.max_rounds = max_rounds
        self.cache = {}
        self.k_target = len(edges)
        self.stats = dict(padded=0, A=0, A_conn=0, A_disc=0, pruned_top=0, dfs_nodes=0, leaves=0,
                          leaf_pin=0, leaf_sc=0, refits=0, bridge_cases=0, joins=0,
                          excluded=0, rounds=0, alive0=0, alive_r=0, alive_f=0, kinv=0)
        if signed:
            d = self.d
            self.lam_c = np.array([np.linalg.eigvalsh(sigma_k(Gam, c))[0] for c in range(d)])
        self.set_incumbent(edges)

    # -- incumbent -----------------------------------------------------------
    def set_incumbent(self, edges):
        fit = self.refit(edges)
        self.S = list(fit.edges); self.k = len(self.S); self.F = fit.F; self.fit = fit
        Sset = set(self.S)
        self.absent = [e for e in self.allE if e not in Sset]

    def _fit(self, edges, w0=None):
        return fit_support(self.Gam, edges, w0=w0, signed=self.signed)

    def refit(self, edges):
        key = frozenset(edges)
        if key in self.cache:
            return self.cache[key]
        fit = self._fit(sorted(edges))
        self.stats['refits'] += 1
        self.cache[key] = fit
        return fit

    def improve(self, edges):
        fit = self.refit(edges)
        if self.verbose:
            print(f"   improvement {self.F:.8f} -> {fit.F:.8f}", flush=True)
        edges = list(fit.edges)
        if len(edges) < self.k_target:
            # pad to k edges with the absent pairs of largest misfit; f cannot increase
            S = set(edges)
            absent = [e for e in self.allE if e not in S]
            gap = np.array([fit.R[e] - self.Gam[e] for e in absent])
            score = np.abs(gap) if self.signed else gap
            order = np.argsort(-score, kind='stable')[:self.k_target - len(edges)]
            edges = edges + [absent[i] for i in order]
            self.stats['padded'] += 1
        self.set_incumbent(edges)

    def base_from_fit(self, fit):
        return BaseInfo(self.Gam, fit.edges, fit.w, fit.F, fit.Theta, fit.G, signed=self.signed)

    # -- misfit and boxes -------------------------------------------------------
    def misfit(self, base, I_, J_):
        delta = self.Gam[I_, J_] - base.R[I_, J_]
        return delta if self.signed else np.minimum(delta, 0.0)

    def box(self, base, I_, J_):
        """valid bound on the conductance of any added edge in the refitted support"""
        if not self.signed:
            return 1.0 / self.Gam[I_, J_]
        if base._Tbox is None:
            d = self.d; v = base.F + np.log(d)
            T = np.empty(d)
            for c in range(d):
                lam = max(self.lam_c[c], 1e-300)
                g = lambda t: -(d - 1) * np.log(t / (d - 1)) + lam * t
                lo = (d - 1) / lam; hi = 2 * lo
                while g(hi) < v:
                    hi *= 2
                for _ in range(80):
                    mid = 0.5 * (lo + hi)
                    if g(mid) < v:
                        lo = mid
                    else:
                        hi = mid
                T[c] = hi
            base._Tbox = T
        T = base._Tbox
        srt = np.sort(T)[:3]; arg = np.argsort(T)[:3]
        out = np.empty(len(I_))
        for t in range(len(I_)):
            for pos in range(3):
                if arg[pos] != I_[t] and arg[pos] != J_[t]:
                    out[t] = 0.5 * srt[pos]; break
        return out

    def pin_loss(self, base, B):
        r = len(B)
        nodes = np.empty(2 * r, int); C = np.zeros((2 * r, 2 * r))
        for t, (i, j) in enumerate(B):
            nodes[2 * t] = i; nodes[2 * t + 1] = j
            delta = self.Gam[i, j] - base.R[i, j]
            if not self.signed:
                delta = min(delta, 0.0)
            C[2 * t, 2 * t + 1] = C[2 * t + 1, 2 * t] = -delta / 2.0
        W = base.Theta[np.ix_(nodes, nodes)]
        lam = np.linalg.eigvals(C @ W).real
        if lam.min() <= -1.0 + 1e-12:
            return np.inf
        return -np.log1p(lam).sum()

    def sc_bound(self, base, B):
        I_ = np.array([e[0] for e in B]); J_ = np.array([e[1] for e in B])
        a = np.abs(self.misfit(base, I_, J_))
        Hs = base.schur(B)
        M = self.box(base, I_, J_)
        return sc_gain(a, Hs, M, isotropic=self.signed)

    def warm_refit(self, base, B):
        """refit base plus B, warm-started at the base conductances"""
        edges = sorted(set(base.edges) | set(B))
        key = frozenset(edges)
        if key in self.cache:
            return self.cache[key]
        wmap = dict(zip(base.edges, base.w))
        if self.signed:
            w0 = np.array([wmap[e] if e in wmap else 1.0 / self.Gam[e] - 1.0 / base.R[e] for e in edges])
        else:
            w0 = np.array([wmap[e] if e in wmap else max(1.0 / self.Gam[e] - 1.0 / base.R[e], 1e-6) for e in edges])
        fit = self._fit(edges, w0=w0)
        self.stats['refits'] += 1
        self.cache[key] = fit
        return fit

    def check_time(self):
        if self.deadline is not None and time.time() > self.deadline:
            raise TimeoutError

    # -- completed perturbations ---------------------------------------------------
    def _exact_loss(self, base, Delta):
        """exact loss -log det(I + K) and spectral norm of K = -1/2 S Delta S"""
        S = base.sqrt_theta()
        Kp = -0.5 * (S @ Delta @ S)
        lam = np.linalg.eigvalsh(0.5 * (Kp + Kp.T))
        sig = max(abs(lam[0]), abs(lam[-1]))
        if lam[0] <= -1.0 + 1e-12:
            return np.inf, sig
        return float(-np.log1p(lam).sum()), sig

    def _delta_restricted(self, Ic, Jc, dc, Tp, IF, JF, idx):
        d = self.d
        Delta = np.zeros((d, d))
        for c in idx:
            Delta[Ic[c], Jc[c]] += dc[c]; Delta[Jc[c], Ic[c]] += dc[c]
            if Tp is not None:
                col = Tp[:, c] * dc[c]
                np.add.at(Delta, (IF, JF), col); np.add.at(Delta, (JF, IF), col)
        return Delta

    def full_positions(self, base, U):
        Kinv, Npos, IN, JN = base.full_completion(self.allE)
        return Kinv, np.array([Npos[e] for e in U], int), IN, JN

    def full_norm2(self, Kinv, nb, dB):
        """squared Frobenius norm of the perturbation pinned on B and completed
        over all other pairs outside the base: (1/4) d' (Kinv_BB)^{-1} d"""
        sub = Kinv[np.ix_(nb, nb)]
        try:
            coef = np.linalg.solve(sub, dB)
        except np.linalg.LinAlgError:
            return np.inf, None
        return 0.25 * float(dB @ coef), coef

    def full_loss_exact(self, base, Kinv, nb, dB, IN, JN, coef=None):
        """exact loss of the fully completed pinned point (Cholesky of the dual covariance)"""
        if coef is None:
            _, coef = self.full_norm2(Kinv, nb, dB)
            if coef is None:
                return np.inf
        x = Kinv[:, nb] @ coef
        d = self.d
        Delta = np.zeros((d, d))
        Delta[IN, JN] = x; Delta[JN, IN] = x
        rs = Delta.sum(1)
        PDP = Delta - (rs[:, None] + rs[None, :]) / d + rs.sum() / d ** 2
        M = base.G - 0.5 * PDP
        try:
            Cm = np.linalg.cholesky(M)
        except np.linalg.LinAlgError:
            return np.inf
        return base.logdetG - float(2.0 * np.log(np.diag(Cm)).sum())

    def full_loss_exact_batch(self, base, X, IN, JN):
        """exact losses of the fully completed pinned points whose perturbations on the
        non-edges of the base are the columns of X (|N| x nsets)"""
        d = self.d; nsets = X.shape[1]
        out = np.empty(nsets)
        chunk = max(1, min(4000, int(3e6 / (d * d))))
        for a in range(0, nsets, chunk):
            Xc = X[:, a:a + chunk].T                     # (nc, |N|)
            nc = Xc.shape[0]
            Delta = np.zeros((nc, d, d))
            Delta[:, IN, JN] = Xc; Delta[:, JN, IN] = Xc
            rs = Delta.sum(2)
            PDP = Delta - (rs[:, :, None] + rs[:, None, :]) / d + (rs.sum(1) / d ** 2)[:, None, None]
            M = base.G[None, :, :] - 0.5 * PDP
            try:
                Cm = np.linalg.cholesky(M)
                ld = 2.0 * np.log(np.diagonal(Cm, axis1=1, axis2=2)).sum(1)
                out[a:a + nc] = base.logdetG - ld
            except np.linalg.LinAlgError:
                lam = np.linalg.eigvalsh(M)
                ok = lam[:, 0] > 1e-14
                ld = np.where(ok, np.log(np.maximum(lam, 1e-300)).sum(1), -np.inf)
                out[a:a + nc] = np.where(ok, base.logdetG - ld, np.inf)
        return out

    # -- connected base: enumerate additions ------------------------------------
    def search_connected(self, base_edges, base, r_left, avail):
        """Exhaustive over all B subset of avail with |B| <= r_left (see the
        correctness proof in the paper).  Sets are enumerated over the edges with
        nonzero misfit at the base; supersets that contain edges with zero misfit
        are covered by the pinning bound or by recursion from a refitted base."""
        self.check_time()
        target = self.F - self.tol
        if base.F < target:
            self.improve(base_edges); return True
        if r_left == 0 or not avail:
            return False
        Th = base.Theta; th = np.diag(Th)
        Ia = np.fromiter((e[0] for e in avail), int, len(avail)); Ja = np.fromiter((e[1] for e in avail), int, len(avail))
        delta_all = self.misfit(base, Ia, Ja)
        s0_all = np.abs(delta_all) * np.sqrt(th[Ia] * th[Ja] / 2.0)
        order = np.argsort(-s0_all)
        order = [int(t) for t in order if s0_all[t] > 1e-14]
        if not order:
            return False
        if base.F - psi(s0_all[order[:r_left]].sum()) >= target:
            self.stats['pruned_top'] += 1
            return False
        sq = np.sqrt(2.0)
        C = list(order)                      # indices into avail, decreasing plain s
        excluded = []
        # ---- restricted completion and exclusion rounds (superset bounds), used by the depth-first search
        if r_left >= 3 and self.completion:
            C, excluded, s, sig, Gm = self._exclusion_rounds(base, base_edges, avail, Ia, Ja, delta_all, s0_all, C, r_left, target)
            if not C:
                return False
        n = len(C)
        Ic = Ia[C]; Jc = Ja[C]; dc = delta_all[C]; s0 = s0_all[C]
        Gm0 = 0.25 * np.outer(dc, dc) * kmat(Th, Ic, Jc, Ic, Jc)
        if not (r_left >= 3 and self.completion):
            s = s0.copy(); sig = s0 / sq; Gm = Gm0
        U = [avail[c] for c in C]
        exset_pairs = set(avail[c] for c in excluded)
        # ---- single additions.  For a complete set (r_left == 1) every bound applies; for a
        # set that may be extended by edges of zero misfit only bounds that remain valid for
        # such supersets apply: plain pinning and the completion clipped to the slack of the
        # zero-misfit pairs.
        pin1 = np.where(s0 ** 2 < 2.0, -np.log1p(-np.minimum(s0 ** 2, 1.999) / 2.0), np.inf)
        alive = base.F - pin1 < target
        self.stats['alive0'] += int(alive.sum())
        Kinv = None; nb = IN = JN = None
        clip_pos = clip_slack = None
        if self.completion and (alive.any() or r_left >= 2):
            Npos, IN, JN = base.completion_setup(self.allE)
            nb = np.array([Npos[e] for e in U], int)
            if r_left >= 2:
                zero = [avail[c] for c in range(len(avail)) if s0_all[c] <= 1e-14]
                if zero:
                    clip_pos = np.array([Npos[e] for e in zero], int)
                    clip_slack = np.maximum(self.Gam[IN[clip_pos], JN[clip_pos]] - base.R[IN[clip_pos], JN[clip_pos]], 0.0)
        def clip(X):
            if clip_pos is not None and r_left >= 2:
                X[clip_pos, :] = np.minimum(X[clip_pos, :], clip_slack[:, None])
            return X
        if alive.any():
            rows = base.G[Ic, :] - base.G[Jc, :]
            QUI = rows @ base.B
            qUU = rows[np.arange(n), Ic] - rows[np.arange(n), Jc]
            Ht = qUU ** 2 - np.einsum('ij,jk,ik->i', QUI ** 2, base.H_II_inv, QUI ** 2)   # Schur complement diagonal
            if r_left == 1:
                a = np.abs(dc)
                Mv = self.box(base, Ic, Jc)
                g1 = sc_gain_single_vec(a, Ht, Mv)
                self.stats['leaf_sc'] += int((alive & (base.F - g1 >= target)).sum())
                alive &= base.F - g1 < target
                if self.completion and alive.any():
                    with np.errstate(divide='ignore', invalid='ignore'):
                        sf = np.sqrt(np.where(Ht > 0, dc ** 2 / Ht, np.inf))
                    alive &= base.F - phi_bound(sf, sf) < target
        if self.completion and alive.any():
            self.stats['alive_r'] += int(alive.sum())
            idx = np.where(alive)[0]
            cols = base.kinv_columns(nb[idx])                     # |N| x len(idx)
            kd = cols[nb[idx], np.arange(len(idx))]
            with np.errstate(divide='ignore', invalid='ignore'):
                X = clip(cols * np.where(kd > 0, dc[idx] / kd, 0.0)[None, :])
            lf = self.full_loss_exact_batch(base, X, IN, JN)
            lf = np.where(kd > 0, lf, np.inf) if r_left == 1 else np.where(kd > 0, lf, 0.0)
            alive[idx[base.F - lf >= target]] = False
        self.stats['alive_f'] += int(alive.sum())
        if r_left == 1:
            self.stats['leaves'] += int(alive.sum())
            for t in np.where(alive)[0]:
                fit = self.warm_refit(base, [U[t]])
                if fit is not None and fit.F < target:
                    self.improve(list(fit.edges)); return True
            return False
        # ---- r_left >= 2: singles that survive the superset-valid bounds are refitted and searched recursively
        recursed = np.zeros(n, bool)
        for t in np.where(alive)[0]:
            self.check_time()
            recursed[t] = True
            fit = self.warm_refit(base, [U[t]])
            if fit is None:
                continue
            if fit.F < target:
                self.improve(list(fit.edges)); return True
            avail2 = [e for e in avail if e != U[t] and e not in exset_pairs]
            if self.search_connected(list(fit.edges), self.base_from_fit(fit), r_left - 1, avail2):
                return True
        keep = np.where(~recursed)[0]
        if len(keep) < 2:
            return False

        def get_kinv():
            nonlocal Kinv
            if Kinv is None:
                Kinv = base.full_completion(self.allE)[0]
            return Kinv

        def leaf_loss(idx, complete):
            """lower bound on the loss of the set idx: plain pinning, then the completed bound
            (exact; clipped to the slack of the zero-misfit pairs if the set may be extended)"""
            B = [U[i] for i in idx]
            l = self.pin_loss(base, B)
            if base.F - l >= target or not self.completion:
                return l
            Kinv = get_kinv()
            n2, coef = self.full_norm2(Kinv, nb[idx], dc[idx])
            if coef is None:
                return l
            if complete:
                sf = np.sqrt(n2) if np.isfinite(n2) else np.inf
                l = min(l, float(phi_bound(sf, sf)))
                if base.F - l >= target:
                    return l
            X = Kinv[:, nb[idx]] @ coef
            X = X[:, None]
            if not complete:
                X = clip(X)
            return min(l, float(self.full_loss_exact_batch(base, X, IN, JN)[0]))

        def ext_bound(idx, t_more, start_pos):
            # bound for idx plus up to t_more later candidates (positions >= start_pos in keep)
            later = keep[start_pos:]
            fro2 = Gm[np.ix_(idx, idx)].sum(); fro = np.sqrt(max(fro2, 0.0))
            fro2p = Gm0[np.ix_(idx, idx)].sum(); frop = np.sqrt(max(fro2p, 0.0))
            if t_more <= 0 or len(later) == 0:
                tau = min(fro, sig[idx].sum()); taup = min(frop, s0[idx].sum() / sq)
                b1 = np.inf if tau >= 1.0 else fro2 / (2.0 * (1.0 - tau))
                b2 = np.inf if taup >= 1.0 else fro2p / (2.0 * (1.0 - taup))
                return min(b1, b2)
            ex_s = np.sort(s[later])[::-1][:t_more].sum(); ex_sig = np.sort(sig[later])[::-1][:t_more].sum()
            F_ext = fro + ex_s
            tau = min(F_ext, min(fro, sig[idx].sum()) + ex_sig)
            b1 = np.inf if tau >= 1.0 else F_ext ** 2 / (2.0 * (1.0 - tau))
            ex_s0 = np.sort(s0[later])[::-1][:t_more].sum()
            F_extp = frop + ex_s0
            taup = min(F_extp, min(frop, s0[idx].sum() / sq) + ex_s0 / sq)
            b2 = np.inf if taup >= 1.0 else F_extp ** 2 / (2.0 * (1.0 - taup))
            return min(b1, b2)

        ks = keep
        S0 = s0[ks]
        # plain pair bounds (valid for the pair and for its extensions by zero-misfit edges)
        fro2p = S0[:, None] ** 2 + S0[None, :] ** 2 + 2.0 * Gm0[np.ix_(ks, ks)]
        frop = np.sqrt(np.maximum(fro2p, 0.0))
        taup = np.minimum(frop, (S0[:, None] + S0[None, :]) / sq)
        pair_plain = np.where(taup < 1.0, fro2p / (2.0 * np.maximum(1.0 - taup, 1e-12)), np.inf)
        kb = nb[ks] if self.completion else None
        dk = dc[ks]

        def pair_cheap(P, Q):
            """fully completed Frobenius bound for the pairs (P, Q), from the 2x2 blocks of Kinv"""
            Kinv = get_kinv()
            ap = Kinv[kb[P], kb[P]]; aq = Kinv[kb[Q], kb[Q]]; c_ = Kinv[kb[P], kb[Q]]; dp = dk[P]; dq = dk[Q]
            dt = ap * aq - c_ ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                q2 = (aq * dp ** 2 - 2.0 * c_ * dp * dq + ap * dq ** 2) / dt
                sf = np.sqrt(np.where(dt > 0, 0.25 * q2, np.inf))
            return phi_bound(sf, sf)

        def pair_exact(P, Q, clipped):
            Kinv = get_kinv()
            ap = Kinv[kb[P], kb[P]]; aq = Kinv[kb[Q], kb[Q]]; c_ = Kinv[kb[P], kb[Q]]; dp = dk[P]; dq = dk[Q]; dt = ap * aq - c_ ** 2
            with np.errstate(divide='ignore', invalid='ignore'):
                cp = (aq * dp - c_ * dq) / dt; cq = (ap * dq - c_ * dp) / dt
            X = Kinv[:, kb[P]] * cp[None, :] + Kinv[:, kb[Q]] * cq[None, :]
            if clipped:
                X = clip(X)
            lf = self.full_loss_exact_batch(base, X, IN, JN)
            return np.where(dt > 0, lf, np.inf)

        if r_left == 2:
            cand = np.argwhere(np.triu(base.F - pair_plain < target, 1))
            self.stats['dfs_nodes'] += len(ks) * (len(ks) - 1) // 2
            if len(cand) == 0:
                return False
            if self.completion:
                bnd = pair_cheap(cand[:, 0], cand[:, 1])
                cand = cand[np.where(base.F - bnd < target)[0]]
                self.stats['leaves'] += len(cand)
                if len(cand) == 0:
                    return False
                lf = pair_exact(cand[:, 0], cand[:, 1], False)
                surv = np.where(base.F - lf < target)[0]
                self.stats['leaf_pin'] += len(cand) - len(surv)
                cand = cand[surv]
            else:
                self.stats['leaves'] += len(cand)
            for p, q_ in cand:
                self.check_time()
                B = [U[ks[p]], U[ks[q_]]]
                if not self.completion:
                    if base.F - self.pin_loss(base, B) >= target:
                        self.stats['leaf_pin'] += 1; continue
                if base.F - self.sc_bound(base, B) >= target:
                    self.stats['leaf_sc'] += 1; continue
                fit = self.warm_refit(base, B)
                if fit is not None and fit.F < target:
                    self.improve(list(fit.edges)); return True
            return False

        if r_left == 3 and self.completion:
            # pairs as partial sets: superset-valid bounds, then refit and recursion
            cand = np.argwhere(np.triu(base.F - pair_plain < target, 1))
            self.stats['dfs_nodes'] += len(ks) * (len(ks) - 1) // 2
            if len(cand):
                lf = pair_exact(cand[:, 0], cand[:, 1], True)
                cand = cand[np.where(base.F - lf < target)[0]]
            for p, q_ in cand:
                self.check_time()
                B = [U[ks[p]], U[ks[q_]]]
                fit = self.warm_refit(base, B)
                if fit is None:
                    continue
                if fit.F < target:
                    self.improve(list(fit.edges)); return True
                avail2 = [e for e in avail if e not in set(B) and e not in exset_pairs]
                if self.search_connected(list(fit.edges), self.base_from_fit(fit), 1, avail2):
                    return True
            # triples: complete sets, fully completed bound vectorized over the third element
            Kinv = get_kinv()
            a_ = Kinv[kb, kb]; Kb = Kinv[np.ix_(kb, kb)]
            m = len(ks)
            for p in range(m - 2):
                self.check_time()
                Qi, Ri = np.triu_indices(m - p - 1, 1)
                Q = Qi + p + 1; R_ = Ri + p + 1
                if len(Q) == 0:
                    continue
                ap = a_[p]; aq = a_[Q]; ar = a_[R_]
                cpq = Kb[p, Q]; cpr = Kb[p, R_]; cqr = Kb[Q, R_]
                dp = dk[p]; dq = dk[Q]; dr = dk[R_]
                det3 = ap * aq * ar + 2.0 * cpq * cqr * cpr - ap * cqr ** 2 - aq * cpr ** 2 - ar * cpq ** 2
                App = aq * ar - cqr ** 2; Aqq = ap * ar - cpr ** 2; Arr = ap * aq - cpq ** 2
                Apq = cpr * cqr - ar * cpq; Apr = cpq * cqr - aq * cpr; Aqr = cpq * cpr - ap * cqr
                with np.errstate(divide='ignore', invalid='ignore'):
                    form = (App * dp ** 2 + Aqq * dq ** 2 + Arr * dr ** 2 + 2.0 * (Apq * dp * dq + Apr * dp * dr + Aqr * dq * dr)) / det3
                    sf3 = np.sqrt(np.where(det3 > 0, 0.25 * form, np.inf))
                bound3 = phi_bound(sf3, sf3)
                self.stats['dfs_nodes'] += len(Q)
                sel = np.where(base.F - bound3 < target)[0]
                if len(sel) == 0:
                    continue
                self.stats['leaves'] += len(sel)
                with np.errstate(divide='ignore', invalid='ignore'):
                    cp = (App[sel] * dp + Apq[sel] * dq[sel] + Apr[sel] * dr[sel]) / det3[sel]
                    cq = (Apq[sel] * dp + Aqq[sel] * dq[sel] + Aqr[sel] * dr[sel]) / det3[sel]
                    cr = (Apr[sel] * dp + Aqr[sel] * dq[sel] + Arr[sel] * dr[sel]) / det3[sel]
                X = Kinv[:, [kb[p]]] * cp[None, :] + Kinv[:, kb[Q[sel]]] * cq[None, :] + Kinv[:, kb[R_[sel]]] * cr[None, :]
                lf = self.full_loss_exact_batch(base, X, IN, JN)
                surv = np.where(base.F - lf < target)[0]
                self.stats['leaf_pin'] += len(sel) - len(surv)
                for t in surv:
                    self.check_time()
                    B = [U[ks[p]], U[ks[Q[sel[t]]]], U[ks[R_[sel[t]]]]]
                    if base.F - self.sc_bound(base, B) >= target:
                        self.stats['leaf_sc'] += 1; continue
                    fit = self.warm_refit(base, B)
                    if fit is not None and fit.F < target:
                        self.improve(list(fit.edges)); return True
            return False

        # general depth-first search for r_left >= 3 without completion, and r_left >= 4
        stack = [(0, [])]
        while stack:
            start, chosen = stack.pop()
            for pos in range(start, len(keep)):
                self.check_time()
                idx = chosen + [keep[pos]]
                if len(idx) < 2:
                    stack.append((pos + 1, idx)); continue
                self.stats['dfs_nodes'] += 1
                t = r_left - len(idx)
                if base.F - ext_bound(idx, t, pos + 1) >= target:
                    continue
                B = [U[i] for i in idx]
                self.stats['leaves'] += 1
                if base.F - leaf_loss(idx, t == 0) >= target:
                    self.stats['leaf_pin'] += 1
                    if t > 0:
                        stack.append((pos + 1, idx))
                    continue
                if t == 0:
                    if base.F - self.sc_bound(base, B) >= target:
                        self.stats['leaf_sc'] += 1; continue
                    fit = self.warm_refit(base, B)
                    if fit is not None and fit.F < target:
                        self.improve(list(fit.edges)); return True
                    continue
                fit = self.warm_refit(base, B)
                if fit is None:
                    continue
                if fit.F < target:
                    self.improve(list(fit.edges)); return True
                Bset = set(B)
                avail2 = [e for e in avail if e not in Bset and e not in exset_pairs]
                if self.search_connected(list(fit.edges), self.base_from_fit(fit), t, avail2):
                    return True
        return False

    def _exclusion_rounds(self, base, base_edges, avail, Ia, Ja, delta_all, s0_all, C, r_left, target):
        """completion of the pinned point over the pairs that are neither in the base
        nor candidates (removed and excluded edges); returns the surviving candidates
        with their completed Frobenius norms, spectral norms and Gram matrix"""
        Th = base.Theta
        Iset = set(base_edges); avail_set = set(avail)
        Fpairs = [e for e in self.allE if e not in Iset and e not in avail_set]
        excluded = []; rounds = 0
        while True:
            self.stats['rounds'] += 1; rounds += 1
            Ic = Ia[C]; Jc = Ja[C]; dc = delta_all[C]; s0 = s0_all[C]; n = len(C)
            K_CC = kmat(Th, Ic, Jc, Ic, Jc)
            Tp = None; IF = JF = None
            if Fpairs:
                IF = np.fromiter((e[0] for e in Fpairs), int, len(Fpairs)); JF = np.fromiter((e[1] for e in Fpairs), int, len(Fpairs))
                K_FF = kmat(Th, IF, JF, IF, JF); K_FC = kmat(Th, IF, JF, Ic, Jc)
                ridge = 1e-10 * max(float(np.diag(K_FF).mean()), 1e-300)
                try:
                    cho = cho_factor(K_FF + ridge * np.eye(len(Fpairs)), lower=True)
                    Tp = -cho_solve(cho, K_FC)
                except Exception:
                    Tp = None
            if Tp is not None:
                Kc = K_CC + K_FC.T @ Tp + Tp.T @ K_FC + Tp.T @ (K_FF @ Tp)
                Kc = 0.5 * (Kc + Kc.T)
                Gm = 0.25 * np.outer(dc, dc) * Kc
                s = np.sqrt(np.maximum(np.diag(Gm), 0.0))
            else:
                Gm = 0.25 * np.outer(dc, dc) * K_CC; s = s0.copy()

            def top_sums(vals, t):
                if t <= 0 or n <= 1:
                    return np.zeros(n)
                srt = np.sort(vals)[::-1]
                pref = srt[:t].sum()
                nxt = srt[t] if t < n else 0.0
                rank = np.empty(n, int); rank[np.argsort(-vals)] = np.arange(n)
                return np.where(rank < t, pref - vals + nxt, pref)
            b_plain = np.array([psi(x) for x in s0 + top_sums(s0, r_left - 1)])
            b_comp = phi_bound(s + top_sums(s, r_left - 1), s + top_sums(s, r_left - 1))
            bound = np.minimum(b_plain, b_comp)
            excl = np.where(base.F - bound >= target)[0]
            if len(excl) == 0 or rounds >= self.max_rounds:
                sig = np.empty(n)
                for c in range(n):
                    _, sig[c] = self._exact_loss(base, self._delta_restricted(Ic, Jc, dc, Tp, IF, JF, [c]))
                sig = np.minimum(sig, s)
                b_comp = phi_bound(s + top_sums(s, r_left - 1), sig + top_sums(sig, r_left - 1))
                bound = np.minimum(b_plain, b_comp)
                excl = np.where(base.F - bound >= target)[0] if rounds < self.max_rounds else np.zeros(0, int)
                if len(excl) == 0:
                    return C, excluded, s, sig, Gm
            exset = set(int(c) for c in excl)
            excluded += [C[c] for c in exset]
            Fpairs = Fpairs + [avail[C[c]] for c in exset]
            C = [C[c] for c in range(n) if c not in exset]
            self.stats['excluded'] += len(exset)
            if not C:
                return C, excluded, None, None, None

    # -- disconnected base ----------------------------------------------------
    def component_fits(self, base_edges):
        d = self.d
        comp = components_of(d, base_edges)
        c = comp.max() + 1
        fits = []
        for cc in range(c):
            nodes = np.where(comp == cc)[0]
            loc = {v: i for i, v in enumerate(nodes)}
            sub = sorted((loc[i], loc[j]) for (i, j) in base_edges if comp[i] == cc)
            if len(nodes) == 1:
                fits.append((nodes, None)); continue
            key = ('comp', frozenset((int(nodes[i]), int(nodes[j])) for i, j in sub), tuple(int(v) for v in nodes))
            if key not in self.cache:
                self.cache[key] = fit_support(self.Gam[np.ix_(nodes, nodes)], sub, signed=self.signed); self.stats['refits'] += 1
            fits.append((nodes, self.cache[key]))
        return comp, fits

    def joined_value(self, fits, bridges):
        d = self.d; c = len(fits)
        val = -np.log(d) + (c - 1) + sum(np.log(self.Gam[i, j]) for i, j in bridges)
        for nodes, fit in fits:
            if fit is not None:
                val += fit.F + np.log(len(nodes))
        return val

    def joined_base(self, fits, bridges):
        """connected base obtained by joining component fits with bridges (exact)"""
        edges_w = []
        for nodes, fit in fits:
            if fit is not None:
                edges_w += [((int(nodes[i]), int(nodes[j])), we) for (i, j), we in zip(fit.edges, fit.w)]
        for b in bridges:
            edges_w.append((b, 1.0 / self.Gam[b]))
        edges_w.sort()
        E = [e for e, _ in edges_w]; w = np.array([we for _, we in edges_w])
        L = laplacian(self.d, E, w)
        G = np.linalg.inv(L + np.ones((self.d, self.d)) / self.d)
        self.stats['joins'] += 1
        return BaseInfo(self.Gam, E, w, self.joined_value(fits, bridges), L, G, signed=self.signed)

    def search_disconnected(self, base_edges, r_left, avail, comp, fits):
        target = self.F - self.tol
        c = comp.max() + 1
        if c - 1 > r_left:
            return False
        cross = [e for e in avail if comp[e[0]] != comp[e[1]]]
        if not cross:
            return False
        if c - 1 == r_left:
            self.stats['bridge_cases'] += 1
            H = nx.Graph(); H.add_nodes_from(range(c))
            for (i, j) in cross:
                a, b = comp[i], comp[j]; wgt = np.log(self.Gam[i, j])
                if H.has_edge(a, b):
                    if wgt < H[a][b]['weight']:
                        H[a][b]['weight'] = wgt; H[a][b]['edge'] = (i, j)
                else:
                    H.add_edge(a, b, weight=wgt, edge=(i, j))
            if not nx.is_connected(H):
                return False
            T = nx.minimum_spanning_tree(H)
            bridges = [T[a][b]['edge'] for a, b in T.edges()]
            if self.joined_value(fits, bridges) < target:
                self.improve(sorted(set(base_edges) | set(bridges))); return True
            return False
        for t, b in enumerate(cross):
            avail2 = [e for e in avail if e != b and e not in cross[:t]]
            nb = sorted(set(base_edges) | {b})
            if c == 2:
                base = self.joined_base(fits, [b])
                if self.search_connected(nb, base, r_left - 1, avail2):
                    return True
            else:
                comp3, fits3 = self.component_fits(nb)
                if self.search_disconnected(nb, r_left - 1, avail2, comp3, fits3):
                    return True
        return False

    # -- main loop --------------------------------------------------------------
    def certify(self, r_max=None, time_limit=600.0, restart=True):
        t0 = time.time()
        self.deadline = t0 + time_limit
        r_max = self.k if r_max is None else min(r_max, self.k)
        certified = 0; r = 1; timed_out = False
        self.rlog = []                     # completion time of each radius, improvements, progress at timeout
        nA = 0
        while r <= r_max:
            improved = False; nA = 0
            try:
                for A in itertools.combinations(range(self.k), r):
                    self.check_time()
                    self.stats['A'] += 1
                    Aset = set(A)
                    base_edges = [self.S[t] for t in range(self.k) if t not in Aset]
                    if is_connected(self.d, base_edges):
                        self.stats['A_conn'] += 1
                        fit = self.refit(base_edges)
                        ok = self.search_connected(base_edges, self.base_from_fit(fit), r, self.absent)
                    else:
                        self.stats['A_disc'] += 1
                        comp, fits = self.component_fits(base_edges)
                        ok = self.search_disconnected(base_edges, r, self.absent, comp, fits)
                    if ok:
                        improved = True; break
                    nA += 1
            except TimeoutError:
                timed_out = True
            if timed_out:
                from math import comb
                self.rlog.append(dict(event='timeout', r=r, time=time.time() - t0, refits=self.stats['refits'],
                                      done=nA, total=comb(self.k, r), F=self.F))
                break
            self.rlog.append(dict(event='improved' if improved else 'certified', r=r, time=time.time() - t0,
                                  refits=self.stats['refits'], done=nA, F=self.F))
            if self.verbose:
                print(f"  r={r}: F={self.F:.8f} improved={improved} stats={self.stats} t={time.time()-t0:.1f}s", flush=True)
            if improved:
                if restart:
                    r = 1; certified = 0
                continue
            certified = r; r += 1
        return dict(edges=self.S, w=self.fit.w, F=self.F, radius=certified,
                    exact=(certified >= self.k), time=time.time() - t0, stats=dict(self.stats),
                    timed_out=timed_out, rlog=list(self.rlog))

# ---------------------------------------------------------------------------
# brute force (small d)
# ---------------------------------------------------------------------------
def brute_force(Gam, k, signed=False):
    d = Gam.shape[0]
    best = (np.inf, None)
    for comb in itertools.combinations(edges_of(d), k):
        if not is_connected(d, comb):
            continue
        v = profile_value(Gam, comb, signed=signed)
        if v < best[0]:
            best = (v, comb)
    return best

# ---------------------------------------------------------------------------
# competitors
# ---------------------------------------------------------------------------
def emtp2_mle(Gam, tol=1e-11):
    d = Gam.shape[0]
    return fit_support(Gam, edges_of(d), tol=tol, maxit=1000)

def emtp2_support(fit, eps=1e-8):
    return [e for e, we in zip(fit.edges, fit.w) if we > eps]

def l1_emtp2(Gam, rho):
    """l1-penalised EMTP2 estimator (extremal graphical lasso under total
    positivity): EMTP2 MLE with variogram Gam + rho off the diagonal."""
    d = Gam.shape[0]
    G2 = Gam + rho * (np.ones((d, d)) - np.eye(d))
    fit = fit_support(G2, edges_of(d), maxit=1000)
    return emtp2_support(fit)

def _lasso_cov(C, c, lam, maxit=500, tol=1e-8):
    """coordinate descent for min 1/2 b'Cb - c'b + lam |b|_1 (covariance form)"""
    p = len(c); b = np.zeros(p); dg = np.diag(C)
    for _ in range(maxit):
        bmax = 0.0
        for j in range(p):
            r = c[j] - C[j] @ b + dg[j] * b[j]
            bn = np.sign(r) * max(abs(r) - lam, 0.0) / dg[j]
            bmax = max(bmax, abs(bn - b[j])); b[j] = bn
        if bmax < tol:
            break
    return b

def eglearn(Gam, rho, method='ns', rule='or'):
    """Engelke, Lalancette and Volgushev (2022): for each k, a Gaussian
    graphical-model estimate on Sigma^(k) (neighbourhood selection by lasso
    regressions, or the graphical lasso), combined by majority vote over k."""
    d = Gam.shape[0]
    votes = np.zeros((d, d)); tot = np.zeros((d, d))
    for k in range(d):
        idx = [i for i in range(d) if i != k]
        S = sigma_k(Gam, k)
        A = np.zeros((d - 1, d - 1))
        if method == 'glasso':
            from sklearn.covariance import graphical_lasso
            try:
                _, P = graphical_lasso(S, alpha=rho, max_iter=500)
            except Exception:
                continue
            A = (np.abs(P) > 1e-8).astype(float); np.fill_diagonal(A, 0)
        else:
            # standardise as in a regression on correlations
            sd = np.sqrt(np.diag(S)); Cr = S / np.outer(sd, sd)
            N = np.zeros((d - 1, d - 1))
            for a in range(d - 1):
                others = [b for b in range(d - 1) if b != a]
                beta = _lasso_cov(Cr[np.ix_(others, others)], Cr[a, others], rho)
                for t, b in enumerate(others):
                    if abs(beta[t]) > 1e-10:
                        N[a, b] = 1.0
            A = np.maximum(N, N.T) if rule == 'or' else N * N.T
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                if a < b:
                    votes[i, j] += A[a, b]; tot[i, j] += 1
    return [(i, j) for i in range(d) for j in range(i + 1, d) if tot[i, j] > 0 and votes[i, j] > tot[i, j] / 2.0]

def fit_general(Gam, edges, solver=None):
    """MLE of the (signed) HR graphical model on a support: same objective as
    fit_support without the sign constraint (Newton, warm start at the EMTP2
    fit).  Returns (F, Theta)."""
    fit0 = fit_support(Gam, edges)
    if fit0 is None:
        return np.inf, None
    fit = fit_support(Gam, edges, w0=fit0.w, signed=True)
    if fit is None or not np.isfinite(fit.F):
        return fit0.F, fit0.Theta
    return min(fit.F, fit0.F), fit.Theta

def extremal_glasso(Gam, rho, solver=None):
    """Graphical lasso for extremes (Wan and Zhou): l1 penalty on the
    off-diagonal entries of the HR precision matrix, no sign constraint."""
    import cvxpy as cp
    d = Gam.shape[0]
    kref = d - 1
    S = sigma_k(Gam, kref)
    K = cp.Variable((d - 1, d - 1), PSD=True)
    Theta_last = -cp.sum(K, axis=1)
    off = []
    for i in range(d - 1):
        for j in range(i + 1, d - 1):
            off.append(K[i, j])
    for i in range(d - 1):
        off.append(Theta_last[i])
    pen = rho * cp.sum(cp.abs(cp.hstack(off)))
    prob = cp.Problem(cp.Maximize(cp.log_det(K) - cp.trace(S @ K) - pen))
    try:
        prob.solve(solver=solver or cp.CLARABEL)
    except Exception:
        prob.solve(solver=cp.SCS, eps=1e-8, max_iters=20000)
    Kv = 0.5 * (K.value + K.value.T)
    Theta = np.zeros((d, d)); Theta[:d - 1, :d - 1] = Kv
    Theta[:d - 1, d - 1] = -Kv.sum(1); Theta[d - 1, :d - 1] = -Kv.sum(0)
    scale = np.abs(Theta).max()
    return [(i, j) for i in range(d) for j in range(i + 1, d) if abs(Theta[i, j]) > 1e-6 * scale]

def bic(F, k, n):
    return n * F + k * np.log(n)

# ---------------------------------------------------------------------------
# synthetic models and scoring
# ---------------------------------------------------------------------------
def random_connected_graph(d, k, rng):
    T = nx.from_prufer_sequence(list(rng.integers(0, d, d - 2))) if d > 2 else nx.path_graph(d)
    E = set(tuple(sorted(e)) for e in T.edges())
    allE = [e for e in edges_of(d) if e not in E]
    extra = rng.choice(len(allE), size=k - (d - 1), replace=False)
    E |= set(allE[t] for t in extra)
    return sorted(E)

def random_hr_model(d, k, rng, wlo=0.5, whi=2.0):
    E = random_connected_graph(d, k, rng)
    w = rng.uniform(wlo, whi, len(E))
    L = laplacian(d, E, w)
    return E, w, resistance_from_L(L)[0]

def random_signed_hr_model(d, k, rng, frac_pos=0.3, wlo=0.5, whi=2.0, max_tries=500):
    """HR model with a fraction of positive off-diagonal precision entries
    (outside the EMTP2 class): the flagged edges receive negative conductances,
    scaled down until Theta is positive semidefinite with kernel spanned by 1."""
    for _ in range(max_tries):
        E = random_connected_graph(d, k, rng)
        w = rng.uniform(wlo, whi, len(E))
        flag = rng.uniform(size=len(E)) < frac_pos
        if not flag.any():
            flag[rng.integers(0, len(E))] = True
        c = 0.5
        for _ in range(8):
            ws = np.where(flag, -c * w, w)
            L = laplacian(d, E, ws)
            ev = np.linalg.eigvalsh(L)
            if ev[1] > 1e-2 * ev[-1] and abs(ev[0]) < 1e-8 * ev[-1]:
                return E, ws, resistance_from_L(L)[0]
            c *= 0.5
    raise RuntimeError("no valid signed model found")

def f1_score(true_edges, est_edges):
    T = set(map(tuple, true_edges)); S = set(map(tuple, est_edges))
    tp = len(T & S)
    prec = tp / len(S) if S else 0.0; rec = tp / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1
