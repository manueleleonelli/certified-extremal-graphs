"""
hrsparse: exact cardinality-constrained structure learning for Huesler-Reiss
graphical models (Laplacian parametrisation, outer approximation, solution path).

Self-contained: numpy, scipy, networkx, scikit-learn (for the EGlearn baseline).
"""
import numpy as np, networkx as nx, time, itertools
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree, connected_components
from scipy.stats import multivariate_normal

# ----------------------------------------------------------------------------
# graph bookkeeping
# ----------------------------------------------------------------------------
def edges_of(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]

def incidence(d, edges):
    B = np.zeros((d, len(edges)))
    for t, (i, j) in enumerate(edges):
        B[i, t] = 1.0; B[j, t] = -1.0
    return B

def laplacian(d, edges, w):
    B = incidence(d, edges)
    return (B * np.asarray(w)) @ B.T

def resistance_from_L(L):
    d = L.shape[0]
    G = np.linalg.inv(L + np.ones((d, d)) / d)
    g = np.diag(G)
    return g[:, None] + g[None, :] - 2.0 * G, G

def is_connected_support(d, edges):
    if len(edges) < d - 1: return False
    A = lil_matrix((d, d))
    for i, j in edges: A[i, j] = 1; A[j, i] = 1
    return connected_components(csr_matrix(A), directed=False)[0] == 1

def components_of(d, edges):
    A = lil_matrix((d, d))
    for i, j in edges: A[i, j] = 1; A[j, i] = 1
    return connected_components(csr_matrix(A), directed=False)[1]

def theta_from_gamma(Gam):
    d = Gam.shape[0]; P = np.eye(d) - np.ones((d, d)) / d
    return np.linalg.pinv(-0.5 * P @ Gam @ P)

def gamma_from_theta(Theta):
    return resistance_from_L(Theta)[0]

def edge_vec(Gam, edges):
    return np.array([Gam[i, j] for i, j in edges])

# ----------------------------------------------------------------------------
# data: multivariate Pareto HR sampling and empirical variogram
# ----------------------------------------------------------------------------
def sigma_k(Gam, k):
    d = Gam.shape[0]; idx = [i for i in range(d) if i != k]
    return 0.5 * (Gam[np.ix_(idx, [k])] + Gam[np.ix_([k], idx)] - Gam[np.ix_(idx, idx)])

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
            if len(sel) == 0: continue
            idx = [i for i in range(d) if i != k]
            logratio = Z[sel] @ chols[k].T - Gam[idx, k] / 2.0     # log(y_-k / y_k)
            yk = 1.0 / U[sel]                                       # standard Pareto
            Y = np.empty((len(sel), d)); Y[:, k] = yk
            Y[:, idx] = yk[:, None] * np.exp(logratio)
            N = (Y > 1).sum(1)
            acc = rng.uniform(size=len(sel)) < 1.0 / N
            out.extend(Y[acc])
    return np.array(out[:n])

def data2mpareto(X, p):
    """Empirical rank transform to standard Pareto margins and thresholding."""
    n, d = X.shape
    ranks = np.argsort(np.argsort(X, axis=0), axis=0) + 1
    F = ranks / (n + 1.0)
    Y = 1.0 / (1.0 - F)
    u = 1.0 / (1.0 - p)
    keep = Y.max(1) > u
    return Y[keep] / u

def emp_vario(Y, thr=1.0):
    """Averaged empirical variogram of Engelke and Hitz (2020): for each k, the
    variogram of log Y over the rows with Y_k > thr, then averaged over k."""
    n, d = Y.shape
    LY = np.log(Y)
    G = np.zeros((d, d)); cnt = 0
    for k in range(d):
        rows = LY[Y[:, k] > thr]
        if len(rows) < 2: continue
        diff = rows[:, :, None] - rows[:, None, :]
        G += diff.var(axis=0, ddof=1); cnt += 1
    return G / cnt

# ----------------------------------------------------------------------------
# inner problem: surrogate MLE on a fixed support (projected Newton)
# ----------------------------------------------------------------------------
def objective(Gam, d, edges, w):
    L = laplacian(d, edges, w)
    s, ld = np.linalg.slogdet(L + np.ones((d, d)) / d)
    if s <= 0: return np.inf
    return -ld + float(np.dot(edge_vec(Gam, edges), w))

def fit_support(Gam, edges, w0=None, emtp2=True, M=None, tol=1e-10, maxit=200):
    """min_w -log det(L(w)+J) + <Gam,w>  over  0<=w (emtp2) or |w|<=M (signed)."""
    d = Gam.shape[0]; k = len(edges)
    g_e = edge_vec(Gam, edges)
    B = incidence(d, edges)
    J = np.ones((d, d)) / d
    if w0 is None:
        w = 1.0 / g_e
    else:
        w = np.array(w0, dtype=float)
    def evalF(w):
        L = (B * w) @ B.T
        try:
            C = np.linalg.cholesky(L + J)
        except np.linalg.LinAlgError:
            return np.inf, None
        ld = 2.0 * np.log(np.diag(C)).sum()
        return -ld + float(g_e @ w), C
    F, C = evalF(w)
    if not np.isfinite(F):
        raise ValueError("initial point infeasible (disconnected support?)")
    for it in range(maxit):
        Ginv = np.linalg.inv(L_plus_J := ((B * w) @ B.T + J))
        Q = B.T @ Ginv @ B                      # k x k, Q_ef = b_e' (L+J)^-1 b_f
        R = np.diag(Q)                          # resistances on the support
        grad = g_e - R
        H = Q * Q
        if emtp2:
            free = (w > 1e-12) | (grad < 0)
        else:
            free = np.ones(k, bool)
            if M is not None:
                free = ~((np.abs(w) >= M - 1e-12) & (np.sign(w) * grad < 0))
        if not free.any():
            break
        p = np.zeros(k)
        try:
            p[free] = -np.linalg.solve(H[np.ix_(free, free)] + 1e-12 * np.eye(free.sum()), grad[free])
        except np.linalg.LinAlgError:
            p[free] = -grad[free]
        # projected backtracking (Armijo)
        t = 1.0; improved = False
        for _ in range(60):
            wn = w + t * p
            if emtp2: wn = np.maximum(wn, 0.0)
            if M is not None: wn = np.clip(wn, -M, M)
            Fn, Cn = evalF(wn)
            if np.isfinite(Fn) and Fn <= F + 1e-4 * grad @ (wn - w):
                improved = True; break
            t *= 0.5
        if not improved:
            break
        step = np.abs(wn - w).max()
        w, F = wn, Fn
        if step < tol * (1 + np.abs(w).max()):
            break
    L = (B * w) @ B.T
    Rfull, _ = resistance_from_L(L)
    return dict(w=w, F=F, R=Rfull, edges=list(edges), iters=it + 1)

# ----------------------------------------------------------------------------
# closed-form building blocks: MST, EMTP2 MLE, greedy
# ----------------------------------------------------------------------------
def mst_edges(Gam):
    d = Gam.shape[0]
    T = minimum_spanning_tree(csr_matrix(np.triu(Gam, 1)))
    T = T.tocoo()
    return sorted((int(min(i, j)), int(max(i, j))) for i, j in zip(T.row, T.col))

def emtp2_mle(Gam, tol=1e-10):
    d = Gam.shape[0]
    fit = fit_support(Gam, edges_of(d), emtp2=True, tol=tol, maxit=500)
    return fit

def divergence(rho):
    return np.log(rho) - 1.0 + 1.0 / rho

def greedy_forward(Gam, kmax, start=None, emtp2=True, ncand=5, M=None):
    """Forward selection from the MST: rank absent edges by the resistance
    misfit divergence, refit the best ncand exactly, add the best."""
    d = Gam.shape[0]
    E = start if start is not None else mst_edges(Gam)
    fit = fit_support(Gam, E, emtp2=emtp2, M=M)
    path = {len(E): fit}
    allE = edges_of(d)
    while len(E) < kmax:
        Es = set(E)
        absent = [e for e in allE if e not in Es]
        rho = np.array([fit['R'][e] / Gam[e] for e in absent])
        if emtp2:
            score = np.where(rho > 1, divergence(np.maximum(rho, 1)), 0.0)
        else:
            score = divergence(rho)
        order = np.argsort(-score)[:ncand]
        best = None
        for t in order:
            En = E + [absent[t]]
            w0 = np.append(fit['w'], max(1.0 / Gam[absent[t]] - 1.0 / fit['R'][absent[t]], 1e-8) if emtp2 else 1.0 / Gam[absent[t]] - 1.0 / fit['R'][absent[t]])
            fn = fit_support(Gam, En, w0=w0, emtp2=emtp2, M=M)
            if best is None or fn['F'] < best['F']:
                best = fn
        E = sorted(best['edges']); fit = best
        path[len(E)] = fit
    return path

def local_search(Gam, fit, emtp2=True, M=None, max_rounds=20):
    """1-swap local search at fixed cardinality."""
    d = Gam.shape[0]; E = list(fit['edges']); F = fit['F']
    allE = edges_of(d)
    for _ in range(max_rounds):
        improved = False
        Es = set(E)
        absent = [e for e in allE if e not in Es]
        rho = np.array([fit['R'][e] / Gam[e] for e in absent])
        cand_in = [absent[t] for t in np.argsort(-divergence(np.maximum(rho, 1 if emtp2 else 1e-9)))[:6]]
        # removal candidates: smallest conductances
        cand_out = [E[t] for t in np.argsort(fit['w'])[:6]]
        for eo in cand_out:
            for ei in cand_in:
                En = [e for e in E if e != eo] + [ei]
                if not is_connected_support(d, En): continue
                fn = fit_support(Gam, sorted(En), emtp2=emtp2, M=M)
                if fn['F'] < F - 1e-9:
                    E, F, fit = sorted(En), fn['F'], fn; improved = True; break
            if improved: break
        if not improved: break
    return fit

# ----------------------------------------------------------------------------
# outer approximation along the cardinality path
# ----------------------------------------------------------------------------
class OAPath:
    def __init__(self, Gam, emtp2=True, M=None, tol=1e-6, verbose=False):
        self.Gam = Gam; self.d = Gam.shape[0]; self.emtp2 = emtp2; self.M = M
        self.E = edges_of(self.d); self.m = len(self.E)
        self.eidx = {e: t for t, e in enumerate(self.E)}
        self.tol = tol; self.verbose = verbose
        self.cuts = []            # (f_t, slope vector g_t, z_t)  :  eta >= f_t + g_t (z - z_t)
        self.cutsets = []         # rows for connectivity
        self.visited = {}         # frozenset(edges) -> fit
        self.Me = 1.0 / edge_vec(Gam, self.E) if emtp2 else np.full(self.m, M)
        # global lower bound from the full model
        self.full = fit_support(Gam, self.E, emtp2=emtp2, M=M, maxit=500)
        self.f_full = self.full['F']
        self.log = []

    def zvec(self, edges):
        z = np.zeros(self.m); z[[self.eidx[e] for e in edges]] = 1.0; return z

    def add_point(self, fit):
        key = frozenset(fit['edges'])
        if key in self.visited: return self.visited[key]
        self.visited[key] = fit
        z = self.zvec(fit['edges'])
        Rv = np.array([fit['R'][e] for e in self.E]); Gv = edge_vec(self.Gam, self.E)
        if self.emtp2:
            slope = -np.maximum(Rv / Gv - 1.0, 0.0)      # = -M_e (R_e - Gam_e)_+
        else:
            slope = -self.Me * np.abs(Rv - Gv)
        slope[z > 0.5] = 0.0
        self.cuts.append((fit['F'], slope, z))
        return fit

    def solve_master(self, k, time_limit=60.0):
        m, d = self.m, self.d
        nvar = m + 1                      # z (m), eta
        c = np.zeros(nvar); c[-1] = 1.0
        rows, lo, hi = [], [], []
        # cuts: eta - g z >= f_t - g z_t
        for f_t, g, z_t in self.cuts:
            r = np.zeros(nvar); r[:m] = -g; r[-1] = 1.0
            rows.append(r); lo.append(f_t - g @ z_t); hi.append(np.inf)
        # cardinality
        r = np.zeros(nvar); r[:m] = 1.0; rows.append(r); lo.append(d - 1); hi.append(k)
        # degree >= 1
        for i in range(d):
            r = np.zeros(nvar)
            for t, (a, b) in enumerate(self.E):
                if a == i or b == i: r[t] = 1.0
            rows.append(r); lo.append(1.0); hi.append(np.inf)
        for r in self.cutsets:
            rr = np.zeros(nvar); rr[:m] = r; rows.append(rr); lo.append(1.0); hi.append(np.inf)
        A = np.array(rows)
        integrality = np.zeros(nvar); integrality[:m] = 1
        bounds = Bounds(np.r_[np.zeros(m), self.f_full - 1.0], np.r_[np.ones(m), np.inf])
        res = milp(c, constraints=LinearConstraint(A, lo, hi), integrality=integrality, bounds=bounds,
                   options={'time_limit': time_limit, 'mip_rel_gap': 1e-9})
        if res.x is None:
            raise RuntimeError("master failed: " + str(res.message))
        z = np.round(res.x[:m]).astype(int)
        return z, res.x[-1], res

    def run(self, kmax, kmin=None, max_iter=2000, time_limit=600.0, incumbent_paths=None):
        d = self.d; Gam = self.Gam
        kmin = d - 1 if kmin is None else kmin
        t0 = time.time()
        # k = d-1: MST is exact
        T = mst_edges(Gam)
        fitT = self.add_point(fit_support(Gam, T, emtp2=self.emtp2, M=self.M))
        results = {}
        results[d - 1] = dict(F=fitT['F'], LB=fitT['F'], edges=T, w=fitT['w'], iters=0, time=time.time() - t0, gap=0.0, status='exact (MST)')
        # greedy incumbents
        gpath = greedy_forward(Gam, kmax, start=T, emtp2=self.emtp2, M=self.M)
        for kk, fit in gpath.items():
            self.add_point(local_search(Gam, fit, emtp2=self.emtp2, M=self.M))
        for k in range(max(kmin, d), kmax + 1):
            tk = time.time(); it = 0
            UB = min(f['F'] for key, f in self.visited.items() if len(key) <= k)
            LB = self.f_full
            status = 'optimal'
            while True:
                if time.time() - tk > time_limit or it >= max_iter:
                    status = 'time/iter limit'; break
                z, eta, res = self.solve_master(k)
                edges = [self.E[t] for t in np.where(z > 0)[0]]
                comp = components_of(d, edges)
                if comp.max() > 0:
                    for cidx in range(comp.max() + 1):
                        S = set(np.where(comp == cidx)[0])
                        r = np.array([1.0 if ((a in S) != (b in S)) else 0.0 for (a, b) in self.E])
                        self.cutsets.append(r)
                    continue
                LB = max(LB, eta)
                key = frozenset(edges)
                if key in self.visited:
                    fit = self.visited[key]
                else:
                    fit = self.add_point(fit_support(Gam, edges, emtp2=self.emtp2, M=self.M))
                    fit2 = local_search(Gam, fit, emtp2=self.emtp2, M=self.M)
                    if fit2['F'] < fit['F'] - 1e-12: self.add_point(fit2)
                UB = min(f['F'] for kk, f in self.visited.items() if len(kk) <= k)
                it += 1
                if self.verbose:
                    print(f"k={k} it={it} LB={LB:.6f} UB={UB:.6f} gap={(UB-LB):.2e} cuts={len(self.cuts)} t={time.time()-t0:.1f}s")
                if UB - LB <= self.tol * max(1.0, abs(UB)):
                    break
            best = min(((kk, f) for kk, f in self.visited.items() if len(kk) <= k), key=lambda x: x[1]['F'])[1]
            results[k] = dict(F=UB, LB=LB, edges=sorted(best['edges']), w=best['w'], iters=it,
                              time=time.time() - tk, gap=(UB - LB) / max(1.0, abs(UB)), status=status, ncuts=len(self.cuts))
        self.results = results
        return results

# ----------------------------------------------------------------------------
# baselines
# ----------------------------------------------------------------------------
def eglearn(Gam, rho, method='glasso'):
    """Engelke, Lalancette and Volgushev (2022): graphical lasso on each
    Sigma^(k) built from Gam, majority vote over k."""
    from sklearn.covariance import graphical_lasso
    d = Gam.shape[0]
    votes = np.zeros((d, d)); tot = np.zeros((d, d))
    for k in range(d):
        idx = [i for i in range(d) if i != k]
        S = sigma_k(Gam, k)
        S = 0.5 * (S + S.T)
        try:
            _, P = graphical_lasso(S, alpha=rho, max_iter=200)
        except Exception:
            continue
        A = (np.abs(P) > 1e-8).astype(float); np.fill_diagonal(A, 0)
        for a, i in enumerate(idx):
            for b, j in enumerate(idx):
                if a < b:
                    votes[i, j] += A[a, b]; tot[i, j] += 1
    edges = [(i, j) for i in range(d) for j in range(i + 1, d) if tot[i, j] > 0 and votes[i, j] > tot[i, j] / 2.0]
    return edges

def l1_emtp2(Gam, rho):
    """ell_1-penalised EMTP2 estimator = EMTP2 MLE with variogram Gam + rho."""
    d = Gam.shape[0]
    G2 = Gam + rho * (np.ones((d, d)) - np.eye(d))
    fit = fit_support(G2, edges_of(d), emtp2=True, maxit=500)
    return [e for e, we in zip(fit['edges'], fit['w']) if we > 1e-8]

# ----------------------------------------------------------------------------
# full HR likelihood (Engelke and Hitz 2020) for model comparison
# ----------------------------------------------------------------------------
def loglik_HR(Y, Gam, k=0, maxpts=None):
    n, d = Y.shape
    idx = [i for i in range(d) if i != k]
    S = sigma_k(Gam, k); S = 0.5 * (S + S.T)
    Sinv = np.linalg.inv(S); ld = np.linalg.slogdet(S)[1]
    x = np.log(Y[:, idx] / Y[:, [k]]) + Gam[idx, k] / 2.0
    q = np.einsum('ni,ij,nj->n', x, Sinv, x)
    logdens = -np.log(Y).sum(1) - np.log(Y[:, k]) - 0.5 * (d - 1) * np.log(2 * np.pi) - 0.5 * ld - 0.5 * q
    # normalising constant V(1,...,1) = sum_k Phi_{d-1}(Gam_{-k,k}/2; Sigma^(k))
    V = 0.0
    for kk in range(d):
        ii = [i for i in range(d) if i != kk]
        Skk = sigma_k(Gam, kk); Skk = 0.5 * (Skk + Skk.T)
        mvn = multivariate_normal(mean=np.zeros(d - 1), cov=Skk, allow_singular=True)
        V += mvn.cdf(Gam[ii, kk] / 2.0, maxpts=maxpts) if maxpts else mvn.cdf(Gam[ii, kk] / 2.0)
    return float(logdens.sum() - n * np.log(V))

# ----------------------------------------------------------------------------
# synthetic models
# ----------------------------------------------------------------------------
def random_connected_graph(d, k, rng):
    """random spanning tree (random Pruefer sequence) plus k-(d-1) random extra edges"""
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
    Gam = resistance_from_L(L)[0]
    return E, w, Gam

def f1_score(true_edges, est_edges):
    T = set(map(tuple, true_edges)); S = set(map(tuple, est_edges))
    tp = len(T & S)
    prec = tp / len(S) if S else 0.0; rec = tp / len(T) if T else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1
