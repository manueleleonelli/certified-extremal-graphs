"""Numerical checks of the identities used in the paper (small d, brute force)."""
import numpy as np, itertools, networkx as nx
from scipy.optimize import minimize
rng = np.random.default_rng(1)

def edges_of(d):
    return [(i, j) for i in range(d) for j in range(i + 1, d)]

def laplacian(d, edges, w):
    L = np.zeros((d, d))
    for (i, j), we in zip(edges, w):
        L[i, i] += we; L[j, j] += we; L[i, j] -= we; L[j, i] -= we
    return L

def resistance(L):
    d = L.shape[0]; J = np.ones((d, d)) / d
    G = np.linalg.inv(L + J)
    g = np.diag(G)
    return g[:, None] + g[None, :] - 2 * G

def theta_from_gamma(Gam):
    d = Gam.shape[0]; P = np.eye(d) - np.ones((d, d)) / d
    return np.linalg.pinv(-0.5 * P @ Gam @ P)

d = 6
E = edges_of(d)
# random connected EMTP2 model: random weights on a random connected graph
G = nx.gnp_random_graph(d, 0.5, seed=3)
while not nx.is_connected(G):
    G = nx.gnp_random_graph(d, 0.5, seed=int(rng.integers(1e6)))
Eg = [tuple(sorted(e)) for e in G.edges()]
w = rng.uniform(0.3, 2.0, len(Eg))
Theta = laplacian(d, Eg, w)
J = np.ones((d, d)) / d
Gam = resistance(Theta)

# (1) Gamma is the variogram of Theta: Theta = pinv(-1/2 P Gam P)
print("1 Theta = (-1/2 P Gam P)^+ :", np.allclose(theta_from_gamma(Gam), Theta))
# (2) pseudo-det identities
ev = np.linalg.eigvalsh(Theta); pdet = np.prod(ev[ev > 1e-10])
print("2a pdet = det(Theta+J):", np.isclose(pdet, np.linalg.det(Theta + J)))
print("2b pdet = d*det(Theta_{-k,-k}):", all(np.isclose(pdet, d * np.linalg.det(np.delete(np.delete(Theta, k, 0), k, 1))) for k in range(d)))
# (3) trace identity tr(Gam Theta) = -2(d-1)
print("3 tr(Gam Theta) = -2(d-1):", np.isclose(np.trace(Gam @ Theta), -2 * (d - 1)))
# (4) Gaussian reduction: log pdet Theta + 1/2 tr(Gam Theta) = log d + log det Theta^(k) - tr(Sigma^(k) Theta^(k))
def sigma_k(Gam, k):
    idx = [i for i in range(d) if i != k]
    return 0.5 * (Gam[np.ix_(idx, [k])] + Gam[np.ix_([k], idx)] - Gam[np.ix_(idx, idx)])
ok = True
Gam2 = resistance(laplacian(d, E, rng.uniform(0.2, 1.5, len(E))))  # another "empirical" variogram
for k in range(d):
    idx = [i for i in range(d) if i != k]
    Tk = Theta[np.ix_(idx, idx)]; Sk = sigma_k(Gam2, k)
    lhs = np.log(pdet) + 0.5 * np.trace(Gam2 @ Theta)
    rhs = np.log(d) + np.linalg.slogdet(Tk)[1] - np.trace(Sk @ Tk)
    ok &= np.isclose(lhs, rhs)
print("4 Gaussian reduction for every k:", ok)
# (5) trace as edge sum: 1/2 tr(Gam w-Laplacian) = -sum_e w_e Gam_e
print("5 tr(Gam L(w)) = -2 <Gam,w>:", np.isclose(np.trace(Gam2 @ Theta), -2 * sum(Gam2[i, j] * we for (i, j), we in zip(Eg, w))))
# (6) tree: Theta = L(1/Gam_e), det(Theta+J) = d prod w_e
T = nx.minimum_spanning_tree(nx.Graph([(i, j, {'weight': Gam2[i, j]}) for i, j in E]))
Et = [tuple(sorted(e)) for e in T.edges()]
wt = np.array([1.0 / Gam2[i, j] for i, j in Et])
Lt = laplacian(d, Et, wt)
Gt = resistance(Lt)
print("6a tree resistance equals Gam on tree edges:", all(np.isclose(Gt[i, j], Gam2[i, j]) for i, j in Et))
print("6b tree det(L+J) = d prod w:", np.isclose(np.linalg.det(Lt + J), d * np.prod(wt)))
# (7) completion KKT: on support E', minimiser of -log det(L(w)+J)+<Gam,w> has R_e(w)=Gam_e on E'
def F(wv, edges, Gam):
    L = laplacian(d, edges, wv)
    s, ld = np.linalg.slogdet(L + J)
    if s <= 0: return 1e10
    return -ld + sum(Gam[i, j] * we for (i, j), we in zip(edges, wv))
Esub = Et + [e for e in E if e not in Et][:3]
res = minimize(F, np.array([1 / Gam2[i, j] for i, j in Esub]), args=(Esub, Gam2), method='L-BFGS-B', bounds=[(1e-9, None)] * len(Esub))
Rfit = resistance(laplacian(d, Esub, res.x))
print("7a KKT resistance matching on edges:", all(np.isclose(Rfit[i, j], Gam2[i, j], atol=1e-5) for i, j in Esub))
print("7b Rayleigh: R_e <= Gam_e off the support (EMTP2 truth):", all(Rfit[i, j] <= Gam2[i, j] + 1e-6 for i, j in E if i != j))
# (7c) optimal value = log pdet(-1/2 P Ghat P) + (d-1) - log d, i.e. objective value = -log det(Lhat + J) + (d-1)
Lhat = laplacian(d, Esub, res.x)
print("7c value = -logdet(Lhat+J) + (d-1):", np.isclose(res.fun, -np.linalg.slogdet(Lhat + J)[1] + (d - 1), atol=1e-5))
# (8) EMTP2 bound w_e <= 1/Gam_e at optimum
print("8 w_e <= 1/Gam_e at optimum:", all(res.x[t] <= 1 / Gam2[i, j] + 1e-6 for t, (i, j) in enumerate(Esub)))
# (9) one-edge gain with other weights fixed: -D(rho), D(rho)=log rho -1 + 1/rho
eabs = [e for e in E if e not in Esub][0]
rho = Rfit[eabs] / Gam2[eabs]
wbest = 1 / Gam2[eabs] - 1 / Rfit[eabs]
gain_exact = F(np.append(res.x, wbest), Esub + [eabs], Gam2) - res.fun
print("9 one-edge fixed gain = -(log rho - 1 + 1/rho):", np.isclose(gain_exact, -(np.log(rho) - 1 + 1 / rho)), rho, gain_exact)
# (10) brute-force: at k=d-1 the exact l0 optimum is the MST
best = None
for comb in itertools.combinations(E, d - 1):
    g = nx.Graph(list(comb))
    g.add_nodes_from(range(d))
    if not nx.is_connected(g): continue
    val = sum(np.log(Gam2[i, j]) for i, j in comb)
    if best is None or val < best[0]: best = (val, comb)
print("10 brute force k=d-1 optimum is MST:", set(best[1]) == set(Et))
# (11) brute-force ell0 at k=d and compare with the OA-type cut validity: f(z) >= f(z_t) - sum_{e notin E_t} (rho_e-1)_+ z_e
def fval(edges):
    g = nx.Graph(list(edges)); g.add_nodes_from(range(d))
    if not nx.is_connected(g): return np.inf
    r = minimize(F, np.array([1 / Gam2[i, j] for i, j in edges]), args=(list(edges), Gam2), method='L-BFGS-B', bounds=[(1e-9, None)] * len(edges))
    return r.fun, r.x
k = d
vals = {}
for comb in itertools.combinations(E, k):
    out = fval(comb)
    if out is np.inf or (isinstance(out, float)): continue
    vals[comb] = out
# cut from the MST point
fT, wT = fval(tuple(Et))
RT = resistance(laplacian(d, Et, wT))
viol = 0
for comb, (v, _) in vals.items():
    bound = fT - sum(max(RT[i, j] / Gam2[i, j] - 1, 0) for (i, j) in comb if (i, j) not in Et)
    if v < bound - 1e-6: viol += 1
print("11 OA big-M cut violated by any k=d support:", viol, "of", len(vals))
print("   best k=d value", min(v for v, _ in vals.values()), "MST value", fT)
