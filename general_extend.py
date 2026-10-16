"""Extend the paths of the general model for the applications until the BIC minimum is interior,
then certify the general solution at its BIC size as before (Colorado: radius one; Danube: radii one and two)."""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from hrl0 import *
from experiments import dump, select_bic_path, OUT, _certify_path

def extend(Gb, n, ckpt, k0, step=40, margin=30, kcap=600):
    kmax = k0
    while True:
        t = time.time()
        path = forward_path(Gb, kmax, signed=True, checkpoint=ckpt)
        items = [(k, f.F, f.edges) for k, f in sorted(path.items())]
        b, kb, Fb, Eb = select_bic_path(items, n)
        print('kmax', kmax, 'reached', max(path), 'bic_k', kb, 'step time', round(time.time() - t), flush=True)
        if kb <= max(path) - margin or max(path) < kmax or kmax >= kcap:
            return path, kb, Fb, Eb
        kmax += step

which = sys.argv[1]
if which == 'colorado':
    p = 0.99
    Gb = np.load(os.path.join(HERE, f'colorado_Gamma_p{p}.npy')); Yc = np.load(os.path.join(HERE, f'colorado_Y_p{p}.npy'))
    n, d = Yc.shape
    fn = f'colorado_p{p}.json'; out = json.load(open(os.path.join(OUT, fn)))
    t0 = time.time()
    path, kb, Fb, Eb = extend(Gb, n, os.path.join(OUT, f'colorado_pathg_ckpt_p{p}.json'), 220)
    g = out['general']
    g['old'] = dict(bic_k=g['bic_k'], bic_F=g['bic_F'], cert=g['cert'], t_path=g['t_path'])
    g['path'] = [dict(k=k, F=f.F, bic=bic(f.F, k, n)) for k, f in sorted(path.items())]
    g['kmax'] = max(path); g['t_path_ext'] = time.time() - t0
    g['bic_k'] = kb; g['bic_F'] = Fb; g['bic_edges'] = list(Eb); g['cert'] = {}
    dump(fn, out)
    per_r, res = _certify_path(Gb, path, kb, [1], 1200, signed=True)
    g['cert'][str(kb)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'])
    dump(fn, out)
    print('colorado general done', kb, res['F'], flush=True)
elif which == 'danube':
    from experiments_v2 import edges_list
    X = np.load(os.path.join(HERE, 'danube_clustered.npy'))
    flow = np.load(os.path.join(HERE, 'danube_flow_edges.npy'))
    flow_edges = sorted((int(min(a, b)) - 1, int(max(a, b)) - 1) for a, b in flow)
    Y = data2mpareto(X, 0.9); n = Y.shape[0]; Gb = emp_vario(Y)
    fn = 'danube_new.json'; out = json.load(open(os.path.join(OUT, fn)))
    t0 = time.time()
    path, kb, Fb, Eb = extend(Gb, n, os.path.join(OUT, 'danube_pathg_ckpt.json'), 110, step=30, margin=20, kcap=465)
    g = out['general']
    g['old'] = dict(bic_k=g['bic_k'], bic_F=g['bic_F'], cert=g['cert'], t_path=g['t_path'])
    g['path'] = [dict(k=k, F=f.F, bic=bic(f.F, k, n), f1_flow=f1_score(flow_edges, f.edges)[2]) for k, f in sorted(path.items())]
    g['kmax'] = max(path); g['t_path_ext'] = time.time() - t0
    g['bic_k'] = kb; g['bic_F'] = Fb; g['bic_edges'] = list(Eb); g['bic_f1_flow'] = f1_score(flow_edges, Eb)[2]; g['cert'] = {}
    dump(fn, out)
    per_r, res = _certify_path(Gb, path, kb, [1, 2], 1800, signed=True)
    g['cert'][str(kb)] = dict(per_r=per_r, edges=list(res['edges']), F=res['F'], f1_flow=f1_score(flow_edges, res['edges'])[2])
    g['F_emtp2_of_general_graph'] = profile_value(Gb, res['edges'])
    dump(fn, out)
    print('danube general done', kb, res['F'], flush=True)
