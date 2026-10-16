"""Radius-two certified search for the EMTP2 solution of the Colorado data at its BIC size (181 edges),
started from the radius-one certified graph written by `experiments.py colorado`.  Writes
results/colorado_r2_<k>.json and, once radius two is certified, replaces the certified graph in
results/colorado_p0.99.json, which Figure 1, Table 4 and the validation analyses read.
Usage: python colorado_radius2.py [budget in seconds, default 2 hours]"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from hrl0 import SwapCertifier
from experiments import dump, OUT

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 2 * 3600
Gb = np.load(os.path.join(HERE, 'colorado_Gamma_p0.99.npy'))
fn = 'colorado_p0.99.json'; out = json.load(open(os.path.join(OUT, fn)))
kb = str(out['bic_k']); c = out['cert'][kb]
start = c.get('edges_radius1', c['edges']); F1 = c.get('F_radius1', c['F'])
cert = SwapCertifier(Gb, [tuple(e) for e in start])
res = cert.certify(r_max=2, time_limit=budget)
edges = [list(map(int, e)) for e in res['edges']]
print('radius', res['radius'], 'F', res['F'], 'timed out', res['timed_out'], 'time', round(res['time']), flush=True)
dump(f'colorado_r2_{kb}.json', dict(rlog=res['rlog'], radius=res['radius'], F=res['F'], edges=edges,
                                    refits=res['stats']['refits'], time=res['time']))
if res['radius'] >= 2:
    c['edges_radius1'], c['F_radius1'] = start, F1
    c['per_r'] = [q for q in c['per_r'] if q['r'] == 1] + [
        dict(r=2, radius=res['radius'], time=res['time'], refits=res['stats']['refits'], F=res['F'],
             improved=bool(res['F'] < F1 - 1e-7), timed_out=res['timed_out'], start='radius-1 certified graph')]
    c['edges'], c['F'] = edges, res['F']
    dump(fn, out)
