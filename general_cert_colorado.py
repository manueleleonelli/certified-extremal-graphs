"""Radius-one certified search for the general-model solution of the Colorado data at its BIC size,
continuing from the incumbent improved in the first 20 minutes, with a long budget."""
import os, sys, json, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from hrl0 import *
from experiments import dump, OUT
budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3 * 3600
Gb = np.load(os.path.join(HERE, 'colorado_Gamma_p0.99.npy'))
fn = 'colorado_p0.99.json'; out = json.load(open(os.path.join(OUT, fn)))
g = out['general']; kb = g['bic_k']; c0 = g['cert'][str(kb)]
cert = SwapCertifier(Gb, [tuple(e) for e in c0['edges']], signed=True)
t0 = time.time(); res = cert.certify(r_max=1, time_limit=budget)
print('done', res['radius'], res['F'], res['timed_out'], time.time() - t0, flush=True)
out = json.load(open(os.path.join(OUT, fn))); g = out['general']
c = g['cert'][str(kb)]
c['continued'] = dict(radius=res['radius'], F=res['F'], timed_out=res['timed_out'], time=time.time() - t0,
                      refits=res['stats']['refits'], rlog=res['rlog'], edges=[list(map(int, e)) for e in res['edges']])
if not res['timed_out']:                 # the certified graph replaces the incumbent of the first 20 minutes
    c.setdefault('first_20min', dict(edges=c['edges'], F=c['F'], per_r=c['per_r']))
    c['edges'], c['F'], c['radius'] = c['continued']['edges'], res['F'], res['radius']
    c['total_time'] = c['first_20min']['per_r'][-1]['time'] + c['continued']['time']
dump(fn, out)
