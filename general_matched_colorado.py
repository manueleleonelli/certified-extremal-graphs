"""Radius-one certified search for the general model of the Colorado data at the size of the EMTP2
solution (181 edges), started from the general path at that size; Figure 1(c) and Table 4 show it.
Writes out['general']['matched'] in results/colorado_p0.99.json.
Usage: python general_matched_colorado.py [budget in seconds, default 1 hour]"""
import os, sys, json
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np
from hrl0 import forward_path, fit_support, SwapCertifier
from experiments import dump, OUT

budget = float(sys.argv[1]) if len(sys.argv) > 1 else 3600
Gb = np.load(os.path.join(HERE, 'colorado_Gamma_p0.99.npy'))
fn = 'colorado_p0.99.json'; out = json.load(open(os.path.join(OUT, fn)))
k = out['bic_k']
ckpt = os.path.join(OUT, 'colorado_pathg_ckpt_p0.99.json')     # path of the general model, if already computed
saved = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
start = [tuple(e) for e in saved[str(k)]] if str(k) in saved else forward_path(Gb, k, signed=True)[k].edges
F_path = fit_support(Gb, start, signed=True).F
res = SwapCertifier(Gb, start, signed=True).certify(r_max=1, time_limit=budget)
fit = fit_support(Gb, res['edges'], signed=True)
out['general']['matched'] = dict(k=k, F_path=F_path, F=res['F'], radius=res['radius'], timed_out=res['timed_out'],
                                 time=res['time'], refits=res['stats']['refits'], rlog=res['rlog'],
                                 n_negative=int((fit.w < 0).sum()), edges=[list(map(int, e)) for e in res['edges']])
print('k', k, 'path', F_path, 'certified', res['F'], 'radius', res['radius'], 'negative', int((fit.w < 0).sum()),
      'time', round(res['time']), flush=True)
dump(fn, out)
