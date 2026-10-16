"""
Post-hoc quantities on the data sets of the exactness experiments, regenerated from
their seeds: (a) the path without two-in-one-out moves against the enumeration
optimum; (b) in the general model, the EMTP2 enumeration optimum at the same size.
Usage: python posthoc_v2.py twoin|gain
"""
import os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
import sys, json, glob, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hrl0 import *
from enum_batch import enumerate_min
import experiments_v2 as X

OUTP = os.path.join(X.OUT, 'posthoc')
os.makedirs(OUTP, exist_ok=True)


def data(exp, key, rep):
    d, ktrue, n = key
    rng = X.rng_for(exp, key, rep)
    if exp == 'exact_signed':
        E, w, Gam = random_signed_hr_model(d, ktrue, rng, frac_pos=0.3)
    else:
        E, w, Gam = random_hr_model(d, ktrue, rng)
    return E, emp_vario(rmpareto_HR(n, Gam, rng))


def twoin():
    out = []
    for fn in sorted(glob.glob(os.path.join(X.OUT, 'exact', '*.json'))):
        r = json.load(open(fn)); key = (r['d'], r['ktrue'], r['n'])
        E, Gb = data('exact', key, r['rep'])
        kmax = max(p['k'] for p in r['problems'])
        path = forward_path(Gb, kmax, two_in=False)
        # check that the data were regenerated exactly
        p0 = r['problems'][0]
        assert abs(fit_support(Gb, [tuple(e) for e in p0['brute_edges']]).F - p0['brute']) < 1e-8
        for p in r['problems']:
            out.append(dict(d=r['d'], ktrue=r['ktrue'], n=r['n'], rep=r['rep'], k=p['k'], brute=p['brute'],
                            path1=path[p['k']].F, miss=bool(path[p['k']].F > p['brute'] + 1e-7)))
        print(fn, sum(o['miss'] for o in out), flush=True)
    json.dump(out, open(os.path.join(OUTP, 'exact_twoin.json'), 'w'))


def gain(shard=0, nshard=1):
    od = os.path.join(OUTP, 'gain'); os.makedirs(od, exist_ok=True)
    files = sorted(glob.glob(os.path.join(X.OUT, 'exact_signed', '*.json')))
    for t, fn in enumerate(files):
        if t % nshard != shard:
            continue
        of = os.path.join(od, os.path.basename(fn))
        if os.path.exists(of):
            continue
        r = json.load(open(fn)); key = (r['d'], r['ktrue'], r['n'])
        E, Gb = data('exact_signed', key, r['rep'])
        p0 = r['problems'][0]
        assert abs(fit_support(Gb, [tuple(e) for e in p0['brute_edges']], signed=True).F - p0['brute']) < 1e-8
        out = []
        for p in r['problems']:
            en = enumerate_min(Gb, p['k'], signed=False)
            out.append(dict(d=r['d'], ktrue=r['ktrue'], n=r['n'], rep=r['rep'], k=p['k'], brute=p['brute'],
                            brute_emtp2=en['F'], gain=en['F'] - p['brute']))
        json.dump(out, open(of, 'w'))
        print(fn, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'twoin':
        twoin()
    else:
        gain(int(sys.argv[2]), int(sys.argv[3]))
