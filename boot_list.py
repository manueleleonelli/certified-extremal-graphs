"""Run the Colorado bootstrap replicates given on the command line (see colorado_extra.boot)."""
import sys, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_v] = '1'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from colorado_extra import boot
for rep in map(int, sys.argv[1:]):
    boot(rep, rep + 1)
