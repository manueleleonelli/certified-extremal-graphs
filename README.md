# certified-extremal-graphs

Code and data for the paper

> Manuele Leonelli. *Cardinality-constrained structure learning for extremal graphical models.* Submitted to INFORMS Journal on Computing, 2026.

The paper studies maximum likelihood estimation of a Hüsler–Reiss graphical model with a bound on the number of edges. The repository contains a Python implementation of the two algorithms of the paper:

- the **solution path** (Algorithm 1), which starts from the minimum spanning tree and gives a candidate graph for every number of edges, and
- the **certified swap search** (Algorithm 2), which proves that no graph within a given number of edge swaps of the incumbent has a better profile likelihood, using the pinning, completed and clipped dual bounds of Section 4.3.

Both work under extremal total positivity (EMTP2, where the precision matrix is a weighted graph Laplacian) and in the general model (`signed=True`). The same code applies to Laplacian-constrained Gaussian graphical models, with the sample variogram of the Gaussian data, the mean squared differences between components, in place of the empirical extremal variogram.

The repository also contains every script, data set and raw result needed to reproduce the tables and figures of the paper and of its online supplement.

[![Open the demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/manueleleonelli/certified-extremal-graphs/blob/main/demo.ipynb)

## Installation

The code needs Python 3.10 or later.

```bash
git clone https://github.com/manueleleonelli/certified-extremal-graphs.git
cd certified-extremal-graphs
pip install -r requirements.txt
```

`gurobipy` is needed only for the mixed-integer baseline of Section 6.2 (`mio_gurobi.py`). The size-limited license that comes with `pip install gurobipy` is enough for the instances of the paper.

**Use single-threaded linear algebra.** The search performs many small dense factorizations, for which a multithreaded BLAS is up to ten times slower. Set

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

before running anything. Several scripts set these variables themselves. Run independent jobs as separate processes to use more cores.

## Quick start

```python
import numpy as np
from hrl0 import data2mpareto, emp_vario, forward_path, SwapCertifier, bic

X = np.load('danube_clustered.npy')          # upper Danube river flows, 31 stations
Y = data2mpareto(X, p=0.9)                    # multivariate Pareto sample above the 0.9 quantile
Gb = emp_vario(Y)                             # empirical variogram
n = Y.shape[0]

path = forward_path(Gb, kmax=60)              # one graph for every k from 30 to 60
k = min(path, key=lambda k: bic(path[k].F, k, n))

cert = SwapCertifier(Gb, path[k].edges)
out = cert.certify(r_max=2, time_limit=600)   # certified over all double swaps
print(k, out['F'], out['radius'], out['edges'])
```

`fit_support(Gb, edges)` returns the profile likelihood of a fixed graph and its fitted conductances, effective resistances and precision matrix. Pass `signed=True` to `forward_path`, `fit_support` and `SwapCertifier` for the general model. The notebook `demo.ipynb` runs this analysis on the Danube data, with the general model and two competitors, in a few minutes.

## Files

### Methods

| File | Content |
| --- | --- |
| `hrl0.py` | The methods of the paper: model utilities, simulation, empirical variogram, `fit_support`, `forward_path` (Algorithm 1), `SwapCertifier` (Algorithm 2), exhaustive enumeration for small graphs, competitors (EMTP2 maximum likelihood, l1-penalized EMTP2, EGlearn, extremal graphical lasso), BIC and F1 score. |
| `hrsparse.py` | Outer approximation with the big-M formulation and a HiGHS master problem, used as a baseline in Section 6.2. |
| `enum_batch.py` | Exhaustive enumeration of connected supports with vectorized fits, independent of the certified search, used to check exactness in Section 6.1. |
| `mio_gurobi.py` | Single-tree branch-and-cut with Gurobi for the big-M and perspective formulations (Section 6.2). |
| `verify_identities.py` | Numerical checks of the identities and bounds of Sections 3 and 4 on small graphs. |
| `demo.ipynb` | Demo on the Danube data. |

### Experiments and summaries

| File | Content |
| --- | --- |
| `experiments_v2.py` | Simulations of Section 6, organized in independent units (experiment, configuration, replication), each with its own seed. |
| `posthoc_v2.py` | Additional quantities on the data sets of Section 6.1, regenerated from their seeds. |
| `make_tables_v2.py` | Tables 1 to 3, the supplementary simulation tables and the summary numbers quoted in Section 6. |
| `experiments.py` | Danube and Colorado analyses (`danube_new`, `colorado`). Its other options are an earlier simulation protocol and are not used in the paper. |
| `general_extend.py` | Extends the path of the general model until the BIC minimum is interior, then certifies the selected graph (Colorado and Danube). |
| `general_cert_colorado.py` | Continues the radius-one certification of the general model on the Colorado data with a longer budget. |
| `general_matched_colorado.py` | Radius-one certification of the general model on the Colorado data at the size of the EMTP2 solution (Figure 1(c), Table 4). |
| `colorado_radius2.py` | Radius-two certification of the Colorado graph at its BIC size. |
| `prep_colorado.py` | Builds the Colorado events and variograms from the daily precipitation data. |
| `colorado_extra.py`, `boot_list.py` | Colorado validation: cross-validation by odd and even years, other thresholds and the bootstrap. |
| `colorado_post.py` | Summaries of the validation and Figure 2. |
| `colorado_fig.py`, `colorado_table.py` | Figure 1 and Table 4. |
| `make_tables.py`, `danube_fig.py` | Danube table and Figure EC.1 of the supplement. |

### Data

| File | Content |
| --- | --- |
| `danube_clustered.npy`, `danube_info.csv`, `danube_flow_edges.npy` | Declustered river flows at 31 stations of the upper Danube basin, station information and the flow graph, from the R package `graphicalExtremes` (data of Asadi, Davison and Engelke, 2015). |
| `coprcp_matrix.pkl`, `coprcp_meta.csv`, `coelev.npz` | Daily precipitation at 64 Colorado stations, April to October 1990 to 2019, station coordinates and elevations, and an elevation grid, from the R package `evgam` (data sets `COprcp` and `COelev`). |
| `colorado_Y_p*.npy`, `colorado_Gamma_p*.npy` | Event maxima and empirical variograms at the thresholds 0.95, 0.975 and 0.99, written by `prep_colorado.py`. |

### Results

`results/` holds the raw output of every run reported in the paper, so that all tables and figures can be rebuilt without rerunning the experiments. `results/v2/<experiment>/` holds one JSON file per simulation unit. The `figures/` folder is created by the scripts.

## Reproducing the paper

All commands are run from the root of the repository with single-threaded BLAS. The table and figure scripts read `results/` and take seconds. The experiments overwrite the corresponding files in `results/`.

### Tables and figures from the stored results

```bash
python make_tables_v2.py all     # Tables 1-3, EC.1-EC.3 (results/v2/tables/) and the numbers of Section 6
python colorado_fig.py           # Figure 1   -> figures/fig_colorado.pdf
python colorado_post.py all      # Figure 2   -> figures/fig_colorado_validation.pdf, and its numbers
python colorado_table.py         # Table 4    -> figures/tab_colorado.tex
python make_tables.py danube     # Table EC.4 -> figures/tab_danube.tex
python danube_fig.py             # Figure EC.1 -> figures/fig_danube.pdf
```

| Paper | Experiment | Summary |
| --- | --- | --- |
| Table 1, Section 6.1 | `experiments_v2.py worker exact`, `posthoc_v2.py twoin` | `make_tables_v2.py exact` |
| Table EC.1 | `experiments_v2.py worker exact_signed`, `posthoc_v2.py gain` | `make_tables_v2.py exact_signed` |
| Section 6.2 | `experiments_v2.py worker oa`, `mio_gurobi.py` | `make_tables_v2.py oa`, `make_tables_v2.py mio` |
| Table 2, Table EC.2 | `experiments_v2.py worker recovery recovery_w2 recovery_w4` | `make_tables_v2.py recovery`, `recovery_w2`, `recovery_w4` |
| Table 3 | `experiments_v2.py worker certify` | `make_tables_v2.py certify` |
| Table EC.3 | `experiments_v2.py worker certify_dense` | `make_tables_v2.py certify_dense` |
| Section 7, Figure 1, Table 4 | Colorado pipeline below | `colorado_fig.py`, `colorado_table.py` |
| Figure 2, Section 7.3 | `colorado_extra.py`, `boot_list.py` | `colorado_post.py all` |
| Table EC.4, Figure EC.1 | `experiments.py danube_new`, `general_extend.py danube` | `make_tables.py danube`, `danube_fig.py` |

### Simulations (Section 6 and supplement)

Each unit of `experiments_v2.py` writes `results/v2/<experiment>/<unit>.json` and is skipped if that file exists, so delete the folder of an experiment to rerun it. A worker claims units through lock files, so several workers can run in parallel on one machine:

```bash
python experiments_v2.py list                       # units done and total per experiment
for i in $(seq 8); do
  python experiments_v2.py worker exact exact_signed oa certify certify_dense recovery recovery_w2 recovery_w4 &
done; wait
python posthoc_v2.py twoin                          # path without two-in-one-out moves (Section 6.1)
python posthoc_v2.py gain 0 1                       # general against EMTP2 optimum (Table EC.1); shard i of m: gain i m
for g in 0 0.1; do for d in 8 10; do python mio_gurobi.py $g $d; done; done     # Gurobi baseline, big-M and perspective
```

Budgets are set in the scripts: 120 seconds for outer approximation and Gurobi, and five or ten minutes per certification in Table 3.

### Colorado application (Section 7)

```bash
python prep_colorado.py 0.99; python prep_colorado.py 0.975; python prep_colorado.py 0.95
python experiments.py colorado 0.99     # path to 200 edges, BIC, radius-one certificate, competitors
python general_extend.py colorado       # path of the general model to its BIC minimum (440 edges)
python general_cert_colorado.py         # radius one for the general model at 440 edges (about three hours)
python general_matched_colorado.py      # radius one for the general model at 181 edges
python colorado_radius2.py              # radius two at 181 edges (about 50 minutes)
python colorado_extra.py holdout        # cross-validation by odd and even years
python colorado_extra.py thresholds     # thresholds 0.95 and 0.975
python boot_list.py $(seq 0 49)         # 50 bootstrap replicates (split the list over processes)
```

then the summary scripts listed above. The times are for one core.

### Upper Danube (online supplement)

```bash
python experiments.py danube_new
python general_extend.py danube
python make_tables.py danube; python danube_fig.py
```

## Citation

```bibtex
@article{leonelli2026cardinality,
  author  = {Leonelli, Manuele},
  title   = {Cardinality-Constrained Structure Learning for Extremal Graphical Models},
  journal = {Submitted to INFORMS Journal on Computing},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see `LICENSE`). The data sets come from the R packages `graphicalExtremes` and `evgam` and keep the terms of their sources.
