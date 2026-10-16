"""Figure 1 of the IJOC paper: three maps side by side (minimum spanning tree, certified EMTP2
solution at the BIC size, certified general solution with the same number of edges)."""
import os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures'); os.makedirs(OUT, exist_ok=True)

def main(p=0.99):
    out = json.load(open(os.path.join(HERE, 'results', f'colorado_p{p}.json')))
    meta = pd.read_csv(os.path.join(HERE, 'coprcp_meta.csv'))
    el = np.load(os.path.join(HERE, 'coelev.npz'))
    lon = meta['lon'].values; lat = meta['lat'].values
    d = out['d']; kb = out['bic_k']
    E_bic = out['cert'][str(kb)]['edges']
    gen = out['general']
    if 'matched' in gen:                 # general model at the size of the EMTP2 solution
        kg = gen['matched']['k']; E_gen = gen['matched']['edges']; lab = 'same size'
    else:
        kg = gen['bic_k']; E_gen = gen['cert'][str(kg)]['edges'] if str(kg) in gen.get('cert', {}) else gen['bic_edges']; lab = 'BIC size'

    shared = set(tuple(e) for e in E_bic)

    def draw(ax, edges, title, mark=None):
        ax.imshow(el['z'].T, origin='lower', extent=[el['x'].min(), el['x'].max(), el['y'].min(), el['y'].max()],
                  cmap='Greys', aspect='auto', alpha=0.55, vmin=1000, vmax=4300)
        for (i, j) in [tuple(e) for e in edges]:
            red = mark is not None and (i, j) not in mark
            ax.plot([lon[i], lon[j]], [lat[i], lat[j]], color='#b2182b' if red else 'black', lw=1.3 if red else 0.9, alpha=0.9, zorder=2)
        ax.scatter(lon, lat, s=26, c='white', edgecolor='black', lw=0.6, zorder=3)
        ax.set_title(title, fontsize=12.5); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-106.05, -103.95); ax.set_ylim(36.95, 41.05)

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 5.0))
    draw(axes[0], out['mst_edges'], f"(a) minimum spanning tree, $k={d-1}$")
    draw(axes[1], E_bic, f"(b) EMTP$_2$ model, BIC size $k={kb}$")
    draw(axes[2], E_gen, f"(c) general model, {lab} $k={kg}$", mark=shared)
    plt.tight_layout(w_pad=1.0)
    fig.savefig(os.path.join(OUT, 'fig_colorado.pdf'))
    print('edges of (c) not in (b):', sum(1 for e in E_gen if tuple(e) not in shared))

if __name__ == '__main__':
    main()
