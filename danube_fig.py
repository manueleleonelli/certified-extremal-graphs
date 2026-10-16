"""Figure EC.1 of the online supplement: estimated graphs for the upper Danube basin (writes figures/fig_danube.pdf)."""
import json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__))
out=json.load(open(os.path.join(HERE,'results','danube_new.json')))
info=pd.read_csv(os.path.join(HERE,'danube_info.csv'))
x=info['PlotCoordX'].values; y=info['PlotCoordY'].values
flow=set(tuple(e) for e in out['flow_edges']); d=out['d']
def draw(ax, edges, title):
    est=set(tuple(e) for e in edges)
    for (i,j) in flow-est: ax.plot([x[i],x[j]],[y[i],y[j]],color='0.75',lw=1.2,ls=':',zorder=1)
    for (i,j) in est-flow: ax.plot([x[i],x[j]],[y[i],y[j]],color='#b2182b',lw=1.0,ls='--',zorder=2)
    for (i,j) in est&flow: ax.plot([x[i],x[j]],[y[i],y[j]],color='black',lw=1.5,zorder=3)
    ax.scatter(x,y,s=48,color='white',edgecolor='black',linewidth=0.8,zorder=4)
    for i in range(d): ax.text(x[i],y[i],str(i+1),fontsize=5,ha='center',va='center',zorder=5)
    ax.set_title(title,fontsize=9); ax.set_axis_off(); ax.set_aspect('equal')
kb=out['bic_k']; c=out['cert'][str(kb)]; eg=out['competitors']['eg']
fig,axes=plt.subplots(1,3,figsize=(10.5,3.9))
draw(axes[0],out['mst_edges'],f"Minimum spanning tree\n$k=30$, $F_1={out['mst_f1_flow']:.2f}$")
draw(axes[1],c['edges'],f"Certified EMTP$_2$ solution, BIC\n$k={kb}$, $F_1={c['f1_flow']:.2f}$")
draw(axes[2],eg['sparsest_edges'],f"EGlearn, sparsest connected\n$k={eg['sparsest_k']}$, $F_1={eg['sparsest_f1']:.2f}$")
plt.tight_layout(w_pad=0.5)
os.makedirs(os.path.join(HERE,'figures'),exist_ok=True)
fig.savefig(os.path.join(HERE,'figures','fig_danube.pdf'),bbox_inches='tight')
