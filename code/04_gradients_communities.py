"""
Figure 4: gradients and communities

Author: Justine Y Hansen
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from scipy.stats import zscore
from netneurotools import plotting
from palettable.cartocolors.qualitative import Safe_7
import mayavi


def save_conte69(brains, outpath):
    mayavi.mlab.figure(brains[0]).scene.parallel_projection = True
    mayavi.mlab.figure(brains[1]).scene.parallel_projection = True
    mayavi.mlab.figure(brains[0]).scene.background = (1, 1, 1)
    mayavi.mlab.figure(brains[1]).scene.background = (1, 1, 1)

    mayavi.mlab.savefig(outpath + '_lhlateral.png', figure=brains[0])
    mayavi.mlab.savefig(outpath + '_rhlateral.png', figure=brains[1])

    # medial view
    mayavi.mlab.view(azimuth=0, elevation=90, distance=450, figure=brains[0])
    mayavi.mlab.view(azimuth=0, elevation=-90, distance=450, figure=brains[1])

    mayavi.mlab.savefig(outpath + '_lhmedial.png', figure=brains[0])
    mayavi.mlab.savefig(outpath + '_rhmedial.png', figure=brains[1])


"""
set-up
"""

path = '/home/jhansen/projects/proj_many_networks/'
parc = 'Schaefer400'

coords = np.genfromtxt(path+'data/parcellation_files/' + parc + '_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)
mask = np.triu(np.ones(nnodes), 1) > 0

# labels for plotting
lhlabels = path+'data/parcellation_files/' + parc + '_order_hemi-L.label.gii'
rhlabels = path+'data/parcellation_files/' + parc + '_order_hemi-R.label.gii'

# load networks
gc = np.load(path+'data/' + parc + '/gene_coexpression.npy')
rs = np.load(path+'data/' + parc + '/receptor_similarity.npy')
ls = np.load(path+'data/' + parc + '/laminar_similarity.npy')
mc = np.load(path+'data/' + parc + '/metabolic_connectivity.npy')
hc = np.load(path+'data/' + parc + '/haemodynamic_connectivity.npy')
ec = np.load(path+'data/' + parc + '/electrophysiological_connectivity.npy')
ts = np.load(path+'data/' + parc + '/temporal_similarity.npy')

networks = {"gc" : gc,
            "rs" : rs,
            "ls" : ls,
            "mc" : mc,
            "hc" : hc,
            "ec" : ec,
            "ts" : ts}

# normalize networks
for network in networks.keys():
    networks[network] = np.arctanh(networks[network])
    networks[network][np.eye(nnodes).astype(bool)] = 0

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)

plt.ion()
plt.rcParams['svg.fonttype'] = 'none'

"""
gradient plots
"""

pca = PCA(n_components=1)

for network in networks.keys():
    pc1 = pca.fit_transform(zscore(networks[network]))  # remove zscore?
    brains = plotting.plot_conte69(pc1,
                                   lhlabels, rhlabels,
                                   surf='inflated', colormap='YlGnBu',
                                   colorbar=False)
    save_conte69(brains, path+'figures/' + parc + '/surface_gradients/pc1/' + network)


"""
scree plot
"""

# scree plot
pca = PCA(n_components=5)
lines = []
fig, ax = plt.subplots()
for i, network in enumerate(networks.keys()):
    pca.fit(zscore(networks[network]))
    lines.append(ax.plot(pca.explained_variance_ratio_,
                         color=Safe_7.mpl_colors[i],
                         linewidth=.5, label=network)[0])
    ax.scatter(np.arange(len(pca.explained_variance_ratio_)),
               pca.explained_variance_ratio_,
               edgecolors=None, color=Safe_7.mpl_colors[i])
ax.legend(handles=lines)
ax.set_xlabel('principal component')
ax.set_ylabel('variance explained')
plt.tight_layout()
plt.savefig(path+'figures/' + parc + '/plot_scree.eps')


"""
gradient correlation matrix
"""

pc1 = np.array([pca.fit_transform(zscore(networks[network])) for network in networks.keys()])
sns.heatmap(np.corrcoef(np.squeeze(pc1)),
            square=True, cmap=cmap_div,
            vmin=-1, vmax=1, linewidths=.5,
            xticklabels=networks.keys(),
            yticklabels=networks.keys(),
            cbar=True, annot=True)
plt.tight_layout()
plt.savefig(path+'figures/' + parc + '/heatmap_network_corr.svg')


"""
community detection (# of modules)
"""

# run community detection using: scpt_community_detection.py

gamma_range = [x/10.0 for x in range(1, 61, 1)]
commun = dict([])

for network in networks.keys():
    commun[network] = np.load(path+'results/community_detection_' + parc + '/community_assignments_' + network + '.npy')

fig, ax = plt.subplots(figsize=(7, 4))
for i, network in enumerate(networks.keys()):
    ax.plot(gamma_range, np.max(commun[network], axis=0),
            c=Safe_7.mpl_colors[i], label=network)
ax.legend()
ax.set_xlabel('gamma parameter')
ax.set_ylabel('number of communities')
# plt.savefig(path+'figures/' + parc + '/plot_ncommun_gamma.eps')
ax.set_ylim([0, 40])
plt.savefig(path+'figures/eps/plot_ncommun_gamma_zoom.eps')

"""
community assignment brains
"""

g = {'gc' : 2.0,
     'rs' : 2.0,
     'mp' : 2.5,
     'fp' : 0.5,
     'fc' : 1.2,  
     'mc' : 1.0,
     'ts' : 0.7}

for network in commun.keys():
    idx = np.where(np.array(gamma_range) == g[network])[0][0]
    brains = plotting.plot_conte69(commun[network][:, idx],
                                   lhlabels, rhlabels,
                                   surf='inflated', colormap='Set2',
                                   vmin=1, vmax=8,
                                   colorbar=False)
    save_conte69(brains, path+'figures/' + parc + '/surface_communities/'+network)
