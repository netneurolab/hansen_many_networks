"""
Figure 3: hubs

Author: Justine Y Hansen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import rankdata, spearmanr
from palettable.cartocolors.qualitative import Safe_7
from netneurotools import plotting, stats, datasets
from neuromaps.datasets import fetch_annotation
from neuromaps.transforms import fslr_to_fslr
from neuromaps.parcellate import Parcellater
import mayavi


def scale_values(values, vmin, vmax, axis=None):
    s = (values - values.min(axis=axis)) / (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin
    return s


def get_color_distribution(scores, cmap="viridis", vmin=None, vmax=None):

    '''
    Function to get a color for individual values of a distribution of scores.
    Copied from VinceBaz/projects_networks/projects_networks/colors.py
    '''

    n = len(scores)

    if vmin is None:
        vmin = np.amin(scores)
    if vmax is None:
        vmax = np.amax(scores)

    cmap = cm.get_cmap(cmap, 256)
    new_colors = cmap(np.linspace(0, 1, 256))

    if vmin != vmax:
        scaled = (scores - vmin)/(vmax - vmin) * 255
        scaled[scaled < 0] = 0
        scaled[scaled > 255] = 255
    else:
        scaled = np.zeros((n)) + 128

    c = np.zeros((n, 4))
    for i in range(n):
        c[i] = new_colors[int(scaled[i]), :]

    return c


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


def corr_spin(x, y, spins, nspins):
    rho, _ = spearmanr(x, y)
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = spearmanr(x, y[spins[:, i]])

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
    return rho, pval



"""
set-up
"""

path = '/home/jhansen/projects/proj_many_networks/'
parc = 'Schaefer400'

coords = np.genfromtxt(path+'data/parcellation_files/' + parc + '_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)
mask = np.triu(np.ones(nnodes), 1) > 0

# get spins
nnull = 1000
hemiid = np.zeros((nnodes, ))
hemiid[:int(nnodes/2)] = 1
spins = stats.gen_spinsamples(coords, hemiid, n_rotate=nnull, seed=1234)

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

# rank transform
netrank = dict([])
for network in networks.keys():
    edgerank = rankdata(networks[network][mask])
    netrank[network] = np.zeros((nnodes, nnodes))
    netrank[network][np.triu(np.ones(nnodes, dtype=bool), 1)] = edgerank
    netrank[network] = netrank[network] + netrank[network].T

plt.ion()

"""
plot strongest edges
"""

for network in networks.keys():

    fig, ax = plt.subplots(figsize=(5, 5))

    vec = networks[network][mask]
    net =  np.zeros(networks[network].shape)
    thresh = np.flipud(np.sort(vec))[int(np.floor(0.005 * len(vec)))]
    net[networks[network] >= thresh] = networks[network][networks[network] >= thresh]
    edges = np.where(net != 0)
    edge_colours = get_color_distribution(net[edges],
                                          vmin=np.min(net[edges])-3*np.std(net[edges]),
                                          cmap='Greys')
    linewidths = scale_values(net[edges], 0.3, 1.2)
    idx = np.argsort(linewidths)  # plot in order of edge strength
    for edge_i, edge_j, c, w in zip(edges[0][idx], edges[1][idx], edge_colours[idx, :], linewidths[idx]):
        x1 = coords[edge_i, 0]
        x2 = coords[edge_j, 0]
        y1 = coords[edge_i, 1]
        y2 = coords[edge_j, 1]
        ax.plot([x1, x2], [y1, y2], c=c, linewidth=w, alpha=1, zorder=0)
    ax.scatter(coords[:, 0], coords[:, 1], c='k', clip_on=False, alpha=1,
               s=scale_values(np.sum(networks[network], axis=1), 2, 10)**2.15,
               linewidths=0, zorder=1)
    ax.set_aspect('equal')
    ax.set_title(network)
    plt.savefig(path+'figures/'+parc+'/plot_strongest_edges_' + network + '.eps')


"""
plot regional hubs
"""

for network in netrank.keys():
    brains = plotting.plot_conte69(np.sum(netrank[network], axis=1),
                                   lhlabels, rhlabels,
                                   surf='inflated', colormap='YlGnBu',
                                   colorbar=False)
    save_conte69(brains, path+'figures/' + parc + '/surface_strength/' + network)


"""
RSN and VE classification
"""

if parc == "Cammoun033":
    info = datasets.fetch_cammoun2012()['info']
    rsnlabels = list(info.query('scale == "scale033" ^ structure == "cortex"')['yeo_7'])
elif parc == "Schaefer400" or parc == "Schaefer100":
    labelinfo = np.loadtxt(path+'data/parcellation_files/' + parc + '_7Networks_order_info.txt')
    rsnlabels = []
    for row in range(0, len(labelinfo), 2):
        rsnlabels.append(labelinfo[row].split('_')[2])

velabels = np.genfromtxt(path+'data/' + parc + '/voneconomo_' + parc + '.csv', delimiter=',')
ve_name = ['primary motor', 'association 1', 'association 2',
           'primary/secondary sensory', 'primary sensory', 'limbic', 'insular']

# identify within/between partition edges
rsn_assign = np.zeros((nnodes, nnodes))
ve_assign = np.zeros((nnodes, nnodes))
for i in range(nnodes):
    for j in range(nnodes):
        if rsnlabels[i] == rsnlabels[j]:
            rsn_assign[i, j] = 1
        if velabels[i] == velabels[j]:
            ve_assign[i, j] = 1

both_assign = np.logical_and(rsn_assign, ve_assign)
densities = range(5, 55, 5)

rsn_overlap = dict([])
ve_overlap = dict([])
both_overlap = dict([])
for network in networks.keys():
    rsn_overlap[network] = np.zeros((len(densities), ))
    ve_overlap[network] = np.zeros((len(densities), ))
    both_overlap[network] = np.zeros((len(densities), ))
    for i in range(len(densities)):
        vec = networks[network][mask]
        d = densities[i]
        thresh = np.flipud(np.sort(vec))[int(np.floor(d * 0.001 * len(vec)))]
        binary_mat = (networks[network] > thresh).astype(int)
        rsn_overlap[network][i] = np.sum(np.logical_and(binary_mat[mask],
                                                        rsn_assign[mask])) \
                                   / np.sum(binary_mat[mask]) * 100
        ve_overlap[network][i] = np.sum(np.logical_and(binary_mat[mask],
                                                       ve_assign[mask])) \
                                   / np.sum(binary_mat[mask]) * 100
        both_overlap[network][i] = np.sum(np.logical_and(binary_mat[mask],
                                                         both_assign[mask])) \
                                   / np.sum(binary_mat[mask]) * 100

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
for i, network in enumerate(networks.keys()):
    ax[0].plot(densities, rsn_overlap[network], c=Safe_7.mpl_colors[i])
    ax[1].plot(densities, ve_overlap[network], c=Safe_7.mpl_colors[i])
ax[0].legend(networks.keys())
ax[0].set_xticklabels([densities[i]/10 for i in range(len(densities))])
ax[0].set_xticks(densities)
ax[0].set_xlabel('Density (%)')
ax[0].set_ylabel('Intra-class edge overlap (%)')
ax[0].set_title('RSN')

ax[1].set_xticklabels([densities[i]/10 for i in range(len(densities))])
ax[1].set_xticks(densities)
ax[1].set_xlabel('Density (%)')
ax[1].set_ylabel('Intra-class edge overlap (%)')
ax[1].set_title('VE')
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/plot_edge_overlap.eps')

"""
cross-modality network
"""

# plot cross-modality edge ranks
fig, ax = plt.subplots(figsize=(5, 5))

edgerank = np.sum(np.array([netrank[network] for network in netrank.keys()]), axis=0)
thresh = np.flipud(np.sort(edgerank[mask]))[int(np.floor(0.005 * len(edgerank[mask])))]
edges = np.where(edgerank >= thresh)
edge_colours = get_color_distribution(edgerank[edges],
                                      vmin=np.min(edgerank[edges]) - 3*np.std(edgerank[edges]),
                                      cmap='Greys')
linewidths = scale_values(edgerank[edges], 0.3, 1.2)
idx = np.argsort(linewidths)  # plot in order of edge strength
for edge_i, edge_j, c, w in zip(edges[0][idx], edges[1][idx], edge_colours[idx, :], linewidths[idx]):
    x1 = coords[edge_i, 0]
    x2 = coords[edge_j, 0]
    y1 = coords[edge_i, 1]
    y2 = coords[edge_j, 1]
    ax.plot([x1, x2], [y1, y2], c=c, linewidth=w, alpha=1, zorder=0)
ax.scatter(coords[:, 0], coords[:, 1], c='k', clip_on=False, alpha=1,
           s=scale_values(np.median(edgerank, axis=1), 2, 10)**2.15,
           linewidths=0, zorder=1)
ax.set_aspect('equal')
ax.set_title('strongest ranked edges')
plt.savefig(path+'figures/'+parc+'/plot_strongest_edges_ranked.eps')

# cross-modality strength map
netrankmed = np.median(np.array([np.sum(netrank[network], axis=1) for network in networks.keys()]), axis=0)
brains = plotting.plot_conte69(netrankmed, lhlabels, rhlabels,
                               surf='inflated', colormap='YlGnBu',
                               colorbar=False)
save_conte69(brains, path+'figures/'+parc+'surface_strength/medianrank')

# compare with evolutionary expansion
evoexp = fetch_annotation(source='hill2010', desc='evoexp')
evoexp = fslr_to_fslr(evoexp, target_density='32k', hemi='R')
parcellater = Parcellater((lhlabels, rhlabels), 'fslr')
evoexp_parc = parcellater.fit_transform(evoexp, 'fslr', hemi='R')
if parc == 'Cammoun033':
    evoexp_parc = np.delete(evoexp_parc, 3)  # remove corpus callosum
r, p = corr_spin(evoexp_parc, netrankmed[int(nnodes/2):], spins[:int(nnodes/2), :], nnull)
