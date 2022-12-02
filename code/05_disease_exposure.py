"""
Figure 5: disease exposure

Author: Justine Y Hansen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from netneurotools import datasets, stats
from scipy.spatial.distance import squareform, pdist
from scipy.stats import zscore, spearmanr
from scipy.optimize import curve_fit
from statsmodels.stats.multitest import multipletests


def exponential(x, a, b, c):
    return a*np.exp(b*x)+c


def regress_dist(x, eu_distance, pars):
    return x - exponential(eu_distance, pars[0], pars[1], pars[2])


def corr_spin(x, y, spins, nspins):
    rho, _ = spearmanr(x, y, nan_policy='omit')
    null = np.zeros((nspins,))

    # null correlation
    for i in range(nspins):
        null[i], _ = spearmanr(x, y[spins[:, i]], nan_policy='omit')

    pval = (1 + sum(abs((null - np.mean(null))) >
                    abs((rho - np.mean(null))))) / (nspins + 1)
    return rho, pval


"""
set-up
"""

path = '/home/jhansen/gitrepos/hansen_many_networks/'

coords = np.genfromtxt(path+'data/parcellation_files/Cammoun033_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)

# get cortical indices
info = pd.read_csv(datasets.fetch_cammoun2012()['info'])
hemiid = np.array(info.query('scale == "scale033" & structure == "cortex"')['hemisphere']) == 'R'
nnull = 1000
spins033 = stats.gen_spinsamples(coords, hemiid,
                                 n_rotate=nnull, seed=1234)
eu_distance = squareform(pdist(coords, metric="euclidean"))
mask = np.triu(np.ones(nnodes), 1) > 0

# load enigma cortical abnormalities
# from https://github.com/netneurolab/hansen_crossdisorder_vulnerability
enigmapath = '/home/jhansen/gitrepos/hansen_crossdisorder_vulnerability/'
disease_profiles = zscore(np.genfromtxt(enigmapath+'data/enigma_ct.csv', delimiter=','))
disease_names = np.load(enigmapath+'data/disorders.npy')

# load networks
gc = np.load(path+'data/Cammoun033/gene_coexpression.npy')
rs = np.load(path+'data/Cammoun033/receptor_similarity.npy')
ls = np.load(path+'data/Cammoun033/laminar_similarity.npy')
mc = np.load(path+'data/Cammoun033/metabolic_connectivity.npy')
hc = np.load(path+'data/Cammoun033/haemodynamic_connectivity.npy')
ec = np.load(path+'data/Cammoun033/electrophysiological_connectivity.npy')
ts = np.load(path+'data/Cammoun033/temporal_similarity.npy')

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

# distance regressed version, for supplement
p0 = [1, -0.05, -0.1]  # initial parameter guess
pars = {}  # exponential curve variables
networks_reg = {}  # distance-regressed similarity matrices (upper triangular)
for network in networks.keys():
    pars[network], _ = curve_fit(exponential, eu_distance[mask],
                                 networks[network][mask], p0,
                                 bounds=([0, -10, -5], [10, 0, 5]))
    regressed = regress_dist(networks[network][mask], eu_distance[mask], pars[network])
    networks_reg[network] = np.zeros((nnodes, nnodes))
    networks_reg[network][np.triu(np.ones(nnodes, dtype=bool), 1)] = regressed
    networks_reg[network] = networks_reg[network] + networks_reg[network].T

# load SC
sc_wei = np.load(path+'data/Cammoun033/consensusSC_wei.npy')

# load colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
plt.rcParams['svg.fonttype'] = 'none'

"""
exposure analysis
"""

nn_corrs = np.zeros((len(networks.keys()), len(disease_names), 2)) # rho, pval

# plot each correlation
fig, axs = plt.subplots(len(networks.keys()), len(disease_names),
                        sharex=True, figsize=(30, 13))
for d, disease in enumerate(disease_names):
    print(disease)
    for n, network in enumerate(networks.keys()):
        node_abnormality = disease_profiles[:, d]
        neighbour_abnormality = np.zeros((nnodes, ))
        net = networks[network].copy()
        net[net < 0] = 0
        for i in range(nnodes):  # for each node
            neighbour_abnormality[i] = np.sum(node_abnormality * net[i, :])/(sum(net[i, :] > 0))
        
        axs[n, d].scatter(node_abnormality, neighbour_abnormality, s=2)
        if n == len(networks.keys())-1:
            axs[n, d].set_xlabel(disease)
        if d == 0:
            axs[n, d].set_ylabel(network)
        axs[n, d].set_aspect(1.0/axs[n, d].get_data_ratio(), adjustable='box')
        # nn_corrs[n, d, 0] = spearmanr(node_abnormality, neighbour_abnormality)[0]
        nn_corrs[n, d, 0], nn_corrs[n, d, 1] = \
            corr_spin(node_abnormality, neighbour_abnormality, spins033, nnull)
# plt.tight_layout()
plt.savefig(path+'figures/Cammoun033/scatter_nncorr.eps')
np.save(path+'results/nn_corrs_networks.npy', nn_corrs)
nn_corrs[:, :, 1] = np.array([multipletests(nn_corrs[:, d, 1], method='fdr_bh')[1] \
                                for d in range(nn_corrs.shape[1])]).T

# plot heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(nn_corrs[:, :, 0], annot=True, linewidths=.5,
            cmap=cmap_div, vmin=-1, vmax=1,
            xticklabels=disease_names, yticklabels=networks.keys())
plt.tight_layout()
plt.savefig(path+'figures/Cammoun033/heatmap_disease_nncorr.svg')

# structure only
nn_corrs_sc = np.zeros((len(disease_names), 2))  # rho, pval
fig, axs = plt.subplots(1, len(disease_names), figsize=(30, 2))
for d, disease in enumerate(disease_names):
    print(disease)
    node_abnormality = disease_profiles[:, d]
    neighbour_abnormality = np.zeros((nnodes, ))
    for i in range(nnodes):
        neighbour_abnormality[i] = np.mean(node_abnormality[sc_wei[i, :] != 0]
                                           * sc_wei[sc_wei[i, :] != 0, i])
    axs[d].scatter(node_abnormality, neighbour_abnormality, s=2)
    axs[d].set_xlabel(disease)
    if d == 0:
        axs[d].set_ylabel('sc exposure')
    axs[d].set_aspect(1.0/axs[d].get_data_ratio(), adjustable='box')
    nn_corrs_sc[d, 0], nn_corrs_sc[d, 1] = \
        corr_spin(node_abnormality, neighbour_abnormality, spins033, nnull)
nn_corrs_sc[:, 1] = multipletests(nn_corrs_sc[:, 1], method='fdr_bh')[1]

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(nn_corrs_sc[:, 0].reshape(1, -1), annot=True, linewidths=.5,
            cmap=cmap_div, vmin=-1, vmax=1, square=True,
            xticklabels=disease_names, yticklabels=['sc'])
plt.tight_layout()
plt.savefig(path+'figures/Cammoun033/heatmap_disease_nncorr_sc.svg')

# distance only
nn_corrs_dist = np.zeros((len(disease_names), 2)) # rho, pval
eu_dist_rec = 1/eu_distance
eu_dist_rec[np.eye(nnodes, dtype=bool)] = 0
fig, axs = plt.subplots(1, len(disease_names), figsize=(30, 13))
for d, disease in enumerate(disease_names):
    print(disease)
    node_abnormality = disease_profiles[:, d]
    neighbour_abnormality = np.zeros((nnodes, ))
    for i in range(nnodes):  # for each node
        neighbour_abnormality[i] = np.sum(node_abnormality * eu_dist_rec[i, :])/(nnodes - 1)
    
    axs[d].scatter(node_abnormality, neighbour_abnormality, s=2)
    if d == 0:
        axs[d].set_ylabel('distance')
    axs[d].set_aspect(1.0/axs[d].get_data_ratio(), adjustable='box')
    nn_corrs_dist[d, 0], nn_corrs_dist[d, 1] = \
        corr_spin(node_abnormality, neighbour_abnormality, spins033, nnull)
nn_corrs_dist[:, 1] = multipletests(nn_corrs_dist[:, 1], method='fdr_bh')[1]

fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(nn_corrs_dist[:, 0].reshape(1, -1), annot=True, linewidths=.5,
            cmap=cmap_div, vmin=-1, vmax=1, square=True,
            xticklabels=disease_names, yticklabels=['dist'])
plt.tight_layout()
plt.savefig(path+'figures/Cammoun033/heatmap_disease_nncorr_dist.svg')

"""
toy figure
"""

from matplotlib import cm

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


def scale_values(values, vmin, vmax, axis=None):
    s = (values - values.min(axis=axis)) / (values.max(axis=axis) - values.min(axis=axis))
    s = s * (vmax - vmin)
    s = s + vmin
    return s


fig, ax = plt.subplots(figsize=(5, 5))
net =  np.zeros(networks['gc'].shape)
node = 23 # 45 # 15 # 23
net[:, node] = networks['gc'][:, node]
edges = np.where(net > 0)
edge_colours = get_color_distribution(net[edges],
                                      vmin=np.min(net[edges])-np.std(net[edges]),
                                      cmap='Blues')
linewidths = scale_values(net[edges], 0.3, 1.2)
idx = np.argsort(linewidths)  # plot in order of edge strength
for edge_i, edge_j, c, w in zip(edges[0][idx], edges[1][idx], edge_colours[idx, :], linewidths[idx]):
    x1 = coords[edge_i, 0]
    x2 = coords[edge_j, 0]
    y1 = coords[edge_i, 1]
    y2 = coords[edge_j, 1]
    ax.plot([x1, x2], [y1, y2], c=c, linewidth=w, alpha=1, zorder=0)
ax.scatter(coords[:, 0], coords[:, 1], c='k', clip_on=False, alpha=1,
           s=scale_values(disease_profiles[:, 1], 3, 10)**2.15,
           linewidths=0, zorder=1)
ax.set_aspect('equal')
plt.savefig(path+'figures/Cammoun033/plot_disease_exposure_toy2_v2.eps')
