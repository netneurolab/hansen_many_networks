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


def exponential(x, a, b, c):
    return a*np.exp(b*x)+c


def regress_dist(x, eu_distance, pars):
    return x - exponential(eu_distance, pars[0], pars[1], pars[2])


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
disease_names = np.load(enigmapath+'data/isorders.npy')

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
    pars[network], _ = curve_fit(exponential, eu_distance[mask], networks[network][mask], p0)
    regressed = regress_dist(networks[network][mask], eu_distance[mask], pars[network])
    networks_reg[network] = np.zeros((nnodes, nnodes))
    networks_reg[network][np.triu(np.ones(nnodes, dtype=bool), 1)] = regressed
    networks_reg[network] = networks_reg[network] + networks_reg[network].T

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
            neighbour_abnormality[i] = np.sum(node_abnormality * net[i, :])/(nnodes)
        
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
np.save(path+'results/nn_corrs_networks.npy', nn_corrs)

# plot heatmap
fig, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(nn_corrs[:, :, 0], annot=True, linewidths=.5,
            cmap=cmap_div, vmin=-1, vmax=1,
            xticklabels=disease_names, yticklabels=networks.keys())
plt.tight_layout()
plt.savefig(path+'figures/heatmap_disease_nncorr.svg')
