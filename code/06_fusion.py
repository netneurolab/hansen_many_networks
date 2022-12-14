"""
Figure 6: Network fusion

Author: Justine Y Hansen
"""


import numpy as np
import pandas as pd
import seaborn as sns
import snf, itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr
from netneurotools import datasets
from palettable.cartocolors.qualitative import Safe_7


"""
set-up
"""

path = '/home/jhansen/gitrepos/hansen_many_networks/'
parc = 'Schaefer400'

coords = np.genfromtxt(path+'data/parcellation_files/' + parc + '_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)
mask = np.triu(np.ones(nnodes), 1) > 0

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

# load SC
sc = np.load(path+'data/' + parc + '/consensusSC.npy')
sc_wei = np.load(path+'data/' + parc + '/consensusSC_wei.npy')

# colourmaps
cmap = np.genfromtxt(path+'data/colourmap.csv', delimiter=',')
cmap_div = ListedColormap(cmap)
cmap_seq = ListedColormap(cmap[128:, :])
cmap_blue = ListedColormap(np.flip(cmap[:128, :], axis=0))

plt.ion()

"""
similarity network fusion
"""

affin = snf.compute.make_affinity(list(networks.values()),
                                  K=np.ceil(nnodes/10).astype(int),
                                  mu=0.5)
fused = snf.compute.snf(affin,
                        K=np.ceil(nnodes/10).astype(int),
                        t=20, alpha=1)

fused[np.eye(nnodes, dtype=bool)] = 0
np.save(path+'results/fused_network.npy', fused)

# heatmap
plt.figure()
sns.heatmap(fused, cmap=cmap_seq,
            square=True, rasterized=True, vmax=0.005)
plt.savefig(path+'figures/'+parc+'/heatmap_fused.eps')

# distance
eu_distance = squareform(pdist(coords, metric="euclidean"))
h = sns.jointplot(x=eu_distance[mask], y=fused[mask], kind='hex',
                  palette=cmap_blue, rasterized=True)
h.set_axis_labels('Euclidean distance', 'fused network', fontsize=8)
plt.savefig(path+'figures/'+parc+'/hexplot_fused.eps')

"""
structure
"""

plt.figure()
plt.scatter(sc_wei[mask][sc_wei[mask] != 0],
              fused[mask][sc_wei[mask] != 0],
              s=2, alpha=0.5, edgecolor=None,
              c='#62acdd')
plt.xlabel('sc weight')
plt.ylabel('edge weight')
plt.savefig(path+'figures/'+parc+'/scatter_fused_sc.eps')

# connected vs not
nnull = 1000
sc_rew = np.load(path+'results/sc_betzelnull_'+parc+'.npy')

fig, axs = plt.subplots()
d = dict({'connected' : fused[mask][np.where(sc[mask] != 0)],
          'not connected' : fused[mask][np.where(sc[mask] == 0)]})
d = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
emp = np.mean(d['connected']) - np.mean(d['not connected'])
null = np.zeros((nnull, ))
for k in range(nnull):
    null[k] = np.mean(fused[mask]
                      [np.where(sc_rew[:, :, k][mask] == 1)]) \
              - np.mean(fused[mask]
                        [np.where(sc_rew[:, :, k][mask] == 0)])
p = (1 + np.sum(np.abs((null - np.mean(null)))
                >= abs((emp - np.mean(null))))) / (nnull + 1)
sns.violinplot(data=d)
plt.savefig(path+'figures/'+parc+'/violin_fused_connectedvsnot.eps')

"""
RNS and VE classification
"""

# all of this is just the same as in 03_hubs.py
if parc == "Cammoun033":
    info = pd.read_csv(datasets.fetch_cammoun2012()['info'])
    rsnlabels = list(info.query('scale == "scale033" & structure == "cortex"')['yeo_7'])
elif parc == "Schaefer400" or parc == "Schaefer100":
    labelinfo = np.loadtxt(path+'data/parcellation_files/' + parc + '_7Networks_order_info.txt',
                           dtype='str', delimiter='tab')
    rsnlabels = []
    for row in range(0, len(labelinfo), 2):
        rsnlabels.append(labelinfo[row].split('_')[2])

velabels = np.genfromtxt(path+'data/' + parc + '/voneconomo_' + parc + '.csv', delimiter=',')
ve_name = ['primary motor', 'association 1', 'association 2',
           'primary/secondary sensory', 'primary sensory', 'limbic', 'insular']

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

# ok this is the fused-specific stuff
rsn_edge_overlap_fused = np.zeros((len(densities), ))
ve_edge_overlap_fused = np.zeros((len(densities), ))
both_edge_overlap_fused = np.zeros((len(densities), ))
for i in range(len(densities)):
    vec = fused[mask]
    d = densities[i]
    thresh = np.flipud(np.sort(vec))[int(np.floor(d * 0.001 * len(vec)))]
    binary_mat = (fused > thresh).astype(int)
    rsn_edge_overlap_fused[i] = np.sum(np.logical_and(binary_mat[mask],
                                                      rsn_assign[mask])) \
                                / np.sum(binary_mat[mask]) * 100
    ve_edge_overlap_fused[i] = np.sum(np.logical_and(binary_mat[mask],
                                                     ve_assign[mask])) \
                               / np.sum(binary_mat[mask]) * 100
    both_edge_overlap_fused[i] = np.sum(np.logical_and(binary_mat[mask],
                                                       both_assign[mask])) \
                                 / np.sum(binary_mat[mask]) * 100

fig, ax = plt.subplots()
i=0
for network in networks.keys():
    ax.plot(densities, rsn_overlap[network],
            c=Safe_7.mpl_colors[i],
            alpha=0.5, linewidth=1, label=network)
    ax.plot(densities, ve_overlap[network],
            c=Safe_7.mpl_colors[i],
            alpha=0.5, linewidth=1,
            linestyle='--')
    i += 1
ax.plot(densities, rsn_edge_overlap_fused, c='k',
        linewidth=2, label='RSN')
ax.plot(densities, ve_edge_overlap_fused, c='k',
        linewidth=2, linestyle='--', label='VE')   
ax.legend()
ax.set_xticklabels([densities[i]/10 for i in range(len(densities))])
ax.set_xticks(densities)
ax.set_xlabel('Density (%)')
ax.set_ylabel('Intra-class edge overlap (%)')
plt.savefig(path+'figures/'+parc+'/plot_intraclassedges_fused.eps')

fig, ax = plt.subplots()
for i, network in enumerate(networks.keys()):
    ax.plot(densities, both_overlap[network],
            c=Safe_7.mpl_colors[i],
            alpha=0.5, linewidth=1, label=network)
ax.plot(densities, both_edge_overlap_fused, c='k',
        linewidth=2, label='fused')
ax.legend()
ax.set_xticklabels([densities[i]/10 for i in range(len(densities))])
ax.set_xticks(densities)
ax.set_xlabel('Density (%)')
ax.set_ylabel('Intra-class edge overlap (%)')
plt.savefig(path+'figures/'+parc+'/plot_intraclassedges_both.eps')

"""
sensitivity/robustness
"""

# different combinations of networks
corrs = []
sets = []
for nnet in range(2, len(networks.keys())):
    print(nnet)
    for subset in itertools.combinations(networks.keys(), nnet):
        sets.append(subset)
        nets = [networks[net] for net in subset]
        affin_subset = snf.compute.make_affinity(nets, K=40, mu=0.5)
        fused_subset = snf.compute.snf(affin_subset, K=40, t=20, alpha=1)
        corrs.append(spearmanr(fused[mask], fused_subset[mask])[0])
sorted_idx = np.argsort(corrs)

plt.figure(figsize=(20, 8))
plt.bar(range(len(corrs)), np.array(corrs)[sorted_idx],
              tick_label=np.array(sets, dtype=object)[sorted_idx])
plt.ylim([0.55, 1])
plt.xticks(rotation=90)
plt.ylabel('correlation to complete fused network')
plt.tight_layout()
plt.savefig(path+'figures/bar_fusion_subsets.png')

# parameter search
K = np.arange(20, 60, 1)
MU = np.arange(3, 9, 1)/10
paramsearch = np.zeros((len(networks['gc'][mask]), len(K), len(MU)))

for ki, k in enumerate(K):
    for mi, mu in enumerate(MU):
        print('K = ' + str(k) + ', mu = ' + str(mu))
        affin = snf.compute.make_affinity(list(networks.values()), K=k, mu=mu)
        fused = snf.compute.snf(affin, K=k, t=20, alpha=1)
        paramsearch[:, ki, mi] = fused[mask]
np.save(path+'results/paramsearch.npy', paramsearch)

paramsearch_corrs = np.zeros((paramsearch.shape[1], paramsearch.shape[2]))
for i in range(paramsearch.shape[1]):
    for j in range(paramsearch.shape[2]):
        paramsearch_corrs[i, j] = spearmanr(paramsearch[:, i, j], fused[mask])[0]

fig, ax = plt.subplots(figsize=(15, 3))
sns.heatmap(paramsearch_corrs.T, square=True,
            cmap=cmap_seq, linewidths=.5, xticklabels=K,
            yticklabels=MU, ax=ax)
ax.set_ylabel('mu')
ax.set_xlabel('K')
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/heatmap_fused_paramsearch.eps')