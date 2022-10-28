"""
Figure 2: Distance + SC + rich club links

Author: Justine Y Hansen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from scipy.stats import rankdata, ttest_ind
from scipy.spatial.distance import squareform, pdist
from palettable.cartocolors.qualitative import Safe_7
from bct import rich_club_bu, randmio_und, degrees_und


def equalObs(x, nbin):
    nlen = len(x)
    return np.interp(np.linspace(0, nlen, nbin + 1),
                     np.arange(nlen),
                     np.sort(x))


def get_perm_p(emp, null):
    return (1 + sum(abs(null - np.mean(null))
                    > abs(emp - np.mean(null)))) / (len(null) + 1)


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


def rich_feeder_peripheral(x, sc, max_k=None, stat='median'):
    """
    Calculates connectivity values in rich, feeder, and peripheral edges.
    Parameters
    ----------
    x : (N, N) numpy.ndarray
        Symmetric correlation or connectivity matrix
    sc : (N, N) numpy.ndarray
        Binary structural connectivity matrix
    stat : {'mean', 'median'}, optional
        Statistic to use over rich/feeder/peripheral links. Default: 'median'
    
    Returns
    -------
    rfp : (3, k) numpy.ndarray
        Array of median rich (0), feeder (1), and peripheral (2)
        values, defined by `x`. `k` is the maximum degree defined on `sc`.
    pvals : (3, k) numpy.ndarray
        p-value for each link, computed using Welch's t-test.
        Rich links are compared against non-rich links. Feeder links are
        compared against peripheral links. Peripheral links are compared
        against feeder links. T-test is one-sided.
    
    Author
    ------
    This code was written by Justine Hansen who promises to fix and even
    optimize the code should any issues arise, provided you let her know.
    """

    stats = ['mean', 'median']
    if stat not in stats:
        raise ValueError(f'Provided stat {stat} not valid.\
                         Must be one of {stats}')

    nnodes = len(sc)
    mask = np.triu(np.ones(nnodes), 1) > 0
    node_degree = degrees_und(sc)
    if max_k is None:
        k = np.max(node_degree).astype(np.int64)
    else:
        k = max_k
    rfp_label = np.zeros((len(sc[mask]), k))

    for degthresh in range(k):  # for each degree threshold
        hub_idx = np.where(node_degree >= degthresh)  # find the hubs
        hub = np.zeros([nnodes, 1])
        hub[hub_idx, :] = 1

        rfp = np.zeros([nnodes, nnodes])      # for each link, define rfp
        for edge1 in range(nnodes):
            for edge2 in range(nnodes):
                if hub[edge1] + hub[edge2] == 2:
                    rfp[edge1, edge2] = 1  # rich
                if hub[edge1] + hub[edge2] == 1:
                    rfp[edge1, edge2] = 2  # feeder
                if hub[edge1] + hub[edge2] == 0:
                    rfp[edge1, edge2] = 3  # peripheral
        rfp_label[:, degthresh] = rfp[mask]

    rfp = np.zeros([3, k])
    pvals = np.zeros([3, k])
    for degthresh in range(k):

        redfunc = np.median if stat == 'median' else np.mean
        for linktype in range(3):
            rfp[linktype, degthresh] = redfunc(x[mask][rfp_label[:, degthresh]
                                                       == linktype + 1])

        # p-value (one-sided Welch's t-test)
        _, pvals[0, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 1],
            x[mask][rfp_label[:, degthresh] != 1],
            equal_var=False, alternative='greater')
        _, pvals[1, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 2],
            x[mask][rfp_label[:, degthresh] == 3],
            equal_var=False, alternative='greater')
        _, pvals[2, degthresh] = ttest_ind(
            x[mask][rfp_label[:, degthresh] == 3],
            x[mask][rfp_label[:, degthresh] == 2],
            equal_var=False, alternative='greater')

    return rfp, pvals


"""
set-up
"""

path = '/home/jhansen/gitrepos/hansen_many_networks/'
parc = 'Schaefer400'

coords = np.genfromtxt(path+'data/parcellation_files/' + parc + '_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)
mask = np.triu(np.ones(nnodes), 1) > 0
nnull = 1000

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

# load SC
sc = np.load(path+'data/' + parc + '/consensusSC.npy')
sc_wei = np.load(path+'data/' + parc + '/consensusSC_wei.npy')

plt.ion()


"""
binning distance
"""

eu_distance = squareform(pdist(coords, metric="euclidean"))

scidx = np.where(sc[mask] == 1)[0]
nbins = 50

n, bin_edges, _ = plt.hist(eu_distance[mask],
                           equalObs(eu_distance[mask], nbins),
                           edgecolor='black')
bins = np.digitize(eu_distance[mask], bin_edges[:-1])

fig, ax = plt.subplots(figsize=(len(netrank.keys()), 4))
run_null = False

if run_null:
    null = np.zeros((len(netrank.keys()), nbins, nnull))
    pvals = np.zeros((len(netrank.keys()), nbins))
    for i in range(nnull):
        print(i)
        nullvec = np.random.RandomState(seed=i).permutation(range(len(eu_distance[mask])))
        bin_idx = np.split(nullvec, nbins)
        for j in range(len(bin_idx)):
            null[:, j, i] = np.array([np.median(netrank[network][mask][bin_idx[j]]) for network in netrank.keys()])
    np.save(path+'results/null_binned_distances_'+parc+'.npy', null)

for n, network in enumerate(networks.keys()):
    b = np.array([np.median(netrank[network][mask][bins==i]) for i in range(1, np.max(bins)+1, 1)])
    if run_null:
        pvals[n, :] = np.array([get_perm_p(b[i], null[n, i, :]) for i in range(nbins)])
        sig = np.where(pvals[n, :] < 0.05)[0]
        nsig = np.where(pvals[n, :] >= 0.05)[0]
        ax.scatter(sig, b[sig], linewidths=0.2, edgecolors='k', s=30,
                   color=Safe_7.mpl_colors[n], label=network)
        ax.scatter(nsig, b[nsig], linewidths=0, s=30,
                   color=Safe_7.mpl_colors[n])
    else:
        ax.scatter(range(nbins), b, linewidths=0, s=30,
                   color=Safe_7.mpl_colors[n], label=network)
    ax.set_ylabel('median edge rank')
    ax.set_xlabel('--> greater distance -->')

plt.legend()
plt.savefig(path+'figures/' + parc + '/scatter_binned_distance.eps')


"""
SC-supported connections
"""

fig, ax = plt.subplots()
sns.kdeplot(data=[netrank[network][mask][np.where(sc[mask] == 1)[0]] for network in netrank.keys()],
            ax=ax, palette=Safe_7.mpl_colors)  # legend is backwards???
ax.set_xlabel('edge rank')
ax.legend(netrank.keys())
plt.savefig(path+'figures/'+parc+'/kdeplot_edgerank.eps')

sc_rew = np.load(path+'results/sc_betzelnull_'+parc+'.npy')
null = np.zeros((len(netrank.keys()), nnull))
for i in range(nnull):
    null[:, i] = np.array([np.mean(netrank[network][mask][np.where(sc_rew[:, :, i][mask] == 1)[0]]) for network in netrank.keys()])
emp = np.array([np.mean(netrank[network][mask][np.where(sc[mask] == 1)[0]]) for network in netrank.keys()])
pval = []
for i in range(len(netrank.keys())):
    pval.append(get_perm_p(emp[i], null[i, :]))

"""
SC rich club
"""

# find rich club coefficient
rc_coef = rich_club_bu(sc)[0]
randcoef = np.zeros((len(rc_coef), nnull))

for i in range(nnull):
    randnet = randmio_und(sc, 10)[0]
    randcoef[:, i] = rich_club_bu(randnet)[0]
    print(i)

ratio_coef = rc_coef / np.mean(randcoef, axis=1)
pval = np.zeros((len(rc_coef), ))
for i in range(len(rc_coef)):
    pval[i] = (np.sum(randcoef[i, :] >= rc_coef[i]) + 1) / (nnull + 1)

plt.figure()
plt.plot(range(len(rc_coef)), ratio_coef)
plt.scatter(np.where(pval < 0.05)[0],
            ratio_coef[np.where(pval < 0.05)[0]],
            marker='o')
plt.xlabel('degree')
plt.ylabel('rich club coefficient ratio')
plt.xlim([5, 50])
plt.ylim([0.9, 2.0])
plt.savefig(path+'figures/'+parc+'/plot_richclubcoef.eps')

# plot rich links
node_degree = degrees_und(sc)
degthreshold = 37  # arbitrary, comes from plot above. change as necessary.
hub_idx = np.where(node_degree >= degthreshold)
hub = np.zeros([nnodes, 1])
hub[hub_idx, :] = 1
rfplabel = np.zeros([nnodes, nnodes])
for edge1 in range(nnodes):
    for edge2 in range(nnodes):
        if hub[edge1] + hub[edge2] == 2:
            rfplabel[edge1, edge2] = 1  # rich
        if hub[edge1] + hub[edge2] == 1:
            rfplabel[edge1, edge2] = 2  # feeder
        if hub[edge1] + hub[edge2] == 0:
            rfplabel[edge1, edge2] = 3  # peripheral

fig, ax = plt.subplots(figsize=(5, 5))

edges = np.where(rfplabel == 1)
edge_colours = get_color_distribution(sc_wei[edges],
                                      vmin=np.min(sc_wei[edges]) - 3*np.std(sc_wei[edges]),
                                      cmap='Greys')
linewidths = scale_values(sc_wei[edges], 0.3, 1.2)
idx = np.argsort(linewidths)  # plot in order of edge strength
for edge_i, edge_j, c, w in zip(edges[0][idx], edges[1][idx], edge_colours[idx, :], linewidths[idx]):
    x1 = coords[edge_i, 0]
    x2 = coords[edge_j, 0]
    y1 = coords[edge_i, 1]
    y2 = coords[edge_j, 1]
    ax.plot([x1, x2], [y1, y2], c=c, linewidth=w, alpha=1, zorder=0)
ax.scatter(coords[:, 0], coords[:, 1], c='k', clip_on=False, alpha=1,
           s=scale_values(np.sum(sc, axis=1), 2, 10)**2.15,
           linewidths=0, zorder=1)
ax.set_aspect('equal')
ax.set_title('rich club')
plt.savefig(path+'figures/eps/plot_rich_club_.eps')

"""
rich-club connections
"""

rfp = {}
pvals = {}

lines = []
fig, ax = plt.subplots()
for i, network in enumerate(netrank.keys()):
    rfp[network], pvals[network] = rich_feeder_peripheral(netrank[network], sc, 50)
    lines.append(ax.plot(rfp[network][0, :], color=Safe_7.mpl_colors[i], linewidth=.5, label=network)[0])
    pidx = np.where(pvals[network][0, :] < 0.05)
    ax.scatter(pidx, rfp[network][0, pidx],
               s=20, linewidths=None, color=Safe_7.mpl_colors[i])
ax.legend(handles=lines)
ax.set_xlabel('degree threshold')
ax.set_ylabel('median edge rank')
ax.set_xlim([5, 50])
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/plot_rich.eps')
