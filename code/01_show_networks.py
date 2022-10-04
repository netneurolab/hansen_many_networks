"""
Figure 1: Organizational properties

Author: Justine Y Hansen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from scipy.spatial.distance import squareform, pdist
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_random_state
from joblib import Parallel, delayed


def exponential(x, a, b, c):
    return a*np.exp(b*x)+c


def compare_exp_lin(x, y, pars):
    """
    compare exponential fit to linear fit.
    x and y should be edge x 1 arrays
    pars should be a len=3 array of exponential parameters
    """
    expfit = exponential(x, pars[0], pars[1], pars[2])
    expresid = sum((y - expfit)**2)
    linreg = LinearRegression()
    linreg.fit(x.reshape(-1, 1), y.reshape(-1, 1))
    linfit = linreg.predict(x.reshape(-1, 1))
    linresid = sum((y.reshape(-1, 1) - linfit)**2)
    return expresid, linresid


def match_length_degree_distribution(data, eu_distance, nbins=10, nswap=None, seed=None):
    """
    Takes a weighted, symmetric connectivity matrix `data` and Euclidean/fiber 
    length matrix `distance` and generates a randomized network with:
        1. exactly the same degree sequence
        2. approximately the same edge length distribution
        3. exactly the same edge weight distribution
        4. approximately the same weight-length relationship

    Parameters
    ----------
    data : (N, N) array-like
        weighted or binary symmetric connectivity matrix.
    distance : (N, N) array-like.3
        symmetric distance matrix.
    nbins : int
        number of distance bins (edge length matrix is performed by swapping
        connections in the same bin). Default = 10.
    nswap : int
        total number of edge swaps to perform. Recommended = nnodes * 20.
        Default = None.

    Returns
    -------
    data : (N, N) array-like
        binary rewired matrix
    W : (N, N) array-like
        weighted rewired matrix
        
    Reference
    ---------
    Betzel, R. F., Bassett, D. S. (2018) Specificity and robustness of long-distance
    connections in weighted, interareal connectomes. PNAS.

    """
    rs = check_random_state(seed)

    nnodes = len(data)             # number of nodes
    
    if nswap is None:
        nswap = nnodes*20          # set default number of swaps
    
    mask = data != 0               # nonzero elements
    mask = np.triu(mask, 1)        # keep upper triangle only
    weights = data[mask]           # values of edge weights
    distances = eu_distance[mask]  # values of edge lengths
    Jdx = np.argsort(distances)    # indices to sort distances in ascending order
    
    bins = np.linspace(min(eu_distance[eu_distance != 0]),
                       max(eu_distance[eu_distance != 0]),
                       nbins+1)  # length/distance of bins
    bins[-1] += 1
    B = np.zeros((nnodes, nnodes, nbins))  # initiate 3D stack of bins
    for k in range(nbins):
        # element is k+1 if the distance falls within the bin, 0 otherwise
        B[:, :, k] = np.logical_and(eu_distance >= bins[k],
                                   eu_distance < bins[k + 1]) * (k + 1)
    # matrix of distance bins
    Bsum = np.sum(B, axis=2)
    
    tmp = np.triu((data != 0)*Bsum, 1)
    row_idx, col_idx = tmp.nonzero()  # indices of edges
    vals = tmp[row_idx, col_idx]
    nedges = len(row_idx)  # number of edges
    iswap = 0             # swap counter
    
    while iswap < nswap:
        myEdge = rs.randint(nedges)   # get a random edge index
        myEdge_row = row_idx[myEdge]  # row idx of edge
        myEdge_col = col_idx[myEdge]  # col idx of edge
        myEdge_bin = vals[myEdge]     # bin of edge
        
        # get indices that can be swapped
        indkeep = (row_idx != myEdge_row) & (row_idx != myEdge_col) \
                  & (col_idx != myEdge_row) & (col_idx != myEdge_col)

        row_idx_keep = row_idx[indkeep]
        col_idx_keep = col_idx[indkeep]
        
        bins_keep = vals[indkeep]  # bins of possible swaps
        
        edge_row = myEdge_row*nnodes + row_idx_keep # edge indices
        edge_row_bins = Bsum[np.unravel_index(edge_row, Bsum.shape)] # matlab-style linear indexing
        edge_col = myEdge_col*nnodes + col_idx_keep # other set of edge indices
        edge_col_bins = Bsum[np.unravel_index(edge_col, Bsum.shape)] 
        
        # get good list of indices
        idx1 = np.logical_and(myEdge_bin == edge_row_bins,
                              bins_keep == edge_col_bins)
        # get other set of good indices
        idx2 = np.logical_and(myEdge_bin == edge_col_bins,
                              bins_keep == edge_row_bins)
        # full set
        goodidx = np.logical_or(idx1, idx2)
        
        # update the indices to keep
        row_idx_keep = row_idx_keep[goodidx]
        col_idx_keep = col_idx_keep[goodidx]
        
        # update the edge indices
        edge_row = myEdge_row*nnodes + row_idx_keep
        edge_col = myEdge_col*nnodes + col_idx_keep
        
        data_row = data[np.unravel_index(edge_row, data.shape)]
        data_col = data[np.unravel_index(edge_col, data.shape)]
        
        # find missing edges
        ind = np.where(np.logical_and(data_row == 0,
                                      data_col == 0).astype(int))[0]
        
        if len(ind) > 0:  # if there is a missing edge
            
            # choose a random swap
            random_swap = ind[rs.randint(len(ind))]
            
            # do the swap
            row_idx_keep = row_idx_keep[random_swap]
            col_idx_keep = col_idx_keep[random_swap]
            
            data[myEdge_row, myEdge_col] = 0
            data[myEdge_col, myEdge_row] = 0
            
            data[row_idx_keep, col_idx_keep] = 0
            data[col_idx_keep, row_idx_keep] = 0
            
            data[myEdge_row, row_idx_keep] = 1
            data[row_idx_keep, myEdge_row] = 1
            
            data[myEdge_col, col_idx_keep] = 1
            data[col_idx_keep, myEdge_col] = 1
            
            other_edge = np.where(indkeep)[0]
            other_edge = other_edge[goodidx]
            other_edge = other_edge[random_swap]
            
            row_idx[myEdge] = min(myEdge_row, row_idx_keep)
            col_idx[myEdge] = max(myEdge_row, row_idx_keep)
            
            row_idx[other_edge] = min(myEdge_col, col_idx_keep)
            col_idx[other_edge] = max(myEdge_col, col_idx_keep)
            
            vals[myEdge] = Bsum[myEdge_row, row_idx_keep]
            vals[other_edge] = Bsum[myEdge_col, col_idx_keep]
            
            iswap += 1
            # if iswap % 100 == 0:
            #     print(iswap)
    
    d = eu_distance[np.where(np.triu(data, 1))]  # get distances where edges are
    jdx = np.argsort(d)                      # sort distances (ascending)
    W = np.zeros((nnodes, nnodes))           # output matrix
    # add weights
    W[np.where(np.triu(data,1))[0][jdx],
      np.where(np.triu(data,1))[1][jdx]] = weights[Jdx]
    
    return data, W


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
cmap_blue = ListedColormap(np.flip(cmap[:128, :], axis=0))

plt.ion()

"""
network heatmap
"""

fig, ax = plt.subplots(1, len(networks.keys()), figsize=(30, 4))
for n, network in enumerate(networks.keys()):
    sns.heatmap(networks[network], square=True, cmap=cmap_div,
                vmin = -3*np.std(networks[network][mask]),
                vmax = 3*np.std(networks[network][mask]),
                ax=ax[n], xticklabels=False, yticklabels=False,
                cbar=False, rasterized=True)
    ax[n].set_title(network)
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/heatmap_networks.eps')

"""
distance hexplot
"""

eu_distance = squareform(pdist(coords, metric="euclidean"))
p0 = [1, -0.05, -0.1]  # initial parameter guess

for network in networks.keys():
    h = sns.jointplot(x=eu_distance[mask], y=networks[network][mask], kind='hex',
                      palette=cmap_blue, rasterized=True)
    h.set_axis_labels('Euclidean distance', network, fontsize=8)
    plt.savefig(path+'figures/' + parc + '/hexplot_' + network + '.eps')

    # compare exponential fit with linear fit
    pars, _ = curve_fit(exponential, eu_distance[mask],
                        networks[network][mask], p0=p0,
                        bounds=([0, -10, -5], [10, 0, 5]))
    print(network)
    print(compare_exp_lin(eu_distance[mask], networks[network][mask], pars))
    print('---------')


"""
connected vs not connected
"""

nnull = 1000
# output = Parallel(n_jobs=36)(delayed(match_length_degree_distribution)(sc, eu_distance, 10, nnodes*20) for i in range(nnull))
sc_rew = np.load(path+'results/sc_betzelnull_'+parc+'.npy')

fig, axs = plt.subplots(1, len(networks.keys()), figsize=(20, 4))
axs = axs.ravel()
for n, network in enumerate(networks.keys()):
    d = dict({'connected' : networks[network][mask][np.where(sc[mask] != 0)],
              'not connected' : networks[network][mask][np.where(sc[mask] == 0)]})
    d = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in d.items()]))
    emp = np.mean(d['connected']) - np.mean(d['not connected'])
    null = np.zeros((nnull, ))
    for k in range(nnull):
        null[k] = np.mean(networks[network][mask]
                          [np.where(sc_rew[:, :, k][mask] == 1)]) \
                  - np.mean(networks[network][mask]
                            [np.where(sc_rew[:, :, k][mask] == 0)])
    p = (1 + np.sum(np.abs((null - np.mean(null)))
                    >= abs((emp - np.mean(null))))) / (nnull + 1)
    sns.boxplot(data=d, ax=axs[n])
    axs[n].set_title([network + ': ' + str(p)[:5]])
plt.tight_layout()
plt.savefig(path+'figures/' + parc + '/boxplot_conn_v_not.eps')


"""
correlation with SC
"""

fig, ax =  plt.subplots(1, len(networks.keys()), figsize=(30, 4))
for n, network in enumerate(networks.keys()):
    print(spearmanr(sc_wei[mask][sc_wei[mask] != 0], networks[network][mask][sc_wei[mask] != 0]))
    ax[n].scatter(sc_wei[mask][sc_wei[mask] != 0],
                networks[network][mask][sc_wei[mask] != 0],
                s=0.75, c='#2080bf', edgecolor=None, rasterized=True)
    ax[n].set_xlabel('connection strength')
    ax[n].set_ylabel('edge strength')
    ax[n].set_title(network)
    ax[n].set_aspect(1./ax[n].get_data_ratio())
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/scatter_sccorr.eps')