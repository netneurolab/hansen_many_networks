"""
Run community detection on lab computer
"""

import numpy as np
from netneurotools.modularity import consensus_modularity
from joblib import Parallel, delayed

def community_detection(A, gamma_range):
    nnodes = len(A)
    ngamma = len(gamma_range)
    consensus = np.zeros((nnodes, ngamma))
    qall = []
    zrand = []
    i = 0
    for g in gamma_range:
        consensus[:, i], q, z = consensus_modularity(A, g, B='negative_asym')
        qall.append(q)
        zrand.append(z)
        i += 1
    return(consensus, qall, zrand)

path = '/home/jhansen/gitrepos/hansen_many_networks/'
parc = 'Schaefer400'

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
nnodes = gc.shape[0]

# normalize networks
for network in networks.keys():
    networks[network] = np.arctanh(networks[network])
    networks[network][np.eye(nnodes).astype(bool)] = 0

# run community detection
gamma_range = [x/10.0 for x in range(1, 61, 1)]
output = Parallel(n_jobs=36)(delayed(community_detection)(networks[network], gamma_range) for network in networks.keys())

# save out
for network in range(len(output)):
    np.save(path+'results/community_detection_' + parc + '/community_assignments_' + list(networks.keys())[network] + '.npy',
            output[network][0])
    np.save(path+'results/community_detection_' + parc + '/community_qall_' + list(networks.keys())[network] + '.npy',
            np.array(output[network][1]))
    np.save(path+'results/community_detection_' + parc + '/community_zrand_' + list(networks.keys())[network] + '.npy',
            np.array(output[network][2]))
