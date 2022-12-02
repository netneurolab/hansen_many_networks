# -*- coding: utf-8 -*-
"""
This script constructs the raw (pre-normalized) similarity networks, or
attempts to describe how raw connectomes were constructed.
Correlated gene expression, receptor similarity, laminar similarity
are straight forward.
Metabolic connectivity, haemodynamic connectivity, and electrophysiological
connectivity require preprocessing and registration of the individual data,
which is done separately.
Temporal similarity involves running HCTSA in Matlab, so I've also left it
out.

Author: Justine Y Hansen
"""

import numpy as np
import pandas as pd
import nibabel as nib
from netneurotools.datasets import fetch_schaefer2018, fetch_cammoun2012
from nilearn.datasets import fetch_atlas_schaefer_2018
from neuromaps.parcellate import Parcellater
import abagen
from scipy.stats import zscore
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

"""
set-up parcellation
"""

path = '/home/jhansen/gitrepos/hansen_many_networks/'
parc = 'Schaefer400'

coords = np.genfromtxt(path+'data/parcellation_files/' + parc + '_coords.txt')
coords = coords[:, -3:]
nnodes = len(coords)

if parc == 'Schaefer400':
    parc_file_mni = fetch_atlas_schaefer_2018(n_rois=400)['maps']
    parc_file_fsaverage = fetch_schaefer2018(version='fsaverage')['400Parcels7Networks']
    cortex = np.arange(nnodes)
elif parc == 'Schaefer100':
    parc_file_mni = fetch_atlas_schaefer_2018(n_rois=100)['maps']
    parc_file_fsaverage = fetch_schaefer2018(version='fsaverage')['100Parcels7Networks']
    cortex = np.arange(nnodes)
elif parc == 'Cammoun033':
    parc_file_mni = fetch_cammoun2012()['scale033']
    parc_file_fsaverage = fetch_cammoun2012(version='fsaverage')['scale033']
    info = pd.read_csv(fetch_cammoun2012()['info'])
    cortex = np.array(info.query('scale == "scale033" & structure == "cortex"')['id']) - 1
    coords = coords[cortex, :]
    nnodes = len(coords)


"""
correlated gene expression
"""

expression = abagen.get_expression_data(parc_file_mni,
                                        lr_mirror='bidirectional',
                                        missing='interpolate',
                                        return_donors=True)
expression, ds = abagen.correct.keep_stable_genes(list(expression.values()),
                                                  threshold=0.1,
                                                  percentile=False,
                                                  return_stability=True)
expression = pd.concat(expression).groupby('label').mean()
gc = np.corrcoef(zscore(expression[cortex, :]))
np.savetxt(path+'data/' + parc + '/gene_coexpression.csv', gc, delimiter=',')
np.save(path+'data/' + parc + '/gene_coexpression.npy', gc)


"""
receptor similarity
"""

# from https://github.com/netneurolab/hansen_receptors
recpath = '/home/jhansen/gitrepos/hansen_receptors/data/PET_nifti_images/'

receptors_nii = [recpath+'/5HT1a_way_hc36_savli.nii',
                 recpath+'/5HT1b_p943_hc22_savli.nii',
                 recpath+'/5HT1b_p943_hc65_gallezot.nii.gz',
                 recpath+'/5HT2a_cimbi_hc29_beliveau.nii',
                 recpath+'/5HT4_sb20_hc59_beliveau.nii',
                 recpath+'/5HT6_gsk_hc30_radhakrishnan.nii.gz',
                 recpath+'/5HTT_dasb_hc100_beliveau.nii',
                 recpath+'/A4B2_flubatine_hc30_hillmer.nii.gz',
                 recpath+'/CB1_omar_hc77_normandin.nii.gz',
                 recpath+'/D1_SCH23390_hc13_kaller.nii',
                 recpath+'/D2_flb457_hc37_smith.nii.gz',
                 recpath+'/D2_flb457_hc55_sandiego.nii.gz',
                 recpath+'/DAT_fpcit_hc174_dukart_spect.nii',
                 recpath+'/GABAa-bz_flumazenil_hc16_norgaard.nii',
                 recpath+'/H3_cban_hc8_gallezot.nii.gz', 
                 recpath+'/M1_lsn_hc24_naganawa.nii.gz',
                 recpath+'/mGluR5_abp_hc22_rosaneto.nii',
                 recpath+'/mGluR5_abp_hc28_dubois.nii',
                 recpath+'/mGluR5_abp_hc73_smart.nii',
                 recpath+'/MU_carfentanil_hc204_kantonen.nii',
                 recpath+'/NAT_MRB_hc77_ding.nii.gz',
                 recpath+'/VAChT_feobv_hc3_spreng.nii',
                 recpath+'/VAChT_feobv_hc4_tuominen.nii',
                 recpath+'/VAChT_feobv_hc5_bedard_sum.nii',
                 recpath+'/VAChT_feobv_hc18_aghourian_sum.nii']

parcellated = {}
name = {}

parcellater = Parcellater(parc_file_mni, 'MNI152')
for receptor in receptors_nii:
    parcellated[receptor] = parcellater.fit_transform(receptor, 'MNI152', True)[cortex]
    name[receptor] = receptor.split('/')[-1]  # get nifti file name
    name[receptor] = name[receptor].split('.')[0]  # remove .nii

r = np.array([*parcellated.values()], dtype=np.float16).T

receptor_names = np.array(["5HT1a", "5HT1b", "5HT2a", "5HT4", "5HT6", "5HTT", "A4B2",
                           "CB1", "D1", "D2", "DAT", "GABAa", "H3", "M1", "mGluR5",
                           "MOR", "NET", "VAChT"])

np.save(path+'data/receptor_names.npy', receptor_names)

receptor_data = np.zeros([nnodes, len(receptor_names)])
receptor_data[:, 0] = r[:, 0]
receptor_data[:, 2:9] = r[:, 3:10]
receptor_data[:, 10:14] = r[:, 12:16]
receptor_data[:, 15:17] = r[:, 19:21]
# weighted average of 5HT1B p943
receptor_data[:, 1] = (zscore(r[:, 1])*22 + zscore(r[:, 2])*65) / (22+65)
# weighted average of D2 flb457
receptor_data[:, 9] = (zscore(r[:, 10])*37 + zscore(r[:, 11])*55) / (37+55)
# weighted average of mGluR5 ABP688
receptor_data[:, 14] = (zscore(r[:, 16])*22 + zscore(r[:, 17])*28 + zscore(r[:, 18])*73) / (22+28+73)
# weighted average of VAChT FEOBV
receptor_data[:, 17] = (zscore(r[:, 21])*3 + zscore(r[:, 22])*4 + zscore(r[:, 23]) + zscore(r[:, 24])) / \
                       (3+4+5+18)
del r

rs = np.corrcoef(zscore(receptor_data))  # receptor similarity
np.savetxt(path+'data/' + parc + '/receptor_similarity.csv', rs, delimiter=',')
np.save(path+'data/' + parc + '/receptor_similarity.npy', rs)


"""
laminar similarity
"""

# from https://github.com/caseypaquola/BigBrainWarp
bigbrainpath = '/home/gitrepos/BigBrainWarp/'
intensities = np.genfromtxt(bigbrainpath+'spaces/tpl-fsaverage/tpl-fsaverage_den-164k_desc-profiles.txt',
                            delimiter=',')  # takes a while

intensities_parc = []
for hem in parc_file_fsaverage:
    labels, ctab, names = nib.freesurfer.read_annot(hem)
    dataparc = np.zeros((np.max(labels), 50))
    for i in range(np.max(labels)):
        dataparc[i, :] = np.mean(intensities[:, np.where(labels == i+1)[0]], axis=1)
    if parc == 'Cammoun033':
        dataparc = np.delete(dataparc, 3, 0)  # remove corpus callosum
    intensities_parc.append(dataparc)
intensities_parc = np.concatenate(intensities_parc)
np.savetxt(path+'data/' + parc + '/bigbrain_intensities.csv', intensities_parc, delimiter=',')

# then I go to matlab to do a partial pearson's correlation,
# correcting for the mean intensity.
# laminar_similarity = partialcorr(intensities, mean(intensities, 2))

mp = np.genfromtxt(path + 'data/' + parc + '/laminar_similarity.csv', delimiter=',')
np.save(path+'data/' + parc + '/laminar_similarity.npy', mp)


"""
metabolic connectivity
"""

# from https://doi.org/10.1038/s41597-020-00699-5
# spatiotemporal filtering happened in Matlab
# registration to MNI-152, parcellating, and making the connectome
# happened in Python using antspy
# takes a long time

"""
haemodynamic connectivity
"""

# comes from HCP

"""
electrophysiological connectivity (MEG)
"""

megfc = np.load(path+'data/' + parc + '/groupFCmeg_aec_orth_' + parc + '.npy.npz')
mask = np.triu(np.ones(nnodes), 1) > 0
megfc_vec = np.array([megfc['megfc'][i, :, :][mask] for i in range(megfc['megfc'].shape[0])])
pca = PCA(n_components=1)
pc1 = pca.fit_transform(megfc_vec.T)
mc = np.zeros((nnodes, nnodes))
mc[np.triu(np.ones(nnodes, dtype=bool), 1)] = np.squeeze(pc1)
mc = mc + mc.T
np.fill_diagonal(mc, 1)
np.savetxt(path + 'data/' + parc + '/electrophysiological_connectivity.csv', mc, delimiter=',')
np.save(path + 'data/' + parc + '/electrophysiological_connectivity.npy', mc)

# supplementary figure

concated = np.concatenate((megfc['megfc'], np.expand_dims(mc, axis=0)))
fig, ax = plt.subplots(1, 7, figsize=(30, 4))
for n in range(concated.shape[0]):
    sns.heatmap(concated[n, :, :], square=True, cmap=cmap_div,
                vmin = -max(abs(concated[n, :, :][mask])),
                vmax = max(abs(concated[n, :, :][mask])),
                ax=ax[n], xticklabels=False, yticklabels=False,
                cbar=False, rasterized=True)
    ax[n].set_title(np.append(megfc['bands'], 'pc1')[n])
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/heatmap_supp_megnetworks.eps')

megfc_vec = np.concatenate((megfc_vec, mc[mask].reshape(1, -1)))
plt.rcParams['svg.fonttype'] = 'none'
fig, ax = plt.subplots()
sns.heatmap(np.corrcoef(megfc_vec), square=True, cmap=cmap_div,
                        vmin=-1, vmax=1, xticklabels=np.append(megfc['bands'], 'pc1'),
                        yticklabels=np.append(megfc['bands'], 'pc1'), annot=True)
plt.tight_layout()
plt.savefig(path+'figures/'+parc+'/heatmap_megcorrs.svg')

"""
temporal similarity
"""

# comes from running hctsa (https://github.com/benfulcher/hctsa)
# on the fMRI time-series, as per Shafiei et al., 2020 (https://doi.org/10.7554/eLife.62116)
# takes a while (like, a couple days to a week for Schaefer 100 with big time parallelization)
