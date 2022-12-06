# Multimodal, multiscale connectivity blueprints of the cerebral cortex

This repository contains code and data used and created in support of my project "Multimodal, multiscale connectivity blueprints of the cerebral cortex", now up on [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.12.02.518906v1.abstract).
All code was written in Python.

All connectivity modes are available in three parcellation resolutions (Cammoun-033/Desikan-Killiany, Schaefer-100 (7 Networks order), Schaefer-400 (7 Networks order)) - see [data](data/).

## `code`
The [code](code/) folder contains all the code used to run the analyses and generate the figures.
This folder contains the following files:
- [01_show_networks.py](code/01_show_networks.py) corresponds to Figure 1 and plots each connectivity mode as well as some fundamental organizational properties (relatoinship with distance, structure).
- [02_richlinks.py](code/02_richlinks.py) corresponds to Figure 2 and compares connectivity modes to one another with respect to (1) distance and (2) structure, especially the rich club.
- [03_hubs.py](code/03_hubs.py) corresponds to Figure 3 and looks at the strongest edges and hubs of each network individually as well as across all networks.
- [04_gradients_communities.py](code/04_gradients_communities.py) corresponds to Figure 4 and looks at the first principal component and community decomposition (Louvain algorithm) of each network.
- [05_disease_exposure.py](code/05_disease_exposure.py) corresponds to Figure 5 and uses ENIGMA data to study how connectivity modes may shape cortical abnormalities. ENIGMA profiles were fetched from [hansen_crossdisorder_vulnerability](https://github.com/netneurolab/hansen_crossdisorder_vulnerability) which uses the [`enigmatoolbox`](https://enigma-toolbox.readthedocs.io/en/latest/).
- [06_fusion.py](code/06_fusion.py) corresponds to Figure 6 and fuses all connectivity modes into a single network. I use [Similarity Network Fusion](http://compbio.cs.toronto.edu/SNF/SNF/Software.html) to pull this off (code from [`snfpy`](https://snfpy.readthedocs.io/en/latest/)).
- [make_networks.py](code/make_networks.py) shows how each connectiviy mode was made and saved.
- [scpt_community_detection.py](code/scpt_community_detection.py) is the script I used to run and parallelize the Louvain algorithm.

## `data`
The [data](data/) folder contains the processed data I used to run the analyses. All seven connectivity modes in region-by-region format are available in three parcellation resolutions, as both `.npy` and `.csv` files.
These can be found in the folder corresponding to each parcellation atlas, e.g. [Cammoun033](data/Cammoun033/) (68 cortical regions, identical to the Desikan-Killiany atlas but ordered differently; this is the version used in Figure 5), [Schaefer100](data/Schaefer100) (100 cortical regions, following the `7Networks` order), and [Schaefer400](data/Schaefer400) (400 cortical regions, following the `7Networks` order; this is the version used in the main analyses).
There are some extra files in there, used in [make_networks.py](code/make_networks.py) or elsewhere.
Parcellation information can be found in the [parcellation_files](data/parcellation_files/) folder.

## `results`
The [results](results/) folder contains outputs from my scripts (if they are small enough to put up on GitHub).

## `manuscript`
The [manuscript](manuscript/) folder contains the manuscript draft.
