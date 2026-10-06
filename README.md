# Spectral Braids: Braid Monodromy for Dynamic Directed Networks

Code for the paper "Spectral Braids: Braid Monodromy for Dynamic Directed Networks".
It computes the spectral braid of a dynamic directed network, turns it into features
(Burau matrix, tracked geometry), generates the two synthetic datasets, and reproduces
every table and figure of the paper.

## Requirements

Python 3.11. Install the libraries with

    pip install -r requirements.txt

The results in the paper were produced with numpy 2.4, scipy 1.17, scikit-learn 1.8,
pandas 3.0, matplotlib 3.10, joblib 1.5 and networkx 3.6.

## The method

| File | Content |
|---|---|
| `braids.py` | Eigenvalue tracking along the path between snapshots (`track`), reading the braid word on a fixed projection line (`word`), free reduction, permutation |
| `feats.py` | Upper-half-plane strands (`strands`), Burau matrix and Burau features, tracked geometry, and the baselines (pooled spectra, start and end, sorted series) |
| `forums.py` | Transition matrix of a network made of groups of 3 or 4 communities; event trajectories (calm / turbulent) |
| `data.py` | The two synthetic datasets: `LABELS` (History, 4 classes) and `ORDERS` (Order, 8 classes); `sample(label, rng, regime, users, deg)` returns the community transition matrices of one network |

Minimal use on your own data: build one row-normalised community transition matrix per
snapshot, then

    import braids, feats
    Lu = feats.strands(list_of_matrices)     # tracked eigenvalues in the upper half-plane
    w = braids.word(Lu)                      # braid word; +i is sigma_i, -i is sigma_i^{-1}
    x = feats.burau_features(w)              # fixed-size feature vector

## Reproducing the experiments

All scripts are run from this folder and use fixed random seeds.

| Paper | Command | Output |
|---|---|---|
| Table II, cross-validation columns | `python exp_main_table.py history 150` and `python exp_main_table.py order 75` | `sparse_history.json`, `sparse_order.json` |
| Table II, transfer columns | `python exp_sensitivity_transfer_timing.py shift` | `shift.json` |
| Sensitivity figure (History) | `python exp_sensitivity_transfer_timing.py sweep U` for `U` in 10, 20, 40, 80, 160 | `sweep_U.json` |
| Sensitivity, Order dataset | `python exp_order_densities.py U` for `U` in 10, 40, 80 | `dens_U.json` |
| Running time | `python exp_sensitivity_transfer_timing.py time` | printed |
| Invariance check (calm vs turbulent) | `python exp_invariance_check.py` | printed |

`U` is the average number of users per community; each user sends 10 interactions per
snapshot. The folder `results/` contains the output files behind the numbers in the paper.
Each run takes a few minutes on four CPU cores.

## Reproducing the figures

Figures are written to `figures_out/`.

| Figure | Command |
|---|---|
| What is a braid | `python fig_what_is_braid.py` |
| Pipeline, class examples, sensitivity | `python fig_pipeline_classes_sweep.py` |
| One crossing; one group of three communities | `python fig_crossing_example.py` |
| Mirror pairs and the upper-half-plane braid | `python fig_uhp.py` |
| Sorted against tracked eigenvalues | `python fig_sort.py` |
| Closed and open paths | `python fig_open.py` |

## Real datasets

`real_data/` contains the scripts for the four real networks discussed in the paper
(e-mail, international trade, Reddit hyperlinks, Bay Area Bike Share). The data files
are not included. Download them from their sources and set the folder at the top of
each script:

- email-Eu-core-temporal and soc-RedditHyperlinks: https://snap.stanford.edu/data/
- Correlates of War dyadic trade: https://correlatesofwar.org/data-sets/bilateral-trade/
- Bay Area Bike Share trip histories, August 2013 to August 2015

These scripts are exploratory and are provided as they were run.

## Preliminary extensions

`preliminary_node_level/` contains two experiments that are not in the paper: the
spectral braid of the node-level matrix without a community partition (`nodelevel.py`)
and node classification through eigenvector loadings (`hybrid.py`).

## Citation

    @inproceedings{spectralbraids,
      title  = {Spectral Braids: Braid Monodromy for Dynamic Directed Networks},
      author = {Aktas, Mehmet Emin and Akbas, Esra},
      year   = {2026}
    }
