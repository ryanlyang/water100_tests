# Repro Runs Layout

This folder is organized for paper-facing reproducibility.

- `waterbirds/train`: fixed training/repro runners.
- `waterbirds/sweeps`: hyperparameter sweep runners.
- `waterbirds/baselines`: zero-shot / non-training baselines.
- `redmeat/train`: fixed training/repro runners.
- `redmeat/sweeps`: hyperparameter sweep runners.
- `redmeat/baselines`: zero-shot / non-training baselines.
- `decoymnist/train`: fixed training/repro runners.
- `decoymnist/baselines`: zero-shot / non-training baselines.
- `evaluation`: lightweight RISE saliency and Pointing Game utilities.
- `third_party`: external dependency code kept vendored and untouched (`GALS`, `CDEP`, `afr`, `group_DRO`).

Path handling in runners is now anchored to `repro_runs/third_party/*` so scripts remain runnable after this reorganization.

ElRep baseline runners are stored with the other non-R4RR methods:
- `other_models/waterbirds/sweeps/elrep_waterbirds_sweep.py`
- `other_models/redmeat/sweeps/elrep_redmeat_sweep.py`
- `other_models/decoymnist/train/elrep_decoy_fixed.py`

Waterbirds and RedMeat ElRep runners sweep ERM learning rates plus `theta1` and
`theta2` representation-regularization weights, then run the best validation
setting across fixed seeds. DecoyMNIST uses the fixed LeNet-style setup.
