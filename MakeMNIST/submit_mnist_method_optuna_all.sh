#!/bin/bash -l
set -Eeuo pipefail

sbatch run_colormnist_vanilla_optuna.sbatch
sbatch run_colormnist_cdep_optuna.sbatch
sbatch run_colormnist_rrr_optuna.sbatch
sbatch run_colormnist_eg_optuna.sbatch

sbatch run_decoymnist_vanilla_optuna.sbatch
sbatch run_decoymnist_cdep_optuna.sbatch
sbatch run_decoymnist_rrr_optuna.sbatch
sbatch run_decoymnist_eg_optuna.sbatch

echo "Submitted 8 Optuna jobs (ColorMNIST + DecoyMNIST, 4 methods each)."
