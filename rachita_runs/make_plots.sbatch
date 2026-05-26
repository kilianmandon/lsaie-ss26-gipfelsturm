#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:05:00
#SBATCH --job-name=rachita-make-plots
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
# We take the full node (matching the training sbatch). Partial allocations
# on Clariden's GH200 partitions cause pyxis to fail to create the per-job
# pmix overlay in /dev/shm/<user>/enrootrun/overlay1/var/spool/slurmd/
# because that directory is shared with whoever else has the other GPUs.
# This script only uses CPU for ~30 seconds; the extra GPUs/CPUs are wasted
# but the job is so short the cost is negligible.

set -euo pipefail

WORKDIR="/iopsstor/scratch/cscs/course_00135/lsaie-ss26-gipfelsturm"
cd "$WORKDIR"

# Pure post-processing job. Loads matplotlib from the alps3 container and
# regenerates rachita_runs/results/figures/8b_{throughput,loss,memory}_bar.png
# plus the JSON sidecar. ~30s of actual work.
#
# srun flags match the working training-job pattern exactly (see any
# rachita-*-gipfel-throughput-*.sbatch). In particular --network=disable_rdzv_get
# is a CSCS-specific flag that disables the CXI rendezvous setup which would
# otherwise make pyxis fail to mount /var/spool/slurmd/pmix.<jobid> in the
# container overlay (the exact error we hit earlier).
# The container's workdir is $HOME (per alps3.toml), not the project dir,
# so paths inside the python invocation must be absolute -- or we cd first.
# Wrapping in bash -c lets us cd inside the container, matching how the
# training sbatch invokes pretrain_gpt.py with an absolute path.
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task=$SLURM_CPUS_PER_TASK --wait 60 \
     bash -c "cd $WORKDIR && python3 rachita_runs/make_plots.py \
        logs/gipfel-throughput-8b-*.log \
        --skip 10 \
        --out-dir rachita_runs/results/figures"
