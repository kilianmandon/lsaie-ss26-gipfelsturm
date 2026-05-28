#!/bin/bash
#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=01:00:00
#SBATCH --job-name=vscode_tunnel
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
###SBATCH --mem=460000
#SBATCH --mem=850000
#SBATCH --no-requeue

set -euo pipefail

echo "START TIME: $(date)"

################ Configs ################
WORKDIR="/users/course_00272/lsaie-ss26-gipfelsturm"

SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-288}
SLURM_NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}

# Logging
PROJECT_NAME=gipfelsturm

#########################################
TUNNEL_CMD="code tunnel --name=$CLUSTER_NAME-tunnel"

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TUNNEL_CMD"

echo "END TIME: $(date)"
