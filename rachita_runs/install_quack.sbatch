#!/bin/bash
#
# One-off SLURM job that installs Quack into the user's ~/.local site-packages,
# inside the alps3 container. The install persists across future runs because
# /users is mounted into every alps3 srun.
#
# Submit with:
#   sbatch rachita_runs/install_quack.sbatch
#
# Optional knobs (set on the sbatch line):
#   --export=SOURCE_DIR=/users/$USER/quack    # install from a local Quack clone
#                                             # instead of PyPI / git+https
#
# Logs land in logs/quack-install-<jobid>.log.

#SBATCH --account=lsaie-ss26
#SBATCH --partition=debug
#SBATCH --time=00:15:00
#SBATCH --job-name=quack-install
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --no-requeue

set -euo pipefail

SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-72}

# Forward SOURCE_DIR (if set on sbatch command line) into the container.
SOURCE_DIR=${SOURCE_DIR:-}

echo "[quack-install] host=$(hostname) date=$(date)"
echo "[quack-install] SOURCE_DIR=${SOURCE_DIR:-<unset>}"

srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 \
     --cpus-per-task "$SLURM_CPUS_PER_TASK" --wait 60 \
     bash -lc "
set -euo pipefail

echo '--- Python / nvcc / GPU ---'
python --version
which nvcc && nvcc --version | tail -1 || echo 'nvcc missing (CUDA toolkit not in PATH)'
nvidia-smi -L || echo 'no nvidia-smi output'

USER_SITE=\$(python -c 'import site; print(site.getusersitepackages())')
echo '[quack-install] user site: '\$USER_SITE
mkdir -p \"\$USER_SITE\"

if python -c 'import quack' 2>/dev/null; then
    echo '[quack-install] quack already importable, nothing to do.'
    python -c 'import quack, os; print(\"quack at\", quack.__file__)'
    exit 0
fi

# 1. Try the PyPI wheel first (fastest if a prebuilt aarch64 wheel exists).
if pip install --user --no-cache-dir quack-kernels; then
    echo '[quack-install] PyPI install succeeded.'
elif [ -n \"${SOURCE_DIR:-}\" ] && [ -d \"${SOURCE_DIR:-}\" ]; then
    echo '[quack-install] PyPI failed; installing from SOURCE_DIR='${SOURCE_DIR}
    pip install --user --no-cache-dir -e \"${SOURCE_DIR}\"
else
    echo '[quack-install] PyPI failed; falling back to GitHub source build.'
    pip install --user --no-cache-dir 'git+https://github.com/Dao-AILab/quack.git'
fi

echo '--- verify import ---'
python -c 'import quack; print(\"quack loaded from\", quack.__file__); \
           fn = getattr(quack, \"rmsnorm\", getattr(quack, \"rms_norm\", None)); \
           print(\"rmsnorm fn:\", fn)'
"
echo "[quack-install] done."
