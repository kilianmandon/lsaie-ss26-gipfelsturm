import argparse
from pathlib import Path
import re
import subprocess
import os
import stat



def convert(sbatch_path):
    with open(sbatch_path, 'r') as f:
        sbatch_script = f.read()


    m = re.search(r'#SBATCH --job-name=([-\w]+)', sbatch_script)
    if m:
        job_name = m.group(1)
    else:
        raise ValueError('Job name could not be parsed.')

    m = re.compile(r'(srun [^\n]*\n)', re.MULTILINE)
    local_script = m.sub(r'# \1 \neval "$TRAINING_CMD"\n', sbatch_script)

    # SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}
    # SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-288}
    # SLURM_NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}
    local_script = local_script.replace('SLURM_GPUS_PER_NODE=${SLURM_GPUS_PER_NODE:-4}', 'SLURM_GPUS_PER_NODE=1')
    local_script = local_script.replace('SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-288}', 'SLURM_CPUS_PER_TASK=8')
    local_script = local_script.replace('SLURM_NNODES=${SLURM_NNODES:-${SLURM_JOB_NUM_NODES:-1}}', 'SLURM_NNODES=1')
    local_script = local_script.replace('set -euo pipefail', '')

    local_script = local_script.replace('DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small', 'DATA_PREFIX=${WORKDIR}/local_data/climbmix_small')
    local_script = local_script.replace('DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache', 'DATASET_CACHE_DIR=${WORKDIR}/local_data/climbmix_small')

    script_path = f'logs/local-{sbatch_path.stem}.sh'

    with open(script_path, 'w') as f:
        f.write(local_script)

    st = os.stat(script_path)
    os.chmod(script_path, st.st_mode | stat.S_IEXEC)

    log_file = f'logs/log-{job_name}.txt'
    with open(log_file, 'w') as f:
        subprocess.call([script_path] , stdout=f, stderr=f)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser('sbatch to bash conversion helper')
    parser.add_argument('input')
    args = parser.parse_args()
    sbatch_path = Path(args.input)
    convert(sbatch_path)