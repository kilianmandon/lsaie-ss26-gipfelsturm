from pathlib import Path
import subprocess
import yaml

def launch(mode, model_size, config, training_steps=None, nodes=4, dry_run=False):
    if mode=='throughput':
        if training_steps is None:
            training_steps = 50
        time = '00:30:00'
        eval_interval=training_steps
        eval_iters=0
        lr_warmup_iters=10
        logging_extra=''
        wandb=False
    elif mode=='train':
        if training_steps is None:
            raise ValueError('training_steps must be explicitly set for train runs.')
        time = '02:30:00'
        eval_interval=1000
        eval_iters=10
        lr_warmup_iters=200
        logging_extra='''   --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
'''
        wandb = True
    elif mode=='attention-bench':
        if training_steps is None:
            training_steps = int(config.get('training_steps', 80))
        time = str(config.get('walltime', '00:30:00'))
        eval_interval=training_steps
        eval_iters=0
        lr_warmup_iters=10
        logging_extra='''   --tensorboard-dir $TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard
'''
        wandb = True

    else:
        raise ValueError('Mode must be either "throughput", "train", or "attention-bench".')

    if lr_warmup_iters >= training_steps:
        lr_warmup_iters = max(training_steps - 1, 0)

    if model_size=='125m':
        num_layers=12; hidden=768;ffn=2048;heads=12;kv_heads=4;mbs=16
    elif model_size=='350m':
        num_layers=24; hidden=1024;ffn=2816;heads=16;kv_heads=4;mbs=8
    elif model_size=='760m':
        num_layers=24; hidden=1536;ffn=4096;heads=16;kv_heads=4;mbs=4
    elif model_size=='1.5b':
        num_layers=48; hidden=1600;ffn=4352;heads=20;kv_heads=4;mbs=4
    elif model_size=='3b':
        num_layers=32; hidden=3072;ffn=8192;heads=24;kv_heads=8;mbs=4
    elif model_size=='8b':
        num_layers=32; hidden=4096;ffn=14336;heads=32;kv_heads=8;mbs=2
    else:
        raise ValueError(f'Unknown model size: {model_size}. Must be either 125m, 350m, 760m, 1.5b, 3b, 8b.')

    mbs = int(config.get('micro_batch_size') or mbs)
    gbs = int(config.get('global_batch_size', 256))
    seq_len = int(config.get('seq_len', 4096))
    tp = int(config.get('tensor_parallel_size', 1))
    pp = int(config.get('pipeline_parallel_size', 1))

    attention_backend_config = config.get('attention_backend')
    attention_preset = config.get('attention_preset')
    if attention_backend_config and attention_preset in (None, 'auto'):
        attention_preset = attention_backend_config
    if attention_preset is None:
        attention_preset = 'auto'
    attention_preset = {'fused': 'cudnn'}.get(attention_preset, attention_preset)
    head_dim = hidden // heads
    if attention_preset == 'fa3' and (head_dim != 128 or tp != 1 or pp != 1):
        raise ValueError(
            'attention_preset=fa3 currently requires head_dim=128, TP=1, PP=1 because '
            f'the local FA3 build was pruned to the 8B benchmark shape. Got '
            f'head_dim={head_dim}, TP={tp}, PP={pp}.'
        )
    if attention_preset == 'fa4' and (head_dim != 128 or tp != 1 or pp != 1):
        raise ValueError(
            'attention_preset=fa4 currently requires head_dim=128, TP=1, PP=1 '
            f'for the guarded 8B benchmark path. Got head_dim={head_dim}, TP={tp}, PP={pp}.'
        )
    attention_mask = config.get('attention_mask', 'causal')
    window_size = int(config.get('window_size', 1024))
    attention_backend, backend_env_block = attention_settings(attention_preset, attention_backend_config)

    if attention_mask == 'causal':
        window_desc = 'full'
    elif attention_mask == 'sliding':
        window_desc = str(window_size)
    else:
        raise ValueError(f'Unknown attention_mask: {attention_mask}. Must be causal or sliding.')

    world_size = nodes * 4
    if world_size % (tp * pp) != 0:
        raise ValueError(f'World size {world_size} is not divisible by TP*PP={tp * pp}.')

    job_name=f'gipfel-{mode}-{model_size}-{attention_preset}-{attention_mask}{window_desc}-{seq_len}seq-{mbs}mbs-{gbs}gbs-{nodes}n'

    if wandb:
        wandb_block = '''
# WANDB
if [ -n "$WANDB_API_KEY" ]; then
    echo "[$(date)] WANDB enabled."
    TRAINING_CMD="$TRAINING_CMD \\
        --wandb-save-dir $LOG_DIR \\
        --wandb-project $PROJECT_NAME \\
        --wandb-exp-name $EXP_NAME-$SLURM_JOB_ID"
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'''
    else:
        wandb_block = 'export WANDB_MODE=disabled'


    Path('logs').mkdir(exist_ok=True, parents=True)

    script_location = Path(f'logs/{job_name}.sbatch')

    whole_script = '#!/bin/bash'

    whole_script += sbatch_directives(job_name, nodes, time, config)
    whole_script += script_body()
    whole_script += script_configs(
        mbs, gbs, seq_len, training_steps, mode, model_size, tp, pp,
        attention_preset, attention_backend, attention_mask, window_size, window_desc,
    )
    whole_script += script_setup(config)
    whole_script += profiling_args(config)
    whole_script += script_attention(backend_env_block, attention_backend, attention_mask, window_size)
    whole_script += script_model(num_layers, hidden, ffn, heads, kv_heads)
    whole_script += script_training(eval_interval, eval_iters, lr_warmup_iters)
    whole_script += script_rest(tp, pp)

    whole_script += f'{logging_extra})'
    whole_script += script_tokenizer()
    
    whole_script += f'\n{wandb_block}\n'
    whole_script += script_footer(config)

    with open(script_location, 'w') as f:
        f.write(whole_script)

    if not dry_run:
        subprocess.run(
            [
                'sbatch',
                '--parsable',
                '-A', str(config.get('account', 'lsaie-ss26')),
                str(script_location),
            ],
            check=True,
        )



    


def attention_settings(attention_preset, attention_backend=None):
    aliases = {
        'fused': 'cudnn',
    }
    preset = aliases.get(attention_preset, attention_preset)
    backend = aliases.get(attention_backend, attention_backend) if attention_backend else None
    if preset in ('fa3', 'fa4') and backend in (None, preset, 'flash'):
        backend = preset
    if backend is not None and backend != preset:
        raise ValueError(
            f'attention_backend={attention_backend} conflicts with attention_preset={attention_preset}.'
        )

    if preset == 'auto':
        return 'auto', '''
unset NVTE_FLASH_ATTN
unset NVTE_FUSED_ATTN
unset NVTE_UNFUSED_ATTN
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
'''
    if preset == 'fa3':
        return 'flash', '''
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
export MEGATRON_FA3_CORE_ATTN=1
export FA3_USERBASE=${FA3_USERBASE:-/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_probe_minimal/python_userbase}
export FA3_SHIM_DIR=${FA3_SHIM_DIR:-$WORKDIR/fa3_shim}
export PYTHONPATH="$FA3_SHIM_DIR:$FA3_USERBASE/lib/python3.12/site-packages:${PYTHONPATH:-}"
if command -v python >/dev/null 2>&1; then
python - <<'PY'
from flash_attn_interface import flash_attn_func
from flash_attn_3.flash_attn_interface import flash_attn_func as te_flash_attn_func
print("FA3 core attention enabled via", flash_attn_func, te_flash_attn_func)
PY
else
    echo "Skipping pre-srun FA3 import check because python is not on the batch host PATH."
fi
'''
    if preset == 'fa4':
        return 'flash', '''
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
export MEGATRON_FA4_CORE_ATTN=1
export FA4_USERBASE=${FA4_USERBASE:-/iopsstor/scratch/cscs/$USER/gipfelsturm/fa4_probe/python_userbase}
export FA4_SITE_PACKAGES=$FA4_USERBASE/lib/python3.12/site-packages
export FA4_CUTLASS_PACKAGES=$FA4_SITE_PACKAGES/nvidia_cutlass_dsl/python_packages
export CUTE_DSL_CACHE_DIR=${CUTE_DSL_CACHE_DIR:-/iopsstor/scratch/cscs/$USER/gipfelsturm/fa4_probe/cute_cache}
export PYTHONPATH="$FA4_SITE_PACKAGES:$FA4_CUTLASS_PACKAGES:${PYTHONPATH:-}"
echo "FA4 core attention enabled from FA4_USERBASE=$FA4_USERBASE"
'''
    if preset == 'flash':
        return 'flash', '''
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
'''
    if preset == 'cudnn':
        return 'fused', '''
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=0
export NVTE_FUSED_ATTN_BACKEND=1
export NVTE_FUSED_ATTN_USE_FAv2_BWD=0
'''
    if preset == 'unfused':
        return 'unfused', '''
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
'''
    raise ValueError(f'Unknown attention_preset: {attention_preset}. Must be auto, flash, fa3, fa4, cudnn, or unfused.')


def sbatch_directives(job_name, nodes, time, config):
    return f'''
#SBATCH --account={config.get('account', 'lsaie-ss26')}
#SBATCH --partition={config.get('partition', 'normal')}
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
'''

def script_body():
    cwd = str(Path('.').resolve())
    return f'''
set -euo pipefail

echo "START TIME: $(date)"

################ Configs ################
WORKDIR="{cwd}"
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
'''


def script_configs(
    mbs, gbs, seq_len, training_steps, mode, model_size, tp, pp,
    attention_preset, attention_backend, attention_mask, window_size, window_desc,
):
    return f'''
# Training config
MBS={mbs}
GBS={gbs}
SEQ_LEN={seq_len}
TRAINING_STEPS={training_steps}
TP={tp}
PP={pp}
ATTN_PRESET={attention_preset}
ATTN_BACKEND={attention_backend}
ATTN_MASK={attention_mask}
WINDOW_SIZE={window_size}

SLURM_GPUS_PER_NODE=${{SLURM_GPUS_PER_NODE:-4}}
SLURM_CPUS_PER_TASK=${{SLURM_CPUS_PER_TASK:-288}}
SLURM_NNODES=${{SLURM_NNODES:-${{SLURM_JOB_NUM_NODES:-1}}}}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME={mode}-{model_size}-{attention_preset}-{attention_mask}{window_desc}-seq{seq_len}-mbs{mbs}-gbs{gbs}-${{SLURM_NNODES}}n
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
'''

def profiling_args(config):
    if config['memory_profiling']:
        flags = [
            '--use-pytorch-profiler',
            '--record-memory-history',
            '--memory-snapshot-path $WORKDIR/logs/memory-$EXP_NAME.pkl'
        ]
    else:
        flags = []

    profiling_args_string = '\n'.join(
        [f'    {s}' for s in flags]
    )

    if profiling_args_string:
        return rf'''
PROFILING_ARGS=(
{profiling_args_string}
)
'''
    else:
        return rf'''
PROFILING_ARGS = ()
'''


    

def script_setup(config):
    transformer_engine_flags = [
        '--transformer-impl transformer_engine',
    ]
    if config['precision'] == 'bf16':
        transformer_engine_flags += [
            '--use-precision-aware-optimizer',
            '--main-grads-dtype bf16',
        ]
    elif config['precision'] == 'fp8_transformer_engine':
        transformer_engine_flags += [
            '--fp8-format hybrid',
            '--fp8-recipe tensorwise',
            '--fp8-param-gather',
            # Not sure if these options are good with fp8 either
            '--main-grads-dtype bf16',
            '--use-precision-aware-optimizer',
            # Reconsider if we should have this
            '--attention-softmax-in-fp32',
        ]
        # TODO: I think we might need --tp-comm-overlap with fp8 if we are doing tensorparallel
    else:
        raise ValueError(f'Precision training not implemented yet: {config["precision"]}.')

    transformer_engine_args_string = '\n'.join(
        [f'    {s}' for s in transformer_engine_flags]
    )

    return rf'''
#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $WORKDIR/logs/megatron-patch.lock bash -c "
cd $MEGATRON_LM_DIR
PATCH_FILES=\$(find $WORKDIR/patches -maxdepth 1 -name '*.patch' | sort)
LAST_PATCH=\$(printf '%s\n' \$PATCH_FILES | tail -n 1)
if git apply --check \$PATCH_FILES 2>/dev/null; then
    git apply \$PATCH_FILES
elif git apply --reverse --check \$LAST_PATCH 2>/dev/null; then
    echo 'Megatron patches already applied.'
else
    echo 'Megatron patch state is neither clean nor already applied.'
    git status --short
    exit 1
fi"
export PYTHONPATH=$MEGATRON_LM_DIR:${{PYTHONPATH:-}}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

echo "Megatron commit: $(git -C $MEGATRON_LM_DIR rev-parse --short HEAD)"
echo "Attention preset/backend: $ATTN_PRESET/$ATTN_BACKEND mask=$ATTN_MASK window=$WINDOW_SIZE seq=$SEQ_LEN mbs=$MBS gbs=$GBS tp=$TP pp=$PP"

TRANSFORMER_ENGINE_ARGS=(
{transformer_engine_args_string}
)
'''

def script_attention(backend_env_block, attention_backend, attention_mask, window_size):
    window_arg = ''
    if attention_mask == 'sliding':
        window_arg = f'''
    --window-size {window_size},0'''

    return f'''
{backend_env_block}

echo "NVTE env: NVTE_FLASH_ATTN=${{NVTE_FLASH_ATTN:-unset}} NVTE_FUSED_ATTN=${{NVTE_FUSED_ATTN:-unset}} NVTE_UNFUSED_ATTN=${{NVTE_UNFUSED_ATTN:-unset}} NVTE_FUSED_ATTN_BACKEND=${{NVTE_FUSED_ATTN_BACKEND:-unset}} NVTE_FUSED_ATTN_USE_FAv2_BWD=${{NVTE_FUSED_ATTN_USE_FAv2_BWD:-unset}}"

ATTENTION_ARGS=(
    --attention-backend {attention_backend}{window_arg}
)
'''

def script_model(num_layers, hidden, ffn, heads, kv_heads):
    return f'''
NETWORK_SIZE_ARGS=(
    --num-layers {num_layers}
    --hidden-size {hidden}
    --ffn-hidden-size {ffn}
    --num-attention-heads {heads}
    --group-query-attention
    --num-query-groups {kv_heads}
    --max-position-embeddings $SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length $SEQ_LEN
)
'''

def script_training(eval_interval, eval_iters, lr_warmup_iters):
    return f'''
TRAINING_ARGS=(
    --micro-batch-size $MBS
    --global-batch-size $GBS
    --train-iters $TRAINING_STEPS
    --log-interval 1
    --eval-interval {eval_interval}
    --eval-iters {eval_iters}
    --cross-entropy-loss-fusion
    --disable-bias-linear
    --optimizer adam
    --dataloader-type single
    --no-check-for-nan-in-loss-and-grad
    --manual-gc
    --manual-gc-interval 50
)

REGULARIZATION_ARGS=(
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --weight-decay 0.1
    --clip-grad 1.0
    --adam-beta1 0.9
    --adam-beta2 0.95
)

LEARNING_RATE_ARGS=(
    --lr 3e-4
    --lr-decay-style constant
    --lr-warmup-iters {lr_warmup_iters}
)
'''


def script_rest(tp, pp):
    return f'''
INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size {tp}
    --pipeline-model-parallel-size {pp}
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress

'''


def script_tokenizer():
    return '''

TOKENIZER_ARGS=(
    --tokenizer-type GPT2BPETokenizer
    --vocab-file $WORKDIR/data/gpt2-vocab.json
    --merge-file $WORKDIR/data/gpt2-merges.txt
)

DATA_ARGS=(
    --data-path $DATA_PREFIX
    --data-cache-path $DATASET_CACHE_DIR
    --split 99,1,0
    --num-workers 1
)

TORCHRUN_ARGS=(
    --nproc-per-node $SLURM_GPUS_PER_NODE
    --nnodes $SLURM_NNODES
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT
    --rdzv_backend c10d
    --max_restarts 0
    --tee 3
)

TRAINING_CMD="torchrun ${TORCHRUN_ARGS[@]} $MEGATRON_LM_DIR/pretrain_gpt.py \\
    ${TRANSFORMER_ENGINE_ARGS[@]} \\
    ${NETWORK_SIZE_ARGS[@]} \\
    ${TRAINING_ARGS[@]} \\
    ${REGULARIZATION_ARGS[@]} \\
    ${LEARNING_RATE_ARGS[@]} \\
    ${INITIALIZATION_ARGS[@]} \\
    ${MIXED_PRECISION_ARGS[@]} \\
    ${DISTRIBUTED_ARGS[@]} \\
    ${ATTENTION_ARGS[@]} \\
    ${LOGGING_ARGS[@]} \\
    ${TOKENIZER_ARGS[@]} \\
    ${DATA_ARGS[@]} \\
    ${PROFILING_ARGS[@]}"
'''


def script_footer(config):
    return f'''
echo "[config]"
echo "{yaml.dump(config)}"
echo "[config_end]"

echo "CMD: $TRAINING_CMD"
srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task $SLURM_CPUS_PER_TASK --wait 60 bash -c "numactl --membind=0-3 $TRAINING_CMD"

echo "END TIME: $(date)"
'''



        
if __name__=='__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
                    prog='Launcher',
                    description='Starts a throughput, train, or attention benchmark run.')

    parser.add_argument('mode', help='Must be throughput, train, or attention-bench')
    parser.add_argument('model_size', help='Must be one of  125m, 350m, 760m, 1.5b, 3b, 8b.')
    parser.add_argument('-n', '--nodes', default=4, required=False)
    parser.add_argument('-t', '--training_steps', default=None, required=False)
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('--dry_run', action='store_true', help='If set, the sbatch script is only generated but not launched.')

    args = parser.parse_args()

    with open('default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.config is not None:
        with open(args.config, 'r') as f:
            config_addon = yaml.safe_load(f)
        if config_addon:
            config.update(config_addon)



    mode = args.mode
    model_size = args.model_size
    nodes = int(args.nodes)
    training_steps = int(args.training_steps) if args.training_steps is not None else None

    
    launch(mode, model_size, config, training_steps, nodes, dry_run=args.dry_run)
