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

    else:
        raise ValueError('Mode must be either "throughput" or "train".')

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

    gbs=256
    seq_len=4096

    job_name=f'gipfel-{mode}-{model_size}-{training_steps}s-{nodes}n'

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

    whole_script += sbatch_directives(job_name, nodes, time)
    whole_script += script_body()
    whole_script += script_configs(mbs, gbs, seq_len, training_steps, mode, model_size)
    whole_script += script_setup(config)
    whole_script += script_model(num_layers, hidden, ffn, heads, kv_heads)
    whole_script += script_training(eval_interval, eval_iters, lr_warmup_iters)
    whole_script += script_rest()

    whole_script += f'{logging_extra})'
    whole_script += script_tokenizer()
    
    whole_script += f'\n{wandb_block}\n'
    whole_script += script_footer(config)

    with open(script_location, 'w') as f:
        f.write(whole_script)

    if not dry_run:
        subprocess.run(['sbatch', str(script_location)])



    


def sbatch_directives(job_name, nodes, time):
    return f'''
#SBATCH --account=lsaie-ss26
#SBATCH --partition=normal
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
echo "START TIME: $(date)"

################ Configs ################
WORKDIR="{cwd}"
MEGATRON_LM_DIR=$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/cache
'''


def script_configs(mbs, gbs, seq_len, training_steps, mode, model_size):
    return f'''
# Training config
MBS={mbs}
GBS={gbs}
SEQ_LEN={seq_len}
TRAINING_STEPS={training_steps}

# Logging
PROJECT_NAME=gipfelsturm
EXP_NAME={mode}-{model_size}-${{SLURM_NNODES}}n
LOG_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/$PROJECT_NAME/$EXP_NAME
TENSORBOARD_DIR=$LOG_DIR/tensorboard
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

    return f'''
#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $MEGATRON_LM_DIR/.git-lock bash -c "cd $MEGATRON_LM_DIR && git checkout -- . && git apply $WORKDIR/patches/*.patch"
export PYTHONPATH=$MEGATRON_LM_DIR:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.triton_cache
export TORCHINDUCTOR_CACHE_DIR=/iopsstor/scratch/cscs/$USER/gipfelsturm/.inductor_cache
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK/SLURM_GPUS_PER_NODE))
MASTER_ADDR=$(hostname)
MASTER_PORT=25678

TRANSFORMER_ENGINE_ARGS=(
{transformer_engine_args_string}
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


def script_rest():
    return '''
INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
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
    ${LOGGING_ARGS[@]} \\
    ${TOKENIZER_ARGS[@]} \\
    ${DATA_ARGS[@]}"
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
                    description='Starts either a throughput or training test run.')

    parser.add_argument('mode', help='Must be throughput or training')
    parser.add_argument('model_size', help='Must be one of  125m, 350m, 760m, 1.5b, 3b, 8b.')
    parser.add_argument('-n', '--nodes', default=4, required=False)
    parser.add_argument('-t', '--training_steps', default=None, required=False)
    parser.add_argument('-c', '--config', default=None)
    parser.add_argument('--dry_run', action=argparse.BooleanOptionalAction, help='If set, the sbatch script is only generated but not launched.')

    args = parser.parse_args()

    with open('default_config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    if args.config is not None:
        with open(args.config, 'r') as f:
            config_addon = yaml.safe_load(f)
        config |= config_addon



    mode = args.mode
    model_size = args.model_size
    nodes = int(args.nodes)
    training_steps = int(args.training_steps) if args.training_steps is not None else None

    
    launch(mode, model_size, config, training_steps, nodes, dry_run=args.dry_run)
