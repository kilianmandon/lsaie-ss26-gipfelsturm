#!/bin/bash
#
# Usage: ./launch.sh <mode> <model_size> [steps] [nodes]
#
# Modes:     throughput       (50 steps, with W&B)
#            train            (N steps, with W&B and Tensorboard)
#            attention-bench  (fixed-config attention backend benchmark)
#
# Sizes:     125m, 350m, 760m, 1.5b, 3b, 8b
#
# Steps:     required for train mode (e.g., 1000, 5000, 15000)
# Nodes:     optional, default 4 (max 8)
#
# Examples:  ./launch.sh throughput 760m
#            ./launch.sh throughput 8b 50 1
#            ./launch.sh train 760m 5000
#            ./launch.sh train 1.5b 3000 8
#            SUBMIT=0 ATTN_PRESET=flash ./launch.sh attention-bench 760m 80 1
#            SUBMIT=0 ATTN_PRESET=fa3 ./launch.sh attention-bench 760m 5 1

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
if [ -f "$SCRIPT_DIR/config.sh" ]; then
    # config.sh is intentionally git-ignored and may contain private values.
    source "$SCRIPT_DIR/config.sh"
fi

WORKDIR=${GIPFEL_WORKDIR:-${WORKDIR:-$PWD}}
SBATCH_ACCOUNT=${SBATCH_ACCOUNT:-lsaie-ss26}
SLURM_PARTITION=${GIPFEL_PARTITION:-normal}
SUBMIT=${SUBMIT:-1}

MODE=${1:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}
MODEL_SIZE=${2:?Usage: ./launch.sh <mode> <model_size> [steps] [nodes]}

################ Mode config ################
case $MODE in
    throughput)
        TRAINING_STEPS=${3:-50}
        NODES=${4:-4}
        TIME=00:30:00
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA=""
        WANDB=true
        ;;
    train)
        TRAINING_STEPS=${3:?Usage: ./launch.sh train <model_size> <steps> [nodes]}
        NODES=${4:-4}
        TIME=02:30:00
        EVAL_INTERVAL=1000
        EVAL_ITERS=10
        LR_WARMUP_ITERS=200
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    attention-bench)
        TRAINING_STEPS=${STEPS:-${3:-80}}
        NODES=${NODES:-${4:-1}}
        TIME=${WALLTIME:-00:30:00}
        EVAL_INTERVAL=$TRAINING_STEPS
        EVAL_ITERS=0
        LR_WARMUP_ITERS=10
        LOGGING_EXTRA="
    --tensorboard-dir \$TENSORBOARD_DIR
    --log-timers-to-tensorboard
    --log-memory-to-tensorboard"
        WANDB=true
        ;;
    *)
        echo "Unknown mode: $MODE. Choose: throughput, train, attention-bench"
        exit 1
        ;;
esac

if [ "$LR_WARMUP_ITERS" -ge "$TRAINING_STEPS" ]; then
    if [ "$TRAINING_STEPS" -gt 1 ]; then
        LR_WARMUP_ITERS=$((TRAINING_STEPS - 1))
    else
        LR_WARMUP_ITERS=0
    fi
fi

################ Model config ################
case $MODEL_SIZE in
    125m)
        NUM_LAYERS=12;  HIDDEN=768;  FFN=2048;  HEADS=12; KV_HEADS=4
        DEFAULT_MBS=16
        ;;
    350m)
        NUM_LAYERS=24; HIDDEN=1024; FFN=2816;  HEADS=16; KV_HEADS=4
        DEFAULT_MBS=8
        ;;
    760m)
        NUM_LAYERS=24; HIDDEN=1536; FFN=4096;  HEADS=16; KV_HEADS=4
        DEFAULT_MBS=4
        ;;
    1.5b)
        NUM_LAYERS=48; HIDDEN=1600; FFN=4352;  HEADS=20; KV_HEADS=4
        DEFAULT_MBS=4
        ;;
    3b)
        NUM_LAYERS=32; HIDDEN=3072; FFN=8192;  HEADS=24; KV_HEADS=8
        DEFAULT_MBS=4
        ;;
    8b)
        NUM_LAYERS=32; HIDDEN=4096; FFN=14336; HEADS=32; KV_HEADS=8
        DEFAULT_MBS=2
        ;;
    *)
        echo "Unknown model size: $MODEL_SIZE. Choose: 125m, 350m, 760m, 1.5b, 3b, 8b"
        exit 1
        ;;
esac

MBS=${MBS:-$DEFAULT_MBS}
GBS=${GBS:-256}
SEQ_LEN=${SEQ_LEN:-4096}
TP=${TP:-1}
PP=${PP:-1}
ATTN_MASK=${ATTN_MASK:-causal}
WINDOW_SIZE=${WINDOW_SIZE:-1024}
if [ -z "${ATTN_PRESET+x}" ]; then
    case ${ATTN_BACKEND:-auto} in
        fused) ATTN_PRESET=cudnn ;;
        auto|flash|fa3|unfused) ATTN_PRESET=${ATTN_BACKEND:-auto} ;;
        *)
            echo "Unknown ATTN_BACKEND: ${ATTN_BACKEND}. Choose: auto, flash, fa3, fused, unfused"
            exit 1
            ;;
    esac
fi

case $ATTN_PRESET in
    auto)
        RESOLVED_ATTN_BACKEND=auto
        BACKEND_ENV_BLOCK='
unset NVTE_FLASH_ATTN
unset NVTE_FUSED_ATTN
unset NVTE_UNFUSED_ATTN
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD'
        ;;
    fa3)
        RESOLVED_ATTN_BACKEND=flash
        BACKEND_ENV_BLOCK='
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD
export MEGATRON_FA3_CORE_ATTN=1
export FA3_USERBASE=${FA3_USERBASE:-/iopsstor/scratch/cscs/$USER/gipfelsturm/fa3_probe_minimal/python_userbase}
export PYTHONPATH="$FA3_USERBASE/lib/python3.12/site-packages:${PYTHONPATH:-}"
python - <<'"'"'PY'"'"'
from flash_attn_interface import flash_attn_func
print("FA3 core attention enabled via", flash_attn_func)
PY'
        ;;
    flash)
        RESOLVED_ATTN_BACKEND=flash
        BACKEND_ENV_BLOCK='
export NVTE_FLASH_ATTN=1
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=0
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD'
        ;;
    cudnn)
        RESOLVED_ATTN_BACKEND=fused
        BACKEND_ENV_BLOCK='
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=1
export NVTE_UNFUSED_ATTN=0
export NVTE_FUSED_ATTN_BACKEND=1
export NVTE_FUSED_ATTN_USE_FAv2_BWD=0'
        ;;
    unfused)
        RESOLVED_ATTN_BACKEND=unfused
        BACKEND_ENV_BLOCK='
export NVTE_FLASH_ATTN=0
export NVTE_FUSED_ATTN=0
export NVTE_UNFUSED_ATTN=1
unset NVTE_FUSED_ATTN_BACKEND
unset NVTE_FUSED_ATTN_USE_FAv2_BWD'
        ;;
    *)
        echo "Unknown ATTN_PRESET: $ATTN_PRESET. Choose: auto, flash, fa3, cudnn, unfused"
        exit 1
        ;;
esac

if [ -n "${ATTN_BACKEND:-}" ] && [ "$ATTN_BACKEND" != "$RESOLVED_ATTN_BACKEND" ] && [ "$ATTN_PRESET" != "fa3" ]; then
    echo "ATTN_BACKEND=$ATTN_BACKEND conflicts with ATTN_PRESET=$ATTN_PRESET, which resolves to $RESOLVED_ATTN_BACKEND"
    exit 1
fi

HEAD_DIM=$((HIDDEN / HEADS))
if [ "$ATTN_PRESET" = "fa3" ]; then
    if [ "$HEAD_DIM" -ne 128 ] || [ "$TP" -ne 1 ] || [ "$PP" -ne 1 ]; then
        echo "ATTN_PRESET=fa3 currently requires head_dim=128, TP=1, PP=1 because the local FA3 build was pruned to the 8B benchmark shape. Got head_dim=$HEAD_DIM TP=$TP PP=$PP."
        exit 1
    fi
fi

case $ATTN_MASK in
    causal) WINDOW_DESC=full ;;
    sliding) WINDOW_DESC=${WINDOW_SIZE} ;;
    *)
        echo "Unknown ATTN_MASK: $ATTN_MASK. Choose: causal, sliding"
        exit 1
        ;;
esac

WORLD_SIZE=$((NODES * 4))
MODEL_PARALLEL_SIZE=$((TP * PP))
if [ $((WORLD_SIZE % MODEL_PARALLEL_SIZE)) -ne 0 ]; then
    echo "World size $WORLD_SIZE is not divisible by TP*PP=$MODEL_PARALLEL_SIZE"
    exit 1
fi

JOB_NAME="gipfel-${MODE}-${MODEL_SIZE}-${ATTN_PRESET}-${ATTN_MASK}${WINDOW_DESC}-${SEQ_LEN}seq-${MBS}mbs-${GBS}gbs-${NODES}n"

################ W&B block ################
if [ "$WANDB" = true ]; then
    WANDB_BLOCK='
# WANDB
if [ -z "${WANDB_API_KEY:-}" ] && [ -f "$WORKDIR/local_files/credentials.md" ]; then
    WANDB_API_KEY=$(awk '"'"'BEGIN{IGNORECASE=1} /^wandb api key:/ {sub(/^[^:]*:[[:space:]]*/, ""); print; exit}'"'"' "$WORKDIR/local_files/credentials.md")
    export WANDB_API_KEY
fi
if [ -n "${WANDB_API_KEY:-}" ]; then
    echo "[$(date)] WANDB enabled."
    export WANDB_ENTITY
    export WANDB_PROJECT="$PROJECT_NAME"
    export WANDB_DIR="$LOG_DIR"
    TRAINING_CMD+=(
        --wandb-save-dir "$LOG_DIR"
        --wandb-entity "$WANDB_ENTITY"
        --wandb-project "$PROJECT_NAME"
        --wandb-exp-name "$EXP_NAME-$SLURM_JOB_ID"
    )
else
    export WANDB_MODE=disabled
    echo "[$(date)] WANDB disabled."
fi'
else
    WANDB_BLOCK='export WANDB_MODE=disabled'
fi

################ Generate script ################
mkdir -p logs

SCRIPT="logs/${JOB_NAME}.sbatch"

cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
HEADER

cat >> "$SCRIPT" << SBATCH_DIRECTIVES
#SBATCH --account=${SBATCH_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${TIME}
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=logs/%x-%j.log
#SBATCH --error=logs/%x-%j.log
#SBATCH --nodes=${NODES}
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --mem=460000
#SBATCH --no-requeue
SBATCH_DIRECTIVES

cat >> "$SCRIPT" << 'BODY_HEAD'
set -euo pipefail

echo "START TIME: $(date)"

################ Configs ################
BODY_HEAD

cat >> "$SCRIPT" << BODY_WORKDIR
WORKDIR=${WORKDIR}
MEGATRON_LM_DIR=\$WORKDIR/Megatron-LM
DATA_PREFIX=/capstor/store/cscs/swissai/infra01/datasets/nvidia/Nemotron-ClimbMix/climbmix_small_megatron/climbmix_small
DATASET_CACHE_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/cache
BODY_WORKDIR

cat >> "$SCRIPT" << CONFIGS

# Training config
MBS=${MBS}
GBS=${GBS}
SEQ_LEN=${SEQ_LEN}
TRAINING_STEPS=${TRAINING_STEPS}
TP=${TP}
PP=${PP}
ATTN_PRESET=${ATTN_PRESET}
ATTN_BACKEND=${RESOLVED_ATTN_BACKEND}
ATTN_MASK=${ATTN_MASK}
WINDOW_SIZE=${WINDOW_SIZE}

# Slurm does not guarantee that every #SBATCH resource value is exported as an
# environment variable on every site, so normalize the values used below.
SLURM_GPUS_PER_NODE=\${SLURM_GPUS_PER_NODE:-4}
SLURM_CPUS_PER_TASK=\${SLURM_CPUS_PER_TASK:-288}
SLURM_NNODES=\${SLURM_NNODES:-\${SLURM_JOB_NUM_NODES:-${NODES}}}

# Logging
WANDB_ENTITY=\${GIPFEL_WANDB_ENTITY:-cler}
PROJECT_NAME=\${GIPFEL_WANDB_PROJECT:-}
if [ -z "\$PROJECT_NAME" ]; then
    PROJECT_NAME="fla's"
fi
EXP_NAME=${MODE}-${MODEL_SIZE}-${ATTN_PRESET}-${ATTN_MASK}${WINDOW_DESC}-seq${SEQ_LEN}-mbs${MBS}-gbs${GBS}-\${SLURM_NNODES}n
LOG_DIR=/iopsstor/scratch/cscs/\$USER/gipfelsturm/\$PROJECT_NAME/\$EXP_NAME
TENSORBOARD_DIR=\$LOG_DIR/tensorboard
CONFIGS

cat >> "$SCRIPT" << 'SETUP'

#########################################

mkdir -p logs $LOG_DIR $TENSORBOARD_DIR $DATASET_CACHE_DIR

cd $MEGATRON_LM_DIR
flock $WORKDIR/logs/megatron-patch.lock bash -c "
cd $MEGATRON_LM_DIR
if git apply --check $WORKDIR/patches/*.patch 2>/dev/null; then
    git apply $WORKDIR/patches/*.patch
elif git apply --reverse --check $WORKDIR/patches/*.patch 2>/dev/null; then
    echo 'Megatron patches already applied.'
else
    echo 'Megatron patch state is neither clean nor already applied.'
    git status --short
    exit 1
fi"
export PYTHONPATH=$MEGATRON_LM_DIR:${PYTHONPATH:-}
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
    --transformer-impl transformer_engine
    --use-precision-aware-optimizer
    --main-grads-dtype bf16
)

SETUP

cat >> "$SCRIPT" << BACKEND_ENV
${BACKEND_ENV_BLOCK}

echo "NVTE env: NVTE_FLASH_ATTN=\${NVTE_FLASH_ATTN:-unset} NVTE_FUSED_ATTN=\${NVTE_FUSED_ATTN:-unset} NVTE_UNFUSED_ATTN=\${NVTE_UNFUSED_ATTN:-unset} NVTE_FUSED_ATTN_BACKEND=\${NVTE_FUSED_ATTN_BACKEND:-unset} NVTE_FUSED_ATTN_USE_FAv2_BWD=\${NVTE_FUSED_ATTN_USE_FAv2_BWD:-unset}"

ATTENTION_ARGS=(
    --attention-backend ${RESOLVED_ATTN_BACKEND}
)
BACKEND_ENV

if [ "$ATTN_MASK" = "sliding" ]; then
cat >> "$SCRIPT" << SLIDING
ATTENTION_ARGS+=(
    --window-size ${WINDOW_SIZE},0
)
SLIDING
fi

cat >> "$SCRIPT" << MODEL
NETWORK_SIZE_ARGS=(
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN}
    --ffn-hidden-size ${FFN}
    --num-attention-heads ${HEADS}
    --group-query-attention
    --num-query-groups ${KV_HEADS}
    --max-position-embeddings \$SEQ_LEN
    --position-embedding-type rope
    --normalization RMSNorm
    --swiglu
    --untie-embeddings-and-output-weights
    --seq-length \$SEQ_LEN
)
MODEL

cat >> "$SCRIPT" << TRAINING

TRAINING_ARGS=(
    --micro-batch-size \$MBS
    --global-batch-size \$GBS
    --train-iters \$TRAINING_STEPS
    --log-interval 1
    --eval-interval ${EVAL_INTERVAL}
    --eval-iters ${EVAL_ITERS}
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
    --lr-warmup-iters ${LR_WARMUP_ITERS}
)
TRAINING

cat >> "$SCRIPT" << 'REST'

INITIALIZATION_ARGS=(
    --seed 42
    --init-method-std 0.02
)

MIXED_PRECISION_ARGS=(
    --bf16
)

DISTRIBUTED_ARGS=(
    --tensor-model-parallel-size $TP
    --pipeline-model-parallel-size $PP
    --use-distributed-optimizer
    --overlap-grad-reduce
    --overlap-param-gather
)

LOGGING_ARGS=(
    --log-throughput
    --log-progress
REST

cat >> "$SCRIPT" << LOGGING_EXTRA
${LOGGING_EXTRA}
)
LOGGING_EXTRA

cat >> "$SCRIPT" << 'TOKENIZER'

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

TRAINING_CMD=(
    torchrun
    "${TORCHRUN_ARGS[@]}"
    "$MEGATRON_LM_DIR/pretrain_gpt.py"
    "${TRANSFORMER_ENGINE_ARGS[@]}"
    "${NETWORK_SIZE_ARGS[@]}"
    "${TRAINING_ARGS[@]}"
    "${REGULARIZATION_ARGS[@]}"
    "${LEARNING_RATE_ARGS[@]}"
    "${INITIALIZATION_ARGS[@]}"
    "${MIXED_PRECISION_ARGS[@]}"
    "${DISTRIBUTED_ARGS[@]}"
    "${ATTENTION_ARGS[@]}"
    "${LOGGING_ARGS[@]}"
    "${TOKENIZER_ARGS[@]}"
    "${DATA_ARGS[@]}"
)

TOKENIZER

cat >> "$SCRIPT" << 'WANDB_PLACEHOLDER'
WANDB_PLACEHOLDER

# Replace placeholder with actual W&B block
sed -i '/^WANDB_PLACEHOLDER$/d' "$SCRIPT"
cat >> "$SCRIPT" << WANDB_INSERT
${WANDB_BLOCK}
WANDB_INSERT

cat >> "$SCRIPT" << 'FOOTER'

printf 'CMD:'
printf ' %q' "${TRAINING_CMD[@]}"
printf '\n'
if [ "${GIPFEL_DIRECT_RUN:-0}" = "1" ]; then
    numactl --membind=0-3 "${TRAINING_CMD[@]}"
else
    srun -lu --mpi=pmix --network=disable_rdzv_get --environment=alps3 --cpus-per-task "$SLURM_CPUS_PER_TASK" --wait 60 bash -c 'numactl --membind=0-3 "$@"' bash "${TRAINING_CMD[@]}"
fi

echo "END TIME: $(date)"
FOOTER

chmod +x "$SCRIPT"

echo "Generated: $SCRIPT"
if [ "$SUBMIT" = "1" ]; then
    unset SBATCH_PARTITION
    sbatch --parsable -A "$SBATCH_ACCOUNT" -p "$SLURM_PARTITION" "$SCRIPT"
else
    echo "SUBMIT=$SUBMIT, not submitting."
fi
