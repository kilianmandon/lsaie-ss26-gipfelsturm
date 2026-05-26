#!/bin/bash
#
# Generate (or generate+submit) every Rachita Stage-1 experiment in sequence.
#
# Usage:
#   bash rachita_runs/run_benchmarks.sh                                # default: dry-run
#   bash rachita_runs/run_benchmarks.sh --submit                       # actually submit
#   bash rachita_runs/run_benchmarks.sh --size 1.5b --nodes 1          # override
#   bash rachita_runs/run_benchmarks.sh --steps 80
#   bash rachita_runs/run_benchmarks.sh --only baseline                # one config
#   bash rachita_runs/run_benchmarks.sh --only baseline,quack          # multiple configs
#   bash rachita_runs/run_benchmarks.sh --list                         # list short names
#
# Defaults: throughput mode, 760m model, 1 node, 50 steps, dry-run.
#
# Everything is delegated to launch_with_config.py, so SLURM-side resource
# allocation and W&B logging behave identically to teammates' runs.

set -euo pipefail

MODE="throughput"
SIZE="760m"
NODES=1
STEPS=50
DRY_RUN=1
ONLY=""
LIST_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --submit)     DRY_RUN=0 ;;
        --dry-run)    DRY_RUN=1 ;;
        --size)       SIZE="$2"; shift ;;
        --nodes)      NODES="$2"; shift ;;
        --steps)      STEPS="$2"; shift ;;
        --mode)       MODE="$2"; shift ;;
        --only)       ONLY="$2"; shift ;;
        --list)       LIST_ONLY=1 ;;
        -h|--help)
            sed -n '2,17p' "$0"
            exit 0
            ;;
        *)
            echo "unknown arg: $1" >&2
            exit 1
            ;;
    esac
    shift
done

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKDIR=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$WORKDIR"

# Format: <short name>:<config-path>
CONFIGS=(
    "baseline:rachita_runs/configs/config_baseline.yaml"
    "torch_compile_default:rachita_runs/configs/config_torch_compile.yaml"
    "torch_compile_fullmodel:rachita_runs/configs/config_torch_compile_fullmodel.yaml"
    "torch_compile_reduce_overhead:rachita_runs/configs/config_torch_compile_reduce_overhead.yaml"
    "torch_compile_max_autotune:rachita_runs/configs/config_torch_compile_max_autotune.yaml"
    "cuda_graphs_local:rachita_runs/configs/config_cuda_graphs_local.yaml"
    "cuda_graphs_te:rachita_runs/configs/config_cuda_graphs_te.yaml"
    "cuda_graphs_te_attn:rachita_runs/configs/config_cuda_graphs_te_attn.yaml"
    "quack:rachita_runs/configs/config_quack.yaml"
    "context_parallel_cp2:rachita_runs/configs/config_context_parallel_cp2.yaml"
)

# Resolve a user-provided --only token against the short name OR the YAML
# stem OR the YAML file path. Returns the canonical short name on stdout, or
# an empty string when nothing matches.
resolve_only() {
    local needle="$1"
    local short cfg stem stem_strip
    for entry in "${CONFIGS[@]}"; do
        short="${entry%%:*}"
        cfg="${entry#*:}"
        stem=$(basename "$cfg" .yaml)
        stem_strip="${stem#config_}"
        if [ "$needle" = "$short" ] \
           || [ "$needle" = "$cfg" ] \
           || [ "$needle" = "$stem" ] \
           || [ "$needle" = "$stem_strip" ]; then
            echo "$short"
            return 0
        fi
    done
    return 1
}

if [ "$LIST_ONLY" = "1" ]; then
    printf '%-32s %s\n' "SHORT_NAME" "CONFIG"
    for entry in "${CONFIGS[@]}"; do
        printf '%-32s %s\n' "${entry%%:*}" "${entry#*:}"
    done
    exit 0
fi

# Build the set of selected short names. --only accepts either a single name
# or a comma-separated list (e.g. "baseline,quack,torch_compile_default").
ONLY_SET=""
if [ -n "$ONLY" ]; then
    OLDIFS="$IFS"
    IFS=','
    for token in $ONLY; do
        IFS="$OLDIFS"
        token="${token## }"
        token="${token%% }"
        if [ -z "$token" ]; then
            continue
        fi
        if ! resolved=$(resolve_only "$token"); then
            echo "ERROR: --only \"$token\" did not match any known config. Available:" >&2
            for entry in "${CONFIGS[@]}"; do
                echo "  ${entry%%:*}   (${entry#*:})" >&2
            done
            exit 2
        fi
        ONLY_SET="$ONLY_SET|$resolved|"
        IFS=','
    done
    IFS="$OLDIFS"
fi

DRY_FLAG=""
if [ "$DRY_RUN" = "1" ]; then
    DRY_FLAG="--dry_run"
    echo "*** DRY-RUN MODE: no jobs will be submitted. Add --submit to actually queue. ***"
else
    echo "*** SUBMIT MODE: each config will be sent to SLURM. ***"
fi

RAN=0

PY=${PY:-python3}

for entry in "${CONFIGS[@]}"; do
    NAME="${entry%%:*}"
    CFG="${entry#*:}"

    if [ -n "$ONLY_SET" ] && [ "${ONLY_SET#*|$NAME|}" = "$ONLY_SET" ]; then
        continue
    fi

    if [ ! -f "$CFG" ]; then
        echo "[skip] $NAME: $CFG missing"
        continue
    fi

    echo "================================================================"
    echo "[run]  $NAME"
    echo "       config: $CFG"
    echo "       mode=$MODE size=$SIZE nodes=$NODES steps=$STEPS dry=$DRY_RUN"
    echo "================================================================"

    mkdir -p logs
    MARKER=$(mktemp)
    sleep 1
    "$PY" launch_with_config.py "$MODE" "$SIZE" \
        -n "$NODES" -t "$STEPS" -c "$CFG" $DRY_FLAG

    # The launcher derives sbatch file names from seq/mbs/gbs/nodes only, so
    # multiple Rachita configs (e.g. cuda_graphs_local vs cuda_graphs_te) would
    # overwrite each other in dry-run mode. Copy the newest sbatch to a
    # rachita-specific filename right after each invocation.
    NEW_SBATCH=$(find logs/ -maxdepth 1 -name '*.sbatch' -newer "$MARKER" -print | head -n 1 || true)
    rm -f "$MARKER"
    if [ -n "$NEW_SBATCH" ]; then
        TAGGED="logs/rachita-${NAME}-$(basename "$NEW_SBATCH")"
        cp "$NEW_SBATCH" "$TAGGED"
        echo "       wrote: $TAGGED"
    fi
    RAN=$((RAN + 1))
done

if [ "$RAN" = "0" ]; then
    echo "WARNING: no runs were executed (filter --only=${ONLY:-} matched nothing)" >&2
    echo "selected set was: ${ONLY_SET}" >&2
    exit 3
fi

if [ "$DRY_RUN" = "1" ]; then
    echo "done ($RAN sbatch files generated, none submitted)."
    echo "Add --submit to actually queue: bash rachita_runs/run_benchmarks.sh --submit"
else
    echo "done ($RAN jobs submitted). Check with: squeue --me"
fi
