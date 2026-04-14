#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:l40s:4
#SBATCH --mem=128G
#SBATCH --time=4:00:00

# ===========================================================================
# Function to display help
# ===========================================================================
show_help() {
    cat << EOF
Usage: ./run_threshold_sweep.sh [OPTIONS]
   or: sbatch run_threshold_sweep.sh [OPTIONS]

Options:
    --model_path PATH          Path to trained model directory
                               Default: final_model in MODEL_DIR
    --batch_size INT           Batch size for inference, Default: 16
    --thresholds FLOATS        Space-separated thresholds to sweep
                               Default: 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50
    -h, --help                 Show this help message and exit

Note:
    - Threshold sweep runs on validation set first
    - Best threshold is then applied to test set
    - Results saved to MODEL_DIR/threshold_sweep/
EOF
    exit 0
}

# ===========================================================================
# Source environment variables and parse arguments
# ===========================================================================
source .slurm
echo "LOGS=${LOGS}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "USER_EMAIL=${USER_EMAIL}"
echo "scontrol exit code: $?"

PYTHON_ARGS=""
FORCE_LOCAL=0
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --model_path|--batch_size)
            PYTHON_ARGS="$PYTHON_ARGS $1 $2"
            shift 2
            ;;
        --thresholds)
            # Collect all threshold values until next flag or end
            PYTHON_ARGS="$PYTHON_ARGS $1"
            shift
            while [[ $# -gt 0 && ! "$1" == --* ]]; do
                PYTHON_ARGS="$PYTHON_ARGS $1"
                shift
            done
            ;;
	--local|-l)
	    FORCE_LOCAL=1
	    shift
	    ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ===========================================================================
# Validate required environment variables
# ===========================================================================
missing_vars=()
[[ -z "$VENV"            ]] && missing_vars+=("VENV")
[[ -z "$HF_HOME"         ]] && missing_vars+=("HF_HOME")
[[ -z "$EXPERIMENT_NAME" ]] && missing_vars+=("EXPERIMENT_NAME")
[[ -z "$LOGS"            ]] && missing_vars+=("LOGS")

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are not set:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please define them in your .slurm file."
    exit 1
fi

# ===========================================================================
# Setup
# ===========================================================================
mkdir -p "$LOGS" "$HF_HOME"

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: uv virtual environment not found at: $VENV"
    echo "Please create it first with: uv venv \$VENV"
    exit 1
fi

source "$VENV/bin/activate"

export HF_HOME
export TOKENIZERS_PARALLELISM=false
export TORCHINDUCTOR_DISABLE=1
export TORCHDYNAMO_DISABLE=1
export ACCELERATE_USE_TORCH_COMPILE=0

# ===========================================================================
# Logging
# ===========================================================================
echo "=========================================="
echo "Job Information"
echo "=========================================="
if [ -n "$SLURM_JOB_ID" ] && [ "$FORCE_LOCAL" -eq 0 ]; then
    echo "Execution Mode  : SLURM"
    echo "Job ID          : $SLURM_JOB_ID"
    echo "Job Name        : $SLURM_JOB_NAME"
    echo "Nodes           : $SLURM_NODELIST"
    echo "GPUs per node   : $SLURM_GPUS_ON_NODE"
    echo "CPUs per task   : $SLURM_CPUS_PER_TASK"
else
    echo "Execution Mode  : Direct"
    echo "Hostname        : $(hostname)"
    echo "CPUs available  : $(nproc)"
    command -v nvidia-smi &> /dev/null && \
        echo "GPUs available  : $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
fi
echo "Experiment      : $EXPERIMENT_NAME"
echo "Start Time      : $(date)"
echo "Python Args     : $PYTHON_ARGS"
echo "=========================================="
echo "Python          : $(python --version)"
echo "PyTorch         : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available  : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count       : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# ===========================================================================
# Master address / port resolution
# ===========================================================================
# Threshold sweep is single-node only — no multi-node needed
if [ -n "$SLURM_JOB_ID" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
else
    MASTER_ADDR="localhost"
fi

MASTER_PORT=29500
NUM_MACHINES=1

echo "MASTER_ADDR     : $MASTER_ADDR"
echo "MASTER_PORT     : $MASTER_PORT"
echo "NUM_MACHINES    : $NUM_MACHINES"
echo "=========================================="

# ===========================================================================
# Launch threshold sweep
# ===========================================================================
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "Number of processes: $NUM_GPUS"

if [ -n "$SLURM_JOB_ID" ] && [ "$FORCE_LOCAL" -eq 0 ]; then
    srun bash -c '
        source '"$VENV"'/bin/activate
        export MASTER_ADDR='"$MASTER_ADDR"'
        export MASTER_PORT='"$MASTER_PORT"'
        export HF_HOME='"$HF_HOME"'
        export TOKENIZERS_PARALLELISM=false
        export TORCHINDUCTOR_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export ACCELERATE_USE_TORCH_COMPILE=0

        accelerate launch \
            --num_machines=1 \
            --mixed_precision=bf16 \
            --machine_rank=0 \
            --main_process_ip='"$MASTER_ADDR"' \
            --main_process_port='"$MASTER_PORT"' \
            --num_processes='"$NUM_GPUS"' \
            src/threshold_sweep.py '"$PYTHON_ARGS"'
    '
else
    accelerate launch \
        --num_machines=1 \
        --mixed_precision=bf16 \
        --machine_rank=0 \
        --main_process_ip="$MASTER_ADDR" \
        --main_process_port="$MASTER_PORT" \
        --num_processes="$NUM_GPUS" \
        src/threshold_sweep.py $PYTHON_ARGS
fi

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
