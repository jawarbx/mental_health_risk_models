#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=20:00:00

# ===========================================================================
# Function to display help
# ===========================================================================
show_help() {
    cat << EOF
Usage: ./run_training.sh [OPTIONS]
   or: sbatch run_training.sh [OPTIONS]

Options:
    --train_split FLOAT        Training data split ratio (0.0 to 1.0), Default: 0.70
    --test_split FLOAT         Test data split ratio (0.0 to 1.0), Default: 0.05
    --val_split FLOAT          Validation data split ratio (0.0 to 1.0), Default: 0.25
    --batch_size INT           Batch size for training, Default: 16
    --num_epochs INT           Number of training epochs, Default: 3
    --learning_rate FLOAT      Learning rate for optimizer, Default: 2e-5
    -h, --help                 Show this help message and exit

Note: Train, test, and validation splits should sum to 1.0
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
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --train_split|--test_split|--val_split|--learning_rate|\
        --batch_size|--num_epochs)
            PYTHON_ARGS="$PYTHON_ARGS $1 $2"
            shift 2
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
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0

# ===========================================================================
# Logging
# ===========================================================================
echo "=========================================="
echo "Job Information"
echo "=========================================="
if [ -n "$SLURM_JOB_ID" ]; then
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
if [ -n "$SLURM_JOB_ID" ]; then
    MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
    NUM_MACHINES=$SLURM_JOB_NUM_NODES
else
    MASTER_ADDR="localhost"
    NUM_MACHINES=1
fi
MASTER_PORT=29500

echo "MASTER_ADDR     : $MASTER_ADDR"
echo "MASTER_PORT     : $MASTER_PORT"
echo "NUM_MACHINES    : $NUM_MACHINES"
echo "=========================================="

# ===========================================================================
# Launch training
# ===========================================================================
NUM_GPUS=$(python -c 'import torch; print(torch.cuda.device_count())')
echo "Number of processes: $((NUM_MACHINES * NUM_GPUS))"

if [ -n "$SLURM_JOB_ID" ]; then
    srun bash -c '
        source '"$VENV"'/bin/activate
        export MASTER_ADDR='"$MASTER_ADDR"'
        export MASTER_PORT='"$MASTER_PORT"'
        export HF_HOME='"$HF_HOME"'
        export TOKENIZERS_PARALLELISM=false
        export TORCHINDUCTOR_DISABLE=1
        export TORCHDYNAMO_DISABLE=1
        export ACCELERATE_USE_TORCH_COMPILE=0
        export NCCL_DEBUG=INFO
        export NCCL_SOCKET_IFNAME=ib0

        accelerate launch \
            --num_machines='"$NUM_MACHINES"' \
	    --mixed_precision=bf16 \
            --machine_rank=$SLURM_NODEID \
            --main_process_ip='"$MASTER_ADDR"' \
            --main_process_port='"$MASTER_PORT"' \
            --num_processes=$(('"$NUM_MACHINES * $NUM_GPUS"')) \
            src/training.py '"$PYTHON_ARGS"'
    '
else
    accelerate launch \
        --num_machines=1 \
	--mixed_precision=bf16 \
        --machine_rank=0 \
        --main_process_ip="$MASTER_ADDR" \
        --main_process_port="$MASTER_PORT" \
        --num_processes="$NUM_GPUS" \
        src/training.py $PYTHON_ARGS
fi

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
