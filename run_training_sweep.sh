#!/bin/bash

#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00

# Function to display help
show_help() {
    cat << EOF
Usage: ./run_training_sweep.sh [OPTIONS]
   or: sbatch run_training_sweep.sh [OPTIONS]

Run one W&B sweep trial per SLURM job.

Options:
    --sweep_id STRING          W&B sweep ID (entity/project/sweep_id). Required.
    -h, --help                 Show this help message and exit

Sweep usage:
    # 1. Create sweep once on login node:
    #    wandb sweep sweep.yaml --project my-project  -> prints SWEEP_ID
    # 2. Submit N jobs (one trial each):
    #    for i in \$(seq 1 20); do sbatch run_training_sweep.sh --sweep_id entity/project/SWEEP_ID; done

EOF
    exit 0
}

SWEEP_ID=""
source .slurm

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        --sweep_id)
            SWEEP_ID="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [[ -z "$SWEEP_ID" ]]; then
    echo "ERROR: --sweep_id is required"
    echo "Use --help for usage information"
    exit 1
fi

# ===========================================================================
# Validate required environment variables
# ===========================================================================
missing_vars=()
[[ -z "${VENV:-}"            ]] && missing_vars+=("VENV")
[[ -z "${HF_HOME:-}"         ]] && missing_vars+=("HF_HOME")
[[ -z "${EXPERIMENT_NAME:-}" ]] && missing_vars+=("EXPERIMENT_NAME")
[[ -z "${LOGS:-}"            ]] && missing_vars+=("LOGS")

if [[ ${#missing_vars[@]} -gt 0 ]]; then
    echo "ERROR: The following required environment variables are not set:"
    for var in "${missing_vars[@]}"; do
        echo "  - $var"
    done
    echo "Please define them in your .slurm file."
    exit 1
fi

if [[ ! -f "$VENV/bin/activate" ]]; then
    echo "ERROR: uv virtual environment not found at: $VENV"
    echo "Please create it first with: uv venv \$VENV"
    exit 1
fi

# ===========================================================================
# Setup
# ===========================================================================
mkdir -p "$LOGS" "$HF_HOME"

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
fi
echo "Experiment      : $EXPERIMENT_NAME"
echo "Mode            : W&B Sweep"
echo "Sweep ID        : $SWEEP_ID"
echo "Start Time      : $(date)"
echo "=========================================="
echo "Python          : $(python --version)"
echo "PyTorch         : $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available  : $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count       : $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

# ===========================================================================
# Launch sweep agent (--count 1 = one trial per job)
# sweep_launcher.sh handles the srun + accelerate launch internally.
# ===========================================================================
chmod +x sweep_launcher.sh
wandb agent --count 1 "$SWEEP_ID"

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
