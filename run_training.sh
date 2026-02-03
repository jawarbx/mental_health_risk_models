#!/bin/bash

# Function to display help
show_help() {
    cat << EOF
Usage: ./run_training.sh [OPTIONS]
   or: sbatch run_training.sh [OPTIONS]

Train MCI prediction model with specified parameters.

Options:
    --train_split FLOAT        Training data split ratio (0.0 to 1.0)
                               Default: 0.70
                               Example: --train_split 0.75
    
    --test_split FLOAT         Test data split ratio (0.0 to 1.0)
                               Default: 0.05
                               Example: --test_split 0.10
    
    --val_split FLOAT          Validation data split ratio (0.0 to 1.0)
                               Default: 0.25
                               Example: --val_split 0.20
    
    --batch_size INT           Batch size for training
                               Default: 16
                               Example: --batch_size 32
    
    --num_epochs INT           Number of training epochs
                               Default: 3
                               Example: --num_epochs 5
    
    --learning_rate FLOAT      Learning rate for optimizer
                               Default: 2e-5
                               Example: --learning_rate 1e-5
    
    -h, --help                 Show this help message and exit

Examples:
    # Direct execution
    ./run_training.sh --batch_size 32 --num_epochs 5
    ./run_training.sh --learning_rate 1e-5 --train_split 0.8
    
    # SLURM submission
    sbatch run_training.sh --batch_size 16 --num_epochs 3
    sbatch run_training.sh --help (not recommended to run this)

Note: Train, test, and validation splits should sum to 1.0

EOF
    exit 0
}

# Parse command line arguments
PYTHON_ARGS=""

while [[ $# -gt 0 ]]; do
	case $1 in
		-h|--help)
			show_help
			;;
		--train_split|--test_split|--val_split|--learning_rate)
			PYTHON_ARGS="$PYTHON_ARGS $1 $2"
			shift 2
			;;
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

# SLURM directives (only used when submitted via sbatch)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --mem=256G
#SBATCH --time=48:00:00

source .slurm

echo "=========================================="
echo "Job Information"
echo "=========================================="
if [ -n "$SLURM_JOB_ID" ]; then
	echo "Execution Mode: SLURM"
	echo "Job ID: $SLURM_JOB_ID"
	echo "Job Name: $SLURM_JOB_NAME"
	echo "Node: $SLURM_NODELIST"
	echo "Number of GPUs: $SLURM_GPUS_ON_NODE"
	echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
else
	echo "Execution Mode: Direct"
	echo "Hostname: $(hostname)"
	echo "Number of CPUs: $(nproc)"
	if command -v nvidia-smi &> /dev/null; then
		echo "Number of GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
	fi
fi

echo "=========================================="
echo "Start Time: $(date)"
echo "Python Arguments: $PYTHON_ARGS"
echo "=========================================="

mkdir -p $LOGS
module load $CONDA_MODULE
conda activate $CONDA_ENV
export TOKENIZERS_PARALLELISM=false
mkdir -p $HF_HOME

echo "Environment Information"
echo "=========================================="
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo "=========================================="

python src/training.py $PYTHON_ARGS

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
